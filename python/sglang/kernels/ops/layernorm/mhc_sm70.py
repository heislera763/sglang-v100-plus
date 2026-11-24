"""FP16 mHC pre-mix with an FP32 projection and fused Sinkhorn finalization.

Volta has neither BF16 tensor cores nor TF32. The default learned projection
uses FP32 cuBLAS. An opt-in small-batch path fuses FP32 projection and squared
norm partials, then reduces them with the gates and Sinkhorn finalization.
Another opt-in path retains both reference reductions and fuses only the
surrounding pointwise operations.
The residual reduction shares the established mHC combine kernel.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.layernorm.mhc import _mhc_combine_kernel


@cache_once
def _prepare_shard_module():
    return load_jit(
        "sm70_mhc_prepare",
        cuda_files=["layernorm/sm70_mhc_prepare.cuh"],
        cuda_wrappers=[("run", "sm70_mhc_prepare")],
    )


@triton.jit
def _cast_square_kernel(X, FLOAT_X, SQUARE, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < SIZE, 0).to(tl.float32)
    tl.store(FLOAT_X + i, x, i < SIZE)
    tl.store(SQUARE + i, x * x, i < SIZE)


@triton.jit
def _project_rms_kernel(
    RESIDUAL,
    FN,
    PARTIALS,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Independent FP32 projection and squared-norm partials per decode row."""
    split = tl.program_id(0)
    group = tl.program_id(1)
    row = tl.program_id(2)
    RESIDUAL += row * K
    PARTIALS += row * SPLITS * 25
    k = split * BLOCK_K + tl.arange(0, BLOCK_K)
    n = group * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(RESIDUAL + k, k < K, 0).to(tl.float32)
    w = tl.load(
        FN + n[:, None] * K + k[None, :], (n[:, None] < 24) & (k[None, :] < K), 0
    )
    projection = tl.sum(w * x[None, :], 1)
    tl.store(PARTIALS + split * 25 + n, projection, n < 24)
    if group == 0:
        tl.store(PARTIALS + split * 25 + 24, tl.sum(x * x, 0))


@triton.jit
def _finalize_kernel(
    MIXES,
    RMS,
    SCALE,
    BASE,
    PRE,
    POST,
    COMB,
    HC: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    ITERS: tl.constexpr,
    PROJECTION_SPLITS: tl.constexpr = 0,
    PROJECTION_K: tl.constexpr = 0,
    RMS_EPS: tl.constexpr = 0.0,
    RMS_FROM_MEAN: tl.constexpr = False,
):
    row = tl.program_id(0)
    j = tl.arange(0, HC)
    jj, kk = j[:, None], j[None, :]
    if PROJECTION_SPLITS:
        MIXES += row * PROJECTION_SPLITS * 25
        split = tl.arange(0, PROJECTION_SPLITS)
        valid = split < PROJECTION_SPLITS
        squares = tl.load(MIXES + split * 25 + 24, valid, 0)
        rms = tl.rsqrt(tl.sum(squares, 0) / PROJECTION_K + RMS_EPS)
        pre_raw = (
            tl.sum(
                tl.load(MIXES + split[:, None] * 25 + j[None, :], valid[:, None], 0), 0
            )
            * rms
        )
        post_raw = (
            tl.sum(
                tl.load(
                    MIXES + split[:, None] * 25 + HC + j[None, :], valid[:, None], 0
                ),
                0,
            )
            * rms
        )
        comb_raw = (
            tl.sum(
                tl.load(
                    MIXES
                    + split[:, None, None] * 25
                    + 2 * HC
                    + jj[None, :, :] * HC
                    + kk[None, :, :],
                    valid[:, None, None],
                    0,
                ),
                0,
            )
            * rms
        )
    else:
        mix = MIXES + row * (2 + HC) * HC
        rms = tl.load(RMS + row)
        if RMS_FROM_MEAN:
            rms = tl.rsqrt(rms + RMS_EPS)
        pre_raw = tl.load(mix + j) * rms
        post_raw = tl.load(mix + HC + j) * rms
        comb_raw = tl.load(mix + 2 * HC + jj * HC + kk) * rms
    pre = tl.sigmoid(pre_raw * tl.load(SCALE) + tl.load(BASE + j)) + PRE_EPS
    post = POST_MULT * tl.sigmoid(
        post_raw * tl.load(SCALE + 1) + tl.load(BASE + HC + j)
    )
    comb = comb_raw * tl.load(SCALE + 2) + tl.load(BASE + 2 * HC + jj * HC + kk)
    comb = tl.exp(comb - tl.max(comb, 1)[:, None])
    comb = comb / tl.sum(comb, 1)[:, None] + SINK_EPS
    comb = comb / (tl.sum(comb, 0)[None, :] + SINK_EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, 1)[:, None] + SINK_EPS)
        comb = comb / (tl.sum(comb, 0)[None, :] + SINK_EPS)
    tl.store(PRE + row * HC + j, pre)
    tl.store(POST + row * HC + j, post)
    tl.store(COMB + row * HC * HC + jj * HC + kk, comb)


def mhc_pre_sm70(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    *,
    fuse_projection=False,
    fuse_pointwise=False,
    eps_outside_sqrt=False,
):
    """Return (FP32 post gates, FP32 residual mixing, FP16 layer input).

    HC=4 contiguous FP16 residuals and contiguous FP32 parameters are required.
    Epsilons and the post multiplier retain the torch fallback's semantics.
    ``fuse_projection`` supports (M, HC, H) = (1..8, 4, 4096); its split
    FP32 summation has a different reduction order from cuBLAS/torch.
    Rows retain the batch-one reduction, including speculative verification.
    ``fuse_pointwise`` retains cuBLAS projection and torch mean reduction,
    fusing only cast/square and epsilon/rsqrt operations for the same shape.
    Bulk (M>=1024, H=4096) reuses the exact cast/mean preparation from the
    sharded path; the full FP32 projection and all finalization remain intact.
    """
    assert residual.ndim == 3 and residual.shape[1] == 4
    assert residual.dtype == torch.float16 and residual.is_contiguous()
    assert all(
        t.dtype == torch.float32 and t.is_contiguous() for t in (fn, hc_scale, hc_base)
    )
    m, hc, h = residual.shape
    assert fn.shape == (24, hc * h)
    assert hc_scale.numel() == 3 and hc_base.numel() == 24
    assert sinkhorn_repeat >= 1
    assert not fuse_projection or (1 <= m <= 8 and h == 4096)
    assert not fuse_pointwise or (m == 1 and h == 4096)
    assert not (eps_outside_sqrt and (fuse_projection or fuse_pointwise))
    assert not (fuse_projection and fuse_pointwise)
    pre = torch.empty((m, hc), dtype=torch.float32, device=residual.device)
    post = torch.empty_like(pre)
    comb = torch.empty((m, hc, hc), dtype=torch.float32, device=residual.device)
    layer_input = torch.empty((m, h), dtype=residual.dtype, device=residual.device)
    if m:
        splits = 0
        if fuse_projection:
            splits = triton.cdiv(hc * h, 1024)
            mixes = torch.empty(
                (m, splits, 25), dtype=torch.float32, device=residual.device
            )
            rms = mixes  # The fused finalizer reads norm partials from mixes.
            _project_rms_kernel[(splits, 6, m)](
                residual,
                fn,
                mixes,
                K=hc * h,
                BLOCK_K=1024,
                BLOCK_N=4,
                SPLITS=splits,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif fuse_pointwise:
            x = torch.empty((m, hc * h), dtype=torch.float32, device=residual.device)
            square = torch.empty_like(x)
            _cast_square_kernel[(triton.cdiv(m * hc * h, 1024),)](
                residual,
                x,
                square,
                SIZE=m * hc * h,
                BLOCK=1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
            # Identical input layout and reduction implementation to the default.
            rms = square.mean(-1)
            mixes = F.linear(x, fn)
        elif m >= 1024 and h == 4096:
            x = torch.empty((m, 16384), dtype=torch.float32, device=residual.device)
            mean = torch.empty(m, dtype=torch.float32, device=residual.device)
            _prepare_shard_module().run(residual, x, mean, 0)
            mixes = F.linear(x, fn)
            rms = (mean.sqrt() + rms_eps).reciprocal() if eps_outside_sqrt else torch.rsqrt(mean + rms_eps)
        else:
            x = residual.view(m, hc * h).float()
            mixes = F.linear(x, fn)
            # FP16 -> FP32 owns a private copy. Enqueue projection first on
            # the same stream, then reuse that copy for the identical square
            # and mean reduction instead of allocating another full tensor.
            mean = x.square_().mean(-1)
            rms = (mean.sqrt() + rms_eps).reciprocal() if eps_outside_sqrt else torch.rsqrt(mean + rms_eps)
        _finalize_kernel[(m,)](
            mixes,
            rms,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            HC=hc,
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            ITERS=sinkhorn_repeat,
            PROJECTION_SPLITS=splits,
            PROJECTION_K=hc * h,
            RMS_EPS=rms_eps,
            RMS_FROM_MEAN=fuse_pointwise,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _mhc_combine_kernel[(m, triton.cdiv(h, 256))](
            residual,
            pre,
            layer_input,
            h,
            hc * h,
            hc,
            1,
            h,
            HC=hc,
            BLOCK_H=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return post.unsqueeze(-1), comb, layer_input


def mhc_pre_shard_sm70(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    *,
    rank,
    size=4,
):
    """One token shard of bulk FP16 mHC PRE, padded for equal local gathers.

    Keep the FP32 cuBLAS projection's full matrix shape: a quarter-row GEMM
    changes its reduction order and can move final Half outputs by two ULPs.
    Fuse the full cast with each shard's squared-norm reduction, retaining the
    contiguous torch mean's FP32 summation order. Partition gates, Sinkhorn and
    residual mixing. Neither the FP16 residual nor its private FP32 cast is
    modified by the statistics calculation.
    Padding has defined zero outputs and is removed by the integration adapter.
    """
    assert residual.ndim == 3 and residual.shape[1:] == (4, 4096)
    assert residual.dtype == torch.float16 and residual.is_contiguous()
    assert all(
        t.dtype == torch.float32 and t.is_contiguous() for t in (fn, hc_scale, hc_base)
    )
    assert fn.shape == (24, 16384)
    assert hc_scale.numel() == 3 and hc_base.numel() == 24
    assert sinkhorn_repeat >= 1 and size == 4 and 0 <= rank < size
    rows = residual.shape[0]
    assert rows == 0 or rows >= 1024, "SM70 mHC shards are for bulk prefill only"
    local_rows = triton.cdiv(rows, size)
    start, stop = min(rank * local_rows, rows), min((rank + 1) * local_rows, rows)
    valid_rows = stop - start
    alloc = torch.zeros if valid_rows < local_rows else torch.empty
    pre = torch.empty((local_rows, 4), dtype=torch.float32, device=residual.device)
    post = alloc((local_rows, 4, 1), dtype=torch.float32, device=residual.device)
    comb = alloc((local_rows, 4, 4), dtype=torch.float32, device=residual.device)
    layer_input = alloc((local_rows, 4096), dtype=torch.float16, device=residual.device)
    if valid_rows:
        x = torch.empty((rows, 16384), device=residual.device, dtype=torch.float32)
        mean = torch.empty(valid_rows, device=residual.device, dtype=torch.float32)
        _prepare_shard_module().run(residual, x, mean, start)
        mixes = F.linear(x, fn)[start:stop]
        rms = torch.rsqrt(mean + rms_eps)
        _finalize_kernel[(valid_rows,)](
            mixes,
            rms,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            HC=4,
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            ITERS=sinkhorn_repeat,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _mhc_combine_kernel[(valid_rows, 16)](
            residual[start:stop],
            pre,
            layer_input,
            4096,
            16384,
            4,
            1,
            4096,
            HC=4,
            BLOCK_H=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return post, comb, layer_input
