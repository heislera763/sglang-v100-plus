"""Small-batch E4M3 experts over the SM70 Marlin packed layout.

Split GEMV work across output and reduction tiles instead of padding one
activation batch to a tensor-core GEMM. Keep Marlin's FP16 dequantization, projection,
activation and weighted-expert boundaries; partial sums use FP32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(
    X,
    W,
    S,
    Ids,
    P,
    N: tl.constexpr,
    K: tl.constexpr,
    INPUT_PER_ROUTE: tl.constexpr,
    TOPK: tl.constexpr,
    ROWS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MACRO: tl.constexpr,
):
    route = tl.program_id(2)
    expert = tl.load(Ids + route)
    if expert >= 0:
        # Cover the four interleaved 64-column subtiles together. Adjacent
        # lanes now read adjacent packed words rather than every fourth word.
        lane = tl.arange(0, BN)
        macro = tl.program_id(0) // (MACRO // BN)
        sub = tl.program_id(0) % (MACRO // BN)
        n = macro * MACRO + (lane % 4) * 64 + sub * (BN // 4) + lane // 4
        k = tl.program_id(1) * BK + tl.arange(0, BK)
        group_tiles = MACRO // 64
        n_tile = n // 64
        word = (
            expert * (K // 16) * (N * 4)
            + (k[:, None] // 16) * (N * 4)
            + (n_tile[None, :] // group_tiles) * group_tiles * 256
            + ((k[:, None] % 16) * 16 + (n[None, :] % 64) // 4) * group_tiles
            + n_tile[None, :] % group_tiles
        )
        packed = tl.load(W + word).to(tl.uint32)
        shift = ((n % 4) // 2 + 2 * (n % 2)) * 8
        code = (packed >> shift[None, :]) & 255
        bits = ((code & 127) << 7) | ((code & 128) << 8)
        biased = bits.to(tl.uint16).to(tl.float16, bitcast=True)
        scale = tl.load(
            S + expert * (K // 128) * N + (k[:, None] // 128) * N + n[None, :]
        )
        weight = (biased.to(tl.float32) * scale.to(tl.float32)).to(tl.float16)
        if INPUT_PER_ROUTE:
            row = route
        else:
            row = route // TOPK if ROWS > 1 else 0
        x = tl.load(X + row * K + k).to(tl.float32)
        accum = tl.sum(x[:, None] * weight.to(tl.float32), axis=0)
        tl.store(P + (route * (K // BK) + tl.program_id(1)) * N + n, accum)


@triton.jit
def _activate(
    P,
    Ids,
    A,
    I: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BN: tl.constexpr,
    CLAMP: tl.constexpr,
):
    route = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    expert = tl.load(Ids + route)
    if expert >= 0:
        splits = tl.arange(0, BS)
        base = (route * SPLITS + splits[:, None]) * (2 * I) + n[None, :]
        mask = splits[:, None] < SPLITS
        gate = tl.sum(tl.load(P + base, mask, 0), axis=0).to(tl.float16).to(tl.float32)
        up = (
            tl.sum(tl.load(P + base + I, mask, 0), axis=0).to(tl.float16).to(tl.float32)
        )
        if CLAMP is not None and CLAMP > 0:
            gate = tl.minimum(gate, CLAMP)
            up = tl.clamp(up, -CLAMP, CLAMP)
        silu = gate / (1 + tl.exp(-gate))
        if CLAMP is not None:
            silu = silu.to(tl.float16).to(tl.float32)
        value = (silu * up).to(tl.float16)
        tl.store(A + route * I + n, value)


@triton.jit
def _combine(
    P,
    Ids,
    Router,
    Out,
    LoraAlpha,
    LoraB,
    LORA: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    row = tl.program_id(1)
    routes = tl.arange(0, BT)
    flat_routes = row * TOPK + routes
    splits = tl.arange(0, BS)
    experts = tl.load(Ids + flat_routes, routes < TOPK, -1)
    valid = (routes < TOPK) & (experts >= 0)
    p = tl.load(
        P
        + (flat_routes[:, None, None] * SPLITS + splits[None, :, None]) * H
        + n[None, None, :],
        valid[:, None, None] & (splits[None, :, None] < SPLITS),
        0,
    )
    down = tl.sum(p, axis=1)
    if LORA:
        alpha = tl.load(LoraAlpha + flat_routes, valid, 0).to(tl.float32)
        b = tl.load(LoraB + experts[:, None] * H + n[None, :], valid[:, None], 0).to(
            tl.float32
        )
        delta = alpha[:, None] * b
        down = (
            (down.to(tl.float16).to(tl.float32) + delta).to(tl.float16).to(tl.float32)
        )
    weights = tl.load(Router + flat_routes, valid, 0)
    weighted = (down * weights[:, None]).to(tl.float16).to(tl.float32)
    result = (tl.sum(weighted, axis=0) * SCALE).to(tl.float16)
    tl.store(Out + row * H + n, result)


def fp8_moe_decode(
    x,
    w13,
    w2,
    s13,
    s2,
    ids,
    weights,
    *,
    down_lora=None,
    clamp=None,
    routed_scale=1.0,
    reuse_buffers=False,
):
    """Small FP8 expert batches with local IDs and -1 for nonlocal routes."""
    from sglang.srt.runtime_context import get_buffer

    hidden = x.shape[-1]
    inner = w2.shape[1] * 16
    if (
        x.ndim != 2
        or not 1 <= x.shape[0] <= 4
        or hidden % 256
        or inner % 128
        or ids.ndim != 2
        or ids.shape[0] != x.shape[0]
        or not 1 <= ids.shape[1] <= 32
        or ids.dtype != torch.int32
        or weights.shape != ids.shape
        or weights.dtype != torch.float32
        or x.dtype != torch.float16
        or w13.dtype != torch.int32
        or w2.dtype != torch.int32
        or s13.dtype != torch.float16
        or s2.dtype != torch.float16
        or tuple(w13.shape[1:]) != (hidden // 16, 8 * inner)
        or tuple(w2.shape[1:]) != (inner // 16, 4 * hidden)
        or w2.shape[0] != w13.shape[0]
        or s13.shape != (w13.shape[0], hidden // 128, 2 * inner)
        or s2.shape != (w13.shape[0], inner // 128, hidden)
    ):
        raise ValueError("Unsupported SM70 FP8 expert decode layout")
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (7, 0):
        raise ValueError("FP8 expert decode requires SM70 CUDA")
    tensors = (x, w13, w2, s13, s2, ids, weights)
    if any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError("FP8 decode requires contiguous colocated tensors")
    topk = ids.shape[1]
    rows = x.shape[0]
    num_routes = rows * topk
    gate_splits, down_splits = hidden // 128, inner // 128

    def allocate():
        return (
            x.new_empty((num_routes, gate_splits, 2 * inner), dtype=torch.float32),
            x.new_empty((num_routes, inner)),
            x.new_empty((num_routes, down_splits, hidden), dtype=torch.float32),
        )

    # Share only when the caller has already established serialized execution.
    gate, activation, down = (
        get_buffer(
            f"v100_fp8_moe_decode:{x.device}:{rows}:{topk}:{hidden}:{inner}", allocate
        )
        if reuse_buffers
        else allocate()
    )
    out = torch.empty_like(x)
    _partial[(2 * inner // 64, gate_splits, num_routes)](
        x,
        w13,
        s13,
        ids,
        gate,
        2 * inner,
        hidden,
        False,
        topk,
        rows,
        64,
        128,
        256,
        num_warps=4,
    )
    _activate[(inner // 128, num_routes)](
        gate,
        ids,
        activation,
        inner,
        gate_splits,
        triton.next_power_of_2(gate_splits),
        128,
        clamp,
        num_warps=4,
    )
    alpha = b = None
    if down_lora is not None:
        from sglang.kernels.ops.lora.sm70_rank_one import rank_one_coefficients

        a, b = down_lora
        alpha = rank_one_coefficients(activation, a, ids)
    _partial[(hidden // 64, down_splits, num_routes)](
        activation,
        w2,
        s2,
        ids,
        down,
        hidden,
        inner,
        True,
        topk,
        rows,
        64,
        128,
        256,
        num_warps=4,
    )
    _combine[(hidden // 64, rows)](
        down,
        ids,
        weights,
        out,
        alpha,
        b,
        down_lora is not None,
        hidden,
        topk,
        down_splits,
        triton.next_power_of_2(down_splits),
        triton.next_power_of_2(topk),
        64,
        routed_scale,
        num_warps=4,
    )
    return out
