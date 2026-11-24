"""Volta HCA over separate FP16 window and compressed-history banks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _accumulate(
    Q,
    Cache,
    Ids,
    row,
    width: tl.constexpr,
    length,
    scale,
    maximum,
    denom,
    acc,
    D: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    d = tl.arange(0, D)
    for start in range(tl.cdiv(length, BK)):
        j = start * BK + tl.arange(0, BK)
        ids = tl.load(Ids + row * width + j, (j < width) & (j < length), -1)
        keys = tl.load(
            Cache + ids[:, None].to(tl.int64) * D + d[None, :], ids[:, None] >= 0, 0
        )
        scores = tl.dot(Q, tl.trans(keys)) * scale
        scores = tl.where(ids[None, :] >= 0, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        rescale = tl.exp(maximum - new_max)
        prob = tl.exp(scores - new_max[:, None])
        denom = denom * rescale + tl.sum(prob, 1)
        acc = acc * rescale[:, None] + tl.dot(prob.to(tl.float16), keys)
        maximum = new_max
    return maximum, denom, acc


@triton.jit
def _hca(
    Q,
    SW,
    CU,
    SI,
    CI,
    SL,
    CL,
    Sink,
    O,
    scale,
    H: tl.constexpr,
    WS: tl.constexpr,
    WC: tl.constexpr,
    D: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, D)
    q = tl.load(Q + (row * H + h[:, None]) * D + d[None, :], h[:, None] < H, 0)
    maximum = tl.load(Sink + h, h < H, 0).to(tl.float32)
    denom = tl.full((BH,), 1, tl.float32)
    acc = tl.full((BH, D), 0, tl.float32)
    maximum, denom, acc = _accumulate(
        q,
        SW,
        SI,
        row,
        WS,
        tl.minimum(WS, tl.load(SL + row)),
        scale,
        maximum,
        denom,
        acc,
        D,
        BH,
        BK,
    )
    if WC:
        maximum, denom, acc = _accumulate(
            q,
            CU,
            CI,
            row,
            WC,
            tl.minimum(WC, tl.load(CL + row)),
            scale,
            maximum,
            denom,
            acc,
            D,
            BH,
            BK,
        )
    tl.store(
        O + (row * H + h[:, None]) * D + d[None, :],
        acc / denom[:, None],
        h[:, None] < H,
    )


def hca_fp16(
    q, window, window_ids, window_lens, compressed, comp_ids, comp_lens, sink, scale
):
    rows, heads, dim = q.shape
    assert dim == 512 and q.dtype == window.dtype == torch.float16
    q = q.contiguous()
    window_ids = window_ids.reshape(rows, -1).contiguous()
    output = torch.empty_like(q)
    width = 0
    if compressed is not None:
        assert compressed.dtype == q.dtype
        comp_ids = comp_ids.reshape(rows, -1).contiguous()
        width = comp_ids.shape[1]
        if heads in (8, 16) and (rows >= 64 or width >= 256):
            from sglang.kernels.ops.attention.dsa.sm70_sparse_mla import (
                _online_selected,
            )

            return _online_selected(
                q,
                window.view(-1, dim),
                window_ids,
                window_lens.view(-1),
                sink,
                scale,
                output,
                extra=(compressed.view(-1, dim), comp_ids, comp_lens.view(-1)),
            )
    else:
        compressed, comp_ids, comp_lens = window, window_ids, window_lens
    _hca[(rows, triton.cdiv(heads, 16))](
        q,
        window,
        compressed,
        window_ids,
        comp_ids,
        window_lens,
        comp_lens,
        sink,
        output,
        scale,
        heads,
        window_ids.shape[1],
        width,
        dim,
        16,
        16,
    )
    return output


@triton.jit
def _store_compressed(X, Plan, Loc, W, Freq, Cache, eps, DECODE: tl.constexpr):
    row = tl.program_id(0)
    seq = tl.load(Plan + row * 4)
    if seq >= 128 and seq % 128 == 0:
        if DECODE:
            idx = row
        else:
            idx = tl.load(Plan + row * 4 + 1) & 65535
        slot = tl.load(Loc + idx).to(tl.int64)
        d = tl.arange(0, 512)
        val = tl.load(X + row * 512 + d).to(tl.float32)
        inv = tl.rsqrt(tl.sum(val * val, 0) / 512.0 + eps)
        val *= inv * tl.load(W + d).to(tl.float32)
        peer = tl.gather(val, d ^ 1, 0)
        freq = Freq + (seq - 128) * 64
        cos = tl.load(freq + ((d - 448) & 62), d >= 448, 1.0)
        sin = tl.load(freq + ((d - 448) & 62) + 1, d >= 448, 0.0)
        out = val * cos + tl.where((d & 1) == 0, -peer, peer) * sin
        tl.store(Cache + slot * 512 + d, out, slot >= 0)


def store_compressed_fp16(kv, plan, out_loc, norm, freqs, cache):
    if kv.shape[0]:
        _store_compressed[(kv.shape[0],)](
            kv,
            plan[1].view(torch.int32),
            out_loc,
            norm.weight,
            torch.view_as_real(freqs),
            cache,
            norm.variance_epsilon,
            plan.is_decode,
        )
