"""FP16 MiniMax block-sparse attention and index scoring on Volta Tensor Cores.

The selected 128-token blocks and FP16 cache contents are unchanged. The
single-request caller supplies logical-to-physical slots and causal prefixes.
No dense history gather or full attention-score matrix is materialized.
"""

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl


def _check_device(q, *tensors):
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("MiniMax SM70 attention requires a Volta CUDA device")
    if any(t.device != q.device for t in tensors):
        raise ValueError("MiniMax SM70 attention tensors must share a CUDA device")


def _check_metadata(slots, prefix, lengths):
    if not (
        slots.ndim == 1
        and slots.numel() > 0
        and slots.dtype == prefix.dtype == lengths.dtype == torch.int32
        and prefix.shape == lengths.shape == (1,)
        and all(t.is_contiguous() for t in (slots, prefix, lengths))
    ):
        raise ValueError(
            "MiniMax SM70 requires contiguous single-request INT32 metadata"
        )


@tilelang.jit(
    compile_flags=["--fmad=false"],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _index_scores(heads):
    rows, capacity, context, blocks = (
        T.dynamic(x) for x in ("rows", "capacity", "context", "blocks")
    )

    @T.prim_func
    def score(
        Q: T.Tensor([rows, heads, 128], T.float16),
        K: T.Tensor([capacity, 1, 128], T.float16),
        Map: T.Tensor([context], T.int32),
        Prefix: T.Tensor([1], T.int32),
        Length: T.Tensor([1], T.int32),
        Scores: T.Tensor([heads, rows, blocks], T.float32),
    ):
        with T.Kernel(T.ceildiv(rows, 16), blocks, heads, threads=128) as (qt, kb, h):
            qs = T.alloc_shared([16, 128], T.float16)
            ks = T.alloc_shared([128, 128], T.float16)
            dots = T.alloc_fragment([16, 128], T.float32)
            maxima = T.alloc_fragment([16], T.float32)
            for i, d in T.Parallel(16, 128):
                qs[i, d] = T.if_then_else(qt * 16 + i < rows, Q[qt * 16 + i, h, d], 0)
            for n, d in T.Parallel(128, 128):
                pos = kb * 128 + n
                slot = T.if_then_else(pos < Length[0], Map[pos], 0)
                ks[n, d] = T.if_then_else(pos < Length[0], K[slot, 0, d], 0)
            T.clear(dots)
            T.gemm(qs, ks, dots, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            for i, n in T.Parallel(16, 128):
                dots[i, n] = T.if_then_else(
                    (kb * 128 + n <= Prefix[0] + qt * 16 + i)
                    & (kb * 128 + n < Length[0]),
                    dots[i, n] * (128**-0.5 * 1.4426950408889634),
                    -T.infinity(T.float32),
                )
            T.reduce_max(dots, maxima, dim=1)
            for i in T.Parallel(16):
                if qt * 16 + i < rows:
                    Scores[h, qt * 16 + i, kb] = maxima[i]

    return score


@tilelang.jit(
    compile_flags=["--fmad=false"],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _sparse_attention(
    heads, kv_heads, topk, splits, tile_k, left_window=-1, causal=True
):
    rows, capacity, context = (T.dynamic(x) for x in ("rows", "capacity", "context"))
    group = heads // kv_heads
    assert group in (8, 16)

    @T.prim_func
    def attend(
        Q: T.Tensor([rows, heads, 128], T.float16),
        K: T.Tensor([capacity, kv_heads, 128], T.float16),
        V: T.Tensor([capacity, kv_heads, 128], T.float16),
        Map: T.Tensor([context], T.int32),
        Prefix: T.Tensor([1], T.int32),
        Length: T.Tensor([1], T.int32),
        Blocks: T.Tensor([kv_heads, rows, topk], T.int32),
        Partials: T.Tensor([rows, splits, heads, 128], T.float32),
        LSE: T.Tensor([rows, splits, heads], T.float32),
    ):
        with T.Kernel(rows, kv_heads, splits, threads=128) as (row, kh, split):
            qs = T.alloc_shared([16, 128], T.float16)
            ks = T.alloc_shared([tile_k, 128], T.float16)
            vs = T.alloc_shared([tile_k, 128], T.float16)
            ps = T.alloc_shared([16, tile_k], T.float16)
            dots = T.alloc_fragment([16, tile_k], T.float32)
            accum = T.alloc_fragment([16, 128], T.float32)
            maximum = T.alloc_fragment([16], T.float32)
            tile_max = T.alloc_fragment([16], T.float32)
            denom = T.alloc_fragment([16], T.float32)
            tile_sum = T.alloc_fragment([16], T.float32)
            alpha = T.alloc_fragment([16], T.float32)
            T.fill(maximum, -T.infinity(T.float32))
            T.clear(denom)
            T.clear(accum)
            for h, d in T.Parallel(16, 128):
                qs[h, d] = T.if_then_else(h < group, Q[row, kh * group + h, d], 0)
            for step in T.serial(T.ceildiv(topk * (128 // tile_k), splits)):
                tile = step * splits + split
                block = T.if_then_else(
                    tile < topk * (128 // tile_k),
                    Blocks[kh, row, tile // (128 // tile_k)],
                    -1,
                )
                if (block >= 0) & (block * 128 < Length[0]):
                    for n, d in T.Parallel(tile_k, 128):
                        pos = block * 128 + (tile % (128 // tile_k)) * tile_k + n
                        slot = T.if_then_else(pos < Length[0], Map[pos], 0)
                        ks[n, d] = T.if_then_else(pos < Length[0], K[slot, kh, d], 0)
                        vs[n, d] = T.if_then_else(pos < Length[0], V[slot, kh, d], 0)
                    T.clear(dots)
                    T.gemm(
                        qs, ks, dots, transpose_B=True, policy=T.GemmWarpPolicy.FullCol
                    )
                    for h, n in T.Parallel(16, tile_k):
                        pos = block * 128 + (tile % (128 // tile_k)) * tile_k + n
                        visible = (pos < Length[0]) & T.if_then_else(
                            left_window < 0,
                            True,
                            pos >= Prefix[0] + row - left_window,
                        )
                        if causal:
                            visible = visible & (pos <= Prefix[0] + row)
                        dots[h, n] = T.if_then_else(
                            visible,
                            dots[h, n] * (128**-0.5 * 1.4426950408889634),
                            -T.infinity(T.float32),
                        )
                    T.reduce_max(dots, tile_max, dim=1)
                    for h in T.Parallel(16):
                        tile_max[h] = T.max(maximum[h], tile_max[h])
                        alpha[h] = T.if_then_else(
                            maximum[h] == -T.infinity(T.float32),
                            0,
                            T.exp2(maximum[h] - tile_max[h]),
                        )
                        maximum[h] = tile_max[h]
                    for h, n in T.Parallel(16, tile_k):
                        dots[h, n] = T.if_then_else(
                            maximum[h] == -T.infinity(T.float32),
                            0,
                            T.exp2(dots[h, n] - maximum[h]),
                        )
                    T.reduce_sum(dots, tile_sum, dim=1)
                    for h in T.Parallel(16):
                        denom[h] = denom[h] * alpha[h] + tile_sum[h]
                    for h, d in T.Parallel(16, 128):
                        accum[h, d] *= alpha[h]
                    T.copy(dots, ps)
                    T.gemm(ps, vs, accum, policy=T.GemmWarpPolicy.FullCol)
            for h, d in T.Parallel(group, 128):
                Partials[row, split, kh * group + h, d] = T.if_then_else(
                    denom[h] > 0, accum[h, d] / denom[h], 0
                )
            for h in T.Parallel(group):
                LSE[row, split, kh * group + h] = T.if_then_else(
                    denom[h] > 0, maximum[h] + T.log2(denom[h]), -T.infinity(T.float32)
                )

    return attend


@triton.jit
def _merge(P, L, O, HEADS: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, BS)
    d = tl.arange(0, 128)
    lse = tl.load(L + (row * SPLITS + s) * HEADS + head, s < SPLITS, float("-inf"))
    largest = tl.max(lse, 0)
    w = tl.where(largest == float("-inf"), 0.0, tl.exp2(lse - largest))
    denom = tl.sum(w, 0)
    p = tl.load(
        P + ((row * SPLITS + s[:, None]) * HEADS + head) * 128 + d[None, :],
        s[:, None] < SPLITS,
        0,
    )
    value = tl.sum(p * w[:, None], 0) / tl.maximum(denom, 1e-30)
    tl.store(O + (row * HEADS + head) * 128 + d, value)


def minimax_index_scores_sm70(q, keys, slots, prefix, lengths, num_blocks):
    """Causal per-query max score for each 128-token block, in base2 units."""
    _check_device(q, keys, slots, prefix, lengths)
    _check_metadata(slots, prefix, lengths)
    if not (
        q.dtype == keys.dtype == torch.float16
        and q.ndim == keys.ndim == 3
        and q.shape[0] > 0
        and q.shape[1] in (1, 2, 4)
        and q.shape[-1] == 128
        and keys.shape[1:] == (1, 128)
        and keys.is_contiguous()
        and 0 < num_blocks <= triton.cdiv(slots.numel(), 128)
    ):
        raise ValueError(
            "SM70 MiniMax index scoring requires single-request FP16 D128 caches"
        )
    output = torch.empty(
        (q.shape[1], q.shape[0], num_blocks), device=q.device, dtype=torch.float32
    )
    _index_scores(q.shape[1])(q.contiguous(), keys, slots, prefix, lengths, output)
    return output


def minimax_sparse_attention_sm70(
    q, k, v, slots, prefix, lengths, blocks, *, left_window=-1, causal=True
):
    """Selected-block FP16 GQA with FP32 accumulation and split-K normalizers."""
    _check_device(q, k, v, slots, prefix, lengths, blocks)
    _check_metadata(slots, prefix, lengths)
    if not (
        q.dtype == k.dtype == v.dtype == torch.float16
        and q.ndim == k.ndim == v.ndim == blocks.ndim == 3
        and q.shape[0] > 0
        and q.shape[-1] == 128
        and k.shape == v.shape
        and k.shape[-1] == 128
        and k.shape[1] > 0
        and q.shape[1] % k.shape[1] == 0
        and q.shape[1] // k.shape[1] in (8, 16)
        and blocks.shape[:2] == (k.shape[1], q.shape[0])
        and 0 < blocks.shape[2] <= 33
        and blocks.dtype == torch.int32
        and k.is_contiguous()
        and v.is_contiguous()
        and -1 <= left_window <= 4096
    ):
        raise ValueError("SM70 MiniMax attention requires single-request FP16 GQA D128")
    rows, heads, _ = q.shape
    splits, tile_k = (32, 64) if rows <= 8 else (1, 64)
    partials = torch.empty(
        (rows, splits, heads, 128), device=q.device, dtype=torch.float32
    )
    lse = torch.empty((rows, splits, heads), device=q.device, dtype=torch.float32)
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _sparse_attention(
        heads, k.shape[1], blocks.shape[2], splits, tile_k, left_window, causal
    )(q.contiguous(), k, v, slots, prefix, lengths, blocks.contiguous(), partials, lse)
    _merge[(rows, heads)](
        partials,
        lse,
        output,
        heads,
        splits,
        triton.next_power_of_2(splits),
        num_warps=4,
    )
    return output


@triton.jit
def _pack_window(
    K,
    V,
    KC,
    VC,
    Ids,
    Ptr,
    PK,
    PV,
    Prefix,
    Length,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    WINDOW: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    KCS: tl.constexpr,
    VCS: tl.constexpr,
    B: tl.constexpr,
):
    begin = tl.load(Ptr)
    count = tl.load(Ptr + 1) - begin
    keep = tl.minimum(count, WINDOW)
    off = tl.program_id(0) * B + tl.arange(0, B)
    row, feature = off // (HEADS * 128), off % (HEADS * 128)
    prefix = row < keep
    active = row < keep + ROWS
    idx = tl.load(Ids + begin + count - keep + row, active & prefix, 0).to(tl.int64)
    a = tl.load(KC + idx * KCS + feature, active & prefix, 0)
    b = tl.load(VC + idx * VCS + feature, active & prefix, 0)
    a = tl.where(
        prefix, a, tl.load(K + (row - keep) * KS + feature, active & ~prefix, 0)
    )
    b = tl.where(
        prefix, b, tl.load(V + (row - keep) * VS + feature, active & ~prefix, 0)
    )
    tl.store(PK + off, a, active)
    tl.store(PV + off, b, active)
    if tl.program_id(0) == 0:
        tl.store(Prefix, keep)
        tl.store(Length, keep + ROWS)


def minimax_window_attention_sm70(
    q, k, v, kc, vc, ids, indptr, left_window, out=None, *, causal=False
):
    """MiniMax draft block with the original per-query left and right bounds.

    Gather only a bounded window, once, then share each FP16 K/V tile across
    the local GQA heads on Volta Tensor Cores. Target sparse KV is untouched.
    """
    _check_device(q, k, v, kc, vc, ids, indptr, *(() if out is None else (out,)))
    if not (
        q.dtype == k.dtype == v.dtype == kc.dtype == vc.dtype == torch.float16
        and q.ndim == k.ndim == v.ndim == kc.ndim == vc.ndim == 3
        and 1 <= q.shape[0] <= 8
        and q.shape[1:] in ((8, 128), (16, 128))
        and k.shape == v.shape == (q.shape[0], 1, 128)
        and kc.shape == vc.shape
        and kc.shape[1:] == (1, 128)
        and indptr.shape == (2,)
        and indptr.dtype == torch.int32
        and ids.dtype in (torch.int32, torch.int64)
        and ids.ndim == 1
        and ids.is_contiguous()
        and indptr.is_contiguous()
        and all(t.stride(-1) == 1 for t in (k, v, kc, vc))
        and (out is None or (out.shape == q.shape and out.dtype == q.dtype))
        and 0 < left_window <= 4096
    ):
        raise ValueError("Unsupported SM70 MiniMax draft window layout")
    rows = q.shape[0]
    capacity = left_window + rows
    packed_k = torch.empty((capacity, 1, 128), device=q.device, dtype=q.dtype)
    packed_v = torch.empty_like(packed_k)
    prefix = torch.empty((1,), device=q.device, dtype=torch.int32)
    lengths = torch.empty_like(prefix)
    _pack_window[(triton.cdiv(capacity * 128, 512),)](
        k,
        v,
        kc,
        vc,
        ids,
        indptr,
        packed_k,
        packed_v,
        prefix,
        lengths,
        rows,
        1,
        left_window,
        k.stride(0),
        v.stride(0),
        kc.stride(0),
        vc.stride(0),
        512,
    )
    slots = torch.arange(capacity, device=q.device, dtype=torch.int32)
    block_ids = torch.arange(
        triton.cdiv(capacity, 128), device=q.device, dtype=torch.int32
    )
    blocks = block_ids[None, None, :].expand(1, rows, -1).contiguous()
    result = minimax_sparse_attention_sm70(
        q,
        packed_k,
        packed_v,
        slots,
        prefix,
        lengths,
        blocks,
        left_window=left_window,
        causal=causal,
    )
    if out is not None:
        out.copy_(result)
        return out
    return result
