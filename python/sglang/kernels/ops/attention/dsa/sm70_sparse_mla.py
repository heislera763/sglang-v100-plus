"""FP16 selected-key MLA with a RoPE tail on Volta Tensor Cores."""

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _sparse_mla_reduce_kernel,
)

_LOG2E = 1.4426950408889634
_VALUE_DIM = 512
_HEAD_TILE = 16
_WINDOW_QUERIES = 16


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
)
def _attention(heads, qk_dim, splits, tile):
    rows, slots, width = (T.dynamic(x) for x in ("rows", "slots", "width"))
    d = _VALUE_DIM
    htile = _HEAD_TILE
    scratch = (rows, splits, htile) if splits > 1 else (1, 1, 1)
    partial_shape = (*scratch, d if splits > 1 else 1)

    @T.prim_func
    def attend(
        Q: T.Tensor([rows, heads, qk_dim], T.float16),
        KV: T.Tensor([slots, qk_dim], T.float16),
        IDs: T.Tensor([rows, width], T.int32),
        Length: T.Tensor([rows], T.int32),
        Sink: T.Tensor([heads], T.float32),
        Out: T.Tensor([rows, heads, d], T.float16),
        Partials: T.Tensor(partial_shape, T.float32),
        LSE: T.Tensor(scratch, T.float32),
        scale: T.float32,
    ):
        with T.Kernel(rows, splits, threads=64) as (row, split):
            q = T.alloc_shared((htile, qk_dim), T.float16)
            k = T.alloc_shared((tile, qk_dim), T.float16)
            v = T.alloc_shared((tile, d), T.float16)
            prob = T.alloc_shared((htile, tile), T.float16)
            selected = T.alloc_shared((tile,), T.int32)
            score = T.alloc_fragment((htile, tile), T.float32)
            acc = T.alloc_fragment((htile, d), T.float32)
            maximum = T.alloc_fragment((htile,), T.float32)
            denom = T.alloc_fragment((htile,), T.float32)
            tile_max = T.alloc_fragment((htile,), T.float32)
            tile_sum = T.alloc_fragment((htile,), T.float32)
            rescale = T.alloc_fragment((htile,), T.float32)

            n = T.min(width, T.max(0, Length[row]))
            num_tiles = T.ceildiv(n, tile)
            tiles_per_split = T.ceildiv(num_tiles, splits)
            begin = split * tiles_per_split
            end = T.min(begin + tiles_per_split, num_tiles)
            T.clear(acc)
            T.clear(denom)
            T.fill(maximum, -T.infinity(T.float32))
            for h, j in T.Parallel(htile, qk_dim):
                q[h, j] = T.if_then_else(h < heads, Q[row, h, j], 0)
            for step in T.serial(begin, end):
                for i in T.Parallel(tile):
                    col = step * tile + i
                    slot = T.if_then_else(col < n, IDs[row, col], -1)
                    selected[i] = T.if_then_else((slot >= 0) & (slot < slots), slot, -1)
                for i, j in T.Parallel(tile, qk_dim):
                    k[i, j] = T.if_then_else(selected[i] >= 0, KV[selected[i], j], 0)
                for i, j in T.Parallel(tile, d):
                    v[i, j] = k[i, j]
                T.clear(score)
                T.gemm(q, k, score, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for h, i in T.Parallel(htile, tile):
                    score[h, i] = T.if_then_else(
                        selected[i] >= 0, score[h, i] * scale, -T.infinity(T.float32)
                    )
                T.reduce_max(score, tile_max, dim=1)
                for h in T.Parallel(htile):
                    tile_max[h] = T.max(maximum[h], tile_max[h])
                    rescale[h] = T.if_then_else(
                        maximum[h] == -T.infinity(T.float32),
                        0,
                        T.exp2(maximum[h] - tile_max[h]),
                    )
                    maximum[h] = tile_max[h]
                for h, i in T.Parallel(htile, tile):
                    score[h, i] = T.if_then_else(
                        maximum[h] == -T.infinity(T.float32),
                        0,
                        T.exp2(score[h, i] - maximum[h]),
                    )
                T.reduce_sum(score, tile_sum, dim=1)
                for h in T.Parallel(htile):
                    denom[h] = denom[h] * rescale[h] + tile_sum[h]
                for h, j in T.Parallel(htile, d):
                    acc[h, j] *= rescale[h]
                T.copy(score, prob)
                T.gemm(prob, v, acc, policy=T.GemmWarpPolicy.FullCol)
            if splits == 1:
                for h, j in T.Parallel(heads, d):
                    Out[row, h, j] = T.if_then_else(
                        denom[h] > 0,
                        acc[h, j] / (denom[h] + T.exp2(Sink[h] * _LOG2E - maximum[h])),
                        0,
                    )
            else:
                for h, j in T.Parallel(heads, d):
                    Partials[row, split, h, j] = T.if_then_else(
                        denom[h] > 0, acc[h, j] / denom[h], 0
                    )
                for h in T.Parallel(heads):
                    LSE[row, split, h] = T.if_then_else(
                        denom[h] > 0, maximum[h] + T.log2(denom[h]), -3.402823466e38
                    )

    return attend


@triton.jit
def _window_union_ids(
    indices,
    lengths,
    union,
    WIDTH: tl.constexpr,
    BANK: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP: tl.constexpr,
):
    group = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    row = group * GROUP
    first_len = tl.minimum(tl.maximum(tl.load(lengths + row), 0), WIDTH)
    first = tl.load(indices + row * WIDTH + col, col < first_len, -1)
    extra = col - first_len + 1
    tail = tl.load(
        indices + (row + extra) * WIDTH + tl.minimum(first_len + extra, WIDTH) - 1,
        (extra >= 1) & (extra < GROUP),
        -1,
    )
    tl.store(
        union + group * BANK + col, tl.where(col < first_len, first, tail), col < BANK
    )


@triton.jit
def _sink_softmax(
    logits,
    indices,
    lengths,
    sinks,
    probs,
    num_slots,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    WINDOW: tl.constexpr,
    GROUP: tl.constexpr,
    scale,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    slot = tl.load(indices + row * WIDTH + col, col < WIDTH, -1)
    if WINDOW:
        last = tl.minimum(tl.load(lengths + row * GROUP), WINDOW) + head // HEADS
        live = (col >= last - WINDOW) & (col < last)
        base = (row * (GROUP * HEADS) + head) * WIDTH
    else:
        live = col < tl.load(lengths + row)
        base = (row * HEADS + head) * WIDTH
    valid = (slot >= 0) & (slot < num_slots) & live
    offset = base + col
    score = tl.load(logits + offset, valid, 0) * scale
    score = tl.where(valid, score, -float("inf"))
    sink = tl.load(sinks + head % HEADS)
    max_score = tl.maximum(sink, tl.max(score, 0))
    numerator = tl.exp(score - max_score)
    denominator = tl.sum(numerator, 0) + tl.exp(sink - max_score)
    tl.store(probs + offset, numerator / denominator, col < WIDTH)


def _prefill(q, kv, indices, lengths, sink, scale, output, ordered_window=False):
    # Gather bounded batches of selected keys for CUDA Tensor Core GEMMs.
    # Sequential queries in an ordered window can share one key bank.
    rows, heads, dim = q.shape
    width = indices.shape[1]
    group_size = _WINDOW_QUERIES if ordered_window else 1
    bank_width = (
        triton.cdiv(width + group_size - 1, 16) * 16 if ordered_window else width
    )
    chunk_groups = min(
        128, max(1, (128 << 20) // (bank_width * dim * q.element_size()))
    )
    num_groups = rows // group_size
    for group in range(0, num_groups, chunk_groups):
        start = group * group_size
        groups = min(chunk_groups, num_groups - group)
        end = start + groups * group_size
        counts = lengths[start:end]
        ids = indices[start:end]
        if ordered_window:
            union = ids.new_empty((groups, bank_width))
            _window_union_ids[(groups,)](
                ids,
                counts,
                union,
                WIDTH=width,
                BANK=bank_width,
                BLOCK=triton.next_power_of_2(bank_width),
                GROUP=group_size,
            )
            ids = union
        bank = kv.index_select(0, ids.reshape(-1).clamp(0, kv.shape[0] - 1))
        bank = bank.view(groups, bank_width, dim)
        query = q[start:end].reshape(groups, group_size * heads, dim)
        logits = torch.bmm(query, bank.transpose(1, 2), out_dtype=torch.float32)
        probs = q.new_empty(logits.shape)
        _sink_softmax[(groups, group_size * heads)](
            logits,
            ids,
            counts,
            sink,
            probs,
            kv.shape[0],
            HEADS=heads,
            WIDTH=bank_width,
            BLOCK=triton.next_power_of_2(bank_width),
            WINDOW=width if ordered_window else 0,
            GROUP=group_size,
            scale=scale,
        )
        output[start:end].copy_(
            torch.bmm(probs, bank[..., :_VALUE_DIM], out_dtype=torch.float32).view(
                end - start, heads, _VALUE_DIM
            )
        )
        del bank
    tail = num_groups * group_size
    if tail < rows:
        _prefill(
            q[tail:], kv, indices[tail:], lengths[tail:], sink, scale, output[tail:]
        )
    return output


def sparse_mla_sm70(
    q, kv, indices, lengths, sink, scale, *, split=False, ordered_window=False
):
    """Selected physical IDs, FP16 Q/KV and one FP32 sink logit per head.

    The RoPE tail participates in QK, while V uses the first 512 latent
    features. Negative and out-of-cache IDs are masked. For decode, all
    splits write their own partials and the sink is applied once in the merge.
    Ordered-window prefill requires consecutive queries in one causal sequence
    and left-packed keys from oldest to newest, with width equal to the window.
    """
    if kv.ndim == 3 and kv.shape[1] == 1:
        kv = kv.squeeze(1)
    if q.ndim != 3 or kv.ndim != 2 or indices.ndim != 2:
        raise ValueError("Expected Q[T,H,D], KV[S,D] and selected IDs[T,K]")
    rows, heads, qk_dim = q.shape
    if not (
        q.is_cuda
        and torch.cuda.get_device_capability(q.device) == (7, 0)
        and heads in (4, 8, 16)
        and qk_dim in (512, 576)
        and kv.shape[1] == qk_dim
        and q.dtype == kv.dtype == torch.float16
        and kv.is_contiguous()
        and indices.dtype == torch.int32
        and indices.shape[0] == rows
        and indices.shape[1] > 0
        and lengths.shape == (rows,)
        and lengths.dtype == torch.int32
        and sink.shape == (heads,)
        and sink.dtype == torch.float32
        and all(x.device == q.device for x in (kv, indices, lengths, sink))
    ):
        raise ValueError("Unsupported SM70 selected-key MLA geometry")
    output = q.new_empty((rows, heads, _VALUE_DIM))
    if not rows:
        return output
    q = q.contiguous()
    indices = indices.contiguous()
    lengths = lengths.contiguous()
    sink = sink.contiguous()
    if not split:
        return _prefill(q, kv, indices, lengths, sink, scale, output, ordered_window)
    tile = 32
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    splits = min(
        triton.cdiv(indices.shape[1], tile), max(1, triton.cdiv(sm_count, rows))
    )
    scratch_shape = (rows, splits, _HEAD_TILE) if splits > 1 else (1, 1, 1)
    scratch = q.new_empty(
        (*scratch_shape, _VALUE_DIM if splits > 1 else 1), dtype=torch.float32
    )
    lse = q.new_empty(scratch_shape, dtype=torch.float32)
    _attention(heads, qk_dim, splits, tile)(
        q,
        kv,
        indices,
        lengths,
        sink,
        output,
        scratch,
        lse,
        scale * _LOG2E,
    )
    if splits > 1:
        _sparse_mla_reduce_kernel[(rows, heads, _VALUE_DIM // 64)](
            lse,
            scratch,
            output,
            H=heads,
            D_V=_VALUE_DIM,
            KV_SPLITS=splits,
            ACTIVE_SPLITS=splits,
            ACTIVE_SPLITS_POW2=triton.next_power_of_2(splits),
            D_CHUNK=64,
            BLOCK_K=tile,
            sink_ptr=sink,
            HAS_SINK=True,
        )
    return output
