"""Volta Tensor Core index scoring for shared and per-query FP16 key banks."""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(
    compile_flags=["--fmad=false"],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _index_score_kernel(shared_keys, capacity):
    rows = T.dynamic("rows")
    if shared_keys:
        capacity = T.dynamic("capacity")
    banks = 1 if shared_keys else rows
    rows_per_block = 2 if shared_keys else 1
    heads, dim, block = 32, 128, 64

    @T.prim_func
    def index_score_sm70_kernel(
        Q: T.Tensor([rows, heads, dim], T.float16),
        K: T.Tensor([banks, capacity, dim], T.float16),
        S: T.Tensor([banks, capacity], T.float32),
        W: T.Tensor([rows, heads], T.float32),
        L: T.Tensor([rows], T.int32),
        Output: T.Tensor([rows, capacity], T.float32),
    ):
        with T.Kernel(
            T.ceildiv(capacity, block), T.ceildiv(rows, rows_per_block), threads=128
        ) as (tile, group):
            qs = T.alloc_shared([heads, dim], T.float16)
            ks = T.alloc_shared([block, dim], T.float16)
            dots = T.alloc_fragment([heads, block], T.float32)
            result = T.alloc_fragment([block], T.float32)
            first = group * rows_per_block
            last = T.min(first + rows_per_block - 1, rows - 1)
            bank = 0 if shared_keys else first
            longest = T.min(capacity, T.max(0, T.max(L[first], L[last])))
            start = tile * block
            if start < longest:
                for n, d in T.Parallel(block, dim):
                    ks[n, d] = T.if_then_else(
                        start + n < longest, K[bank, start + n, d], 0
                    )
                # Shared prefill keys serve two queries without doubling scratch.
                for item in T.serial(rows_per_block):
                    row = first + item
                    if row < rows:
                        length = T.min(capacity, T.max(0, L[row]))
                        T.copy(Q[row, :, :], qs)
                        T.clear(dots)
                        T.gemm(
                            qs,
                            ks,
                            dots,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for h, n in T.Parallel(heads, block):
                            dots[h, n] = (
                                T.if_then_else(dots[h, n] < 0, 0, dots[h, n])
                                * W[row, h]
                            )
                        T.reduce_sum(dots, result, dim=0)
                        for n in T.Parallel(block):
                            if start + n < capacity:
                                Output[row, start + n] = T.if_then_else(
                                    start + n < length,
                                    result[n] * S[bank, start + n],
                                    -T.infinity(T.float32),
                                )
            else:
                for item in T.serial(rows_per_block):
                    row = first + item
                    if row < rows:
                        for n in T.Parallel(block):
                            if start + n < capacity:
                                Output[row, start + n] = -T.infinity(T.float32)

    return index_score_sm70_kernel


def mqa_logits_sm70(query, keys, key_scales, weights, lengths):
    """FP32 weighted scores with causal masking and FP16 Tensor Core operands.

    Keys are shared [history,128] or per-query [rows,history,128]. Software-FP8
    queries convert losslessly to FP16. Shared histories retain dynamic-width
    compilation and paired-query reuse. Decode retains capacity specialization.
    """
    shared = keys.ndim == 2
    scale_shapes = (
        (keys.shape[:1], (*keys.shape[:1], 1)) if shared else (keys.shape[:2],)
    )
    if not (
        query.is_cuda
        and torch.cuda.get_device_capability(query.device) == (7, 0)
        and query.ndim == 3
        and query.shape[1:] == (32, 128)
        and query.dtype in (torch.float16, torch.float8_e4m3fn)
        and keys.ndim in (2, 3)
        and (shared or (1 <= query.shape[0] <= 8 and keys.shape[0] == query.shape[0]))
        and keys.shape[-1] == 128
        and keys.dtype == torch.float16
        and keys.is_contiguous()
        and keys.data_ptr() % 16 == 0
        and key_scales.shape in scale_shapes
        and weights.shape == query.shape[:2]
        and key_scales.dtype == weights.dtype == torch.float32
        and lengths.numel() == query.shape[0]
        and lengths.dtype == torch.int32
        and all(t.device == query.device for t in (keys, key_scales, weights, lengths))
    ):
        raise ValueError(
            "SM70 index scoring requires [rows,32,128] FP16/FP8 queries, "
            "aligned shared/per-query FP16 keys, FP32 scales/weights and int32 lengths"
        )
    rows, capacity = query.shape[0], keys.shape[-2]
    output = torch.empty((rows, capacity), device=query.device, dtype=torch.float32)
    if rows and capacity:
        query = query.half().contiguous()
        if query.data_ptr() % 16:
            raise ValueError("SM70 index scoring queries require 16-byte alignment")
        banks = 1 if shared else rows
        _index_score_kernel(shared, None if shared else capacity)(
            query,
            keys.reshape(banks, capacity, 128),
            key_scales.reshape(banks, capacity).contiguous(),
            weights.contiguous(),
            lengths.reshape(-1).contiguous(),
            output,
        )
    return output
