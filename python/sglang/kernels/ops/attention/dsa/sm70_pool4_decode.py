"""Device-driven pool4 cache updates/gathers for SM70 decode and graph replay."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.sm70_indexer import (
    _decode_e4m3fn,
    _hadamard_stage,
    _quantize_row,
)


def pool4_cache_page_size(cache):
    """Number of pooled keys per row-major FP8/FP32 cache page."""
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 2
        or cache.shape[1] not in (16 * 132, 64 * 132)
    ):
        raise ValueError("SM70 pool4 requires 16- or 64-slot FP8/FP32 cache pages")
    return cache.shape[1] // 132


@triton.jit
def _pool4_decode_write(
    K,
    G,
    APE,
    TAIL_K,
    TAIL_G,
    CACHE,
    REQS,
    LENGTHS,
    TABLE,
    K_ROW: tl.constexpr,
    K_COL: tl.constexpr,
    G_ROW: tl.constexpr,
    G_COL: tl.constexpr,
    A_ROW: tl.constexpr,
    A_COL: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TOKEN_PAGE_SIZE: tl.constexpr,
    INDEX_PAGE_SIZE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    RETAIN_CLOSED_TAIL: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(REQS + b).to(tl.int64)
    length = tl.load(LENGTHS + b).to(tl.int32)
    if (length > 0) & (req >= 0):
        prefix = length - 1
        slot = prefix % 4
        cols = tl.arange(0, 128)
        key = tl.load(K + b * K_ROW + cols * K_COL).to(tl.bfloat16).to(tl.float32)
        gate = tl.load(G + b * G_ROW + cols * G_COL).to(tl.float32)
        if RETAIN_CLOSED_TAIL:
            gate = gate.to(tl.bfloat16).to(tl.float32)
        if length % 4 == 0:
            slots = tl.arange(0, 4)
            closed_tail_pos = (prefix - slot + slots) % TAIL_SIZE
            previous_k = tl.load(
                TAIL_K
                + req * TAIL_SIZE * 128
                + closed_tail_pos[:, None] * 128
                + cols[None, :],
                mask=slots[:, None] < slot,
                other=0,
            ).to(tl.float32)
            previous_g = tl.load(
                TAIL_G
                + req * TAIL_SIZE * 128
                + closed_tail_pos[:, None] * 128
                + cols[None, :],
                mask=slots[:, None] < slot,
                other=0,
            ).to(tl.float32)
            keys = tl.where(slots[:, None] == slot, key[None, :], previous_k)
            scores = tl.where(slots[:, None] == slot, gate[None, :], previous_g)
            scores += tl.load(APE + slots[:, None] * A_ROW + cols[None, :] * A_COL).to(
                tl.float32
            )
            prob = tl.exp(scores - tl.max(scores, 0)[None, :])
            prob = prob / tl.sum(prob, 0)[None, :]
            x = tl.sum(prob * keys, 0).to(tl.bfloat16).to(tl.float32)
            x = _hadamard_stage(x, 64, 1)
            x = _hadamard_stage(x, 32, 2)
            x = _hadamard_stage(x, 16, 4)
            x = _hadamard_stage(x, 8, 8)
            x = _hadamard_stage(x, 4, 16)
            x = _hadamard_stage(x, 2, 32)
            x = _hadamard_stage(x, 1, 64)
            encoded, scale = _quantize_row(
                (x * 128**-0.5).to(tl.bfloat16).to(tl.float32), ROUND_SCALE
            )
            pool_id = prefix // 4
            physical = tl.load(
                TABLE
                + req * TABLE_STRIDE
                + pool_id // INDEX_PAGE_SIZE * (INDEX_PAGE_SIZE * 4)
            ).to(tl.int64)
            page = physical // TOKEN_PAGE_SIZE
            offset = pool_id % INDEX_PAGE_SIZE
            tl.store(CACHE + page * CACHE_STRIDE + offset * 128 + cols, encoded)
            scale_ptr = (
                CACHE + page * CACHE_STRIDE + INDEX_PAGE_SIZE * 128 + offset * 4
            ).to(tl.pointer_type(tl.float32))
            tl.store(scale_ptr, scale)
        if (length % 4 != 0) | RETAIN_CLOSED_TAIL:
            tail_pos = prefix % TAIL_SIZE
            tl.store(TAIL_K + req * TAIL_SIZE * 128 + tail_pos * 128 + cols, key)
            tl.store(TAIL_G + req * TAIL_SIZE * 128 + tail_pos * 128 + cols, gate)


@triton.jit
def _pool4_decode_gather(
    CACHE,
    TABLE,
    REQS,
    LENGTHS,
    K_OUT,
    S_OUT,
    POOLS,
    CACHE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TOKEN_PAGE_SIZE: tl.constexpr,
    INDEX_PAGE_SIZE: tl.constexpr,
    MAX_POOLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(REQS + b).to(tl.int64)
    length = tl.load(LENGTHS + b).to(tl.int32)
    pools = tl.where(req >= 0, length // 4, 0)
    if tl.program_id(1) == 0:
        tl.store(POOLS + b, pools)
    # Coarse CTAs skip unused history before touching keys/scales or scratch.
    # The scorer masks every read using POOLS; future rows stay untouched.
    start = tl.program_id(1) * BLOCK
    if start < tl.minimum(pools, MAX_POOLS):
        ids = start + tl.arange(0, BLOCK)
        valid = (ids < pools) & (ids < MAX_POOLS)
        cols = tl.arange(0, 128)
        physical = tl.load(
            TABLE + req * TABLE_STRIDE + ids // INDEX_PAGE_SIZE * (INDEX_PAGE_SIZE * 4),
            mask=valid,
            other=0,
        ).to(tl.int64)
        pages = physical // TOKEN_PAGE_SIZE
        offsets = ids % INDEX_PAGE_SIZE
        encoded = tl.load(
            CACHE
            + pages[:, None] * CACHE_STRIDE
            + offsets[:, None] * 128
            + cols[None, :],
            mask=valid[:, None],
            other=0,
        )
        keys = _decode_e4m3fn(encoded)
        scale_ptr = (
            CACHE + pages * CACHE_STRIDE + INDEX_PAGE_SIZE * 128 + offsets * 4
        ).to(tl.pointer_type(tl.float32))
        scales = tl.load(scale_ptr, mask=valid, other=0)
        tl.store(
            K_OUT + (b * MAX_POOLS + ids[:, None]) * 128 + cols[None, :],
            keys,
            mask=valid[:, None],
        )
        tl.store(S_OUT + b * MAX_POOLS + ids, scales, mask=valid)


def pool4_decode_sm70(
    keys,
    gates,
    ape,
    tail_keys,
    tail_gates,
    cache,
    token_table,
    request_ids,
    seq_lens,
    round_scale=False,
    retain_closed_tail=False,
    *,
    token_page_size=64,
):
    """Append one key per request, close pools on-device, and gather FP16 keys.

    The gathered capacity depends only on token-table width, never host length.
    Only the live prefix is initialized, accompanied by a device-side pool count.
    Consumers must mask future entries with this count before reading scratch.
    Cache and tail storage are request-owned; all addresses survive graph replay.
    """
    batch = keys.shape[0]
    if keys.shape != (batch, 128) or gates.shape != keys.shape or ape.shape != (4, 128):
        raise ValueError(
            "SM70 decode requires one 128-wide key/gate per request and pool4 APE"
        )
    if (
        tail_keys.shape != tail_gates.shape
        or tail_keys.shape[-1] != 128
        or tail_keys.shape[1] < 4
    ):
        raise ValueError(
            "SM70 decode requires matching 128-wide tail rings of at least four slots"
        )
    if not (
        tail_keys.is_contiguous()
        and tail_gates.is_contiguous()
        and cache.is_contiguous()
    ):
        raise ValueError("SM70 decode cache and tail rings must be contiguous")
    index_page_size = pool4_cache_page_size(cache)
    if token_page_size not in (64, 256):
        raise ValueError("SM70 pool4 supports 64- or 256-token allocation pages")
    capacity = token_table.shape[1] // 4
    gathered = torch.empty(
        (batch, capacity, 128), dtype=torch.float16, device=keys.device
    )
    scales = torch.empty((batch, capacity), dtype=torch.float32, device=keys.device)
    pools = torch.empty((batch,), dtype=torch.int32, device=keys.device)
    _pool4_decode_write[(batch,)](
        keys,
        gates,
        ape,
        tail_keys,
        tail_gates,
        cache,
        request_ids,
        seq_lens,
        token_table,
        *keys.stride(),
        *gates.stride(),
        *ape.stride(),
        tail_keys.shape[1],
        cache.stride(0),
        token_table.stride(0),
        token_page_size,
        index_page_size,
        round_scale,
        retain_closed_tail,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _pool4_decode_gather[(batch, triton.cdiv(capacity, 64))](
        cache,
        token_table,
        request_ids,
        seq_lens,
        gathered,
        scales,
        pools,
        cache.stride(0),
        token_table.stride(0),
        token_page_size,
        index_page_size,
        capacity,
        64,
        num_warps=4,
    )
    return gathered, scales, pools


@triton.jit
def _pool4_spec_write(
    K,
    G,
    APE,
    TAIL_K,
    TAIL_G,
    CACHE,
    REQS,
    STARTS,
    TAIL_STARTS,
    WRITE_LOCS,
    OUT_LOCS,
    EFFECTIVE_N,
    K_ROW: tl.constexpr,
    G_ROW: tl.constexpr,
    A_ROW: tl.constexpr,
    A_COL: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    INDEX_PAGE_SIZE: tl.constexpr,
    WRITE_STRIDE: tl.constexpr,
    N: tl.constexpr,
    MAX_CLOSED: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_EFFECTIVE_N: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(REQS + b).to(tl.int64)
    start = tl.load(STARTS + b).to(tl.int32)
    active = tl.load(OUT_LOCS + b * N) != 0
    if active & (req >= 0) & (start >= 0):
        cols = tl.arange(0, 128)
        for i in tl.static_range(N):
            row = b * N + i
            pos = (start + i) % TAIL_SIZE
            key = tl.load(K + row * K_ROW + cols).to(tl.bfloat16)
            gate = tl.load(G + row * G_ROW + cols).to(tl.bfloat16)
            tl.store(TAIL_K + (req * TAIL_SIZE + pos) * 128 + cols, key)
            tl.store(TAIL_G + (req * TAIL_SIZE + pos) * 128 + cols, gate)
        tl.debug_barrier()
        effective_n = tl.load(EFFECTIVE_N + b) if HAS_EFFECTIVE_N else N
        closed = (start + effective_n) // 4 - start // 4
        base = tl.load(TAIL_STARTS + b).to(tl.int32)
        slots = tl.arange(0, 4)
        for p in tl.static_range(MAX_CLOSED):
            if p < closed:
                positions = (base + p * 4 + slots) % TAIL_SIZE
                offsets = (req * TAIL_SIZE + positions[:, None]) * 128 + cols[None, :]
                keys = tl.load(TAIL_K + offsets).to(tl.float32)
                scores = tl.load(TAIL_G + offsets).to(tl.float32)
                scores += tl.load(
                    APE + slots[:, None] * A_ROW + cols[None, :] * A_COL
                ).to(tl.float32)
                prob = tl.exp(scores - tl.max(scores, 0)[None, :])
                prob = prob / tl.sum(prob, 0)[None, :]
                x = tl.sum(prob * keys, 0).to(tl.bfloat16).to(tl.float32)
                x = _hadamard_stage(x, 64, 1)
                x = _hadamard_stage(x, 32, 2)
                x = _hadamard_stage(x, 16, 4)
                x = _hadamard_stage(x, 8, 8)
                x = _hadamard_stage(x, 4, 16)
                x = _hadamard_stage(x, 2, 32)
                x = _hadamard_stage(x, 1, 64)
                encoded, scale = _quantize_row(
                    (x * 128**-0.5).to(tl.bfloat16).to(tl.float32), ROUND_SCALE
                )
                loc = tl.load(WRITE_LOCS + b * WRITE_STRIDE + p).to(tl.int64)
                page, offset = loc // INDEX_PAGE_SIZE, loc % INDEX_PAGE_SIZE
                tl.store(CACHE + page * CACHE_STRIDE + offset * 128 + cols, encoded)
                scale_ptr = (
                    CACHE + page * CACHE_STRIDE + INDEX_PAGE_SIZE * 128 + offset * 4
                ).to(tl.pointer_type(tl.float32))
                tl.store(scale_ptr, scale)


def pool4_spec_sm70(
    keys,
    gates,
    ape,
    tail_keys,
    tail_gates,
    cache,
    token_table,
    request_ids,
    write_starts,
    tail_starts,
    write_locations,
    out_locations,
    query_request_ids,
    query_lengths,
    effective_num_tokens=None,
    round_scale=False,
    *,
    token_page_size=64,
):
    """Apply upstream chain verify/draft-extend plans using software FP8.

    Stage every candidate in the extra-width tail ring, but close only pools
    within effective_num_tokens when supplied by draft-extend. Rejected future
    pools remain outside the next round's causal lengths and are overwritten
    before becoming visible. Gather scratch is initialized only inside each
    query's live pool count. The caller owns plans, token pages and rollback.
    """
    batch = request_ids.numel()
    if batch == 0 or keys.shape[0] % batch:
        raise ValueError("Speculative pool4 requires a fixed chain width per request")
    n = keys.shape[0] // batch
    max_closed = (n + 3) // 4
    if not 1 <= n <= 8 or keys.shape != gates.shape or keys.shape[1:] != (128,):
        raise ValueError("SM70 pool4 supports chain widths 1..8 and 128-wide keys")
    index_page_size = pool4_cache_page_size(cache)
    if token_page_size not in (64, 256):
        raise ValueError("SM70 pool4 supports 64- or 256-token allocation pages")
    if (
        ape.shape != (4, 128)
        or tail_keys.shape != tail_gates.shape
        or tail_keys.shape[1] < 4 + n
        or tail_keys.shape[-1] != 128
        or tail_keys.dtype != torch.bfloat16
        or tail_gates.dtype != torch.bfloat16
        or write_locations.shape != (batch, max_closed)
        or out_locations.numel() != keys.shape[0]
        or query_lengths.numel() != keys.shape[0]
        or query_request_ids.numel() != keys.shape[0]
        or write_starts.numel() != batch
        or tail_starts.numel() != batch
    ):
        raise ValueError("Invalid SM70 speculative pool4 plan or cache layout")
    tensors = [
        keys,
        gates,
        ape,
        tail_keys,
        tail_gates,
        cache,
        token_table,
        request_ids,
        write_starts,
        tail_starts,
        write_locations,
        out_locations,
        query_request_ids,
        query_lengths,
    ]
    if effective_num_tokens is not None:
        if effective_num_tokens.numel() != batch:
            raise ValueError("Effective chain lengths must match request count")
        tensors.append(effective_num_tokens)
    if any(t.device != keys.device or not t.is_contiguous() for t in tensors):
        raise ValueError("SM70 speculative pool4 requires contiguous colocated tensors")
    capacity = token_table.shape[1] // 4
    gathered = torch.empty(
        (keys.shape[0], capacity, 128), dtype=torch.float16, device=keys.device
    )
    scales = torch.empty(
        (keys.shape[0], capacity), dtype=torch.float32, device=keys.device
    )
    pools = torch.empty((keys.shape[0],), dtype=torch.int32, device=keys.device)
    _pool4_spec_write[(batch,)](
        keys,
        gates,
        ape,
        tail_keys,
        tail_gates,
        cache,
        request_ids,
        write_starts,
        tail_starts,
        write_locations,
        out_locations,
        effective_num_tokens,
        keys.stride(0),
        gates.stride(0),
        *ape.stride(),
        tail_keys.shape[1],
        cache.stride(0),
        index_page_size,
        write_locations.stride(0),
        n,
        max_closed,
        round_scale,
        effective_num_tokens is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _pool4_decode_gather[(keys.shape[0], triton.cdiv(capacity, 64))](
        cache,
        token_table,
        query_request_ids,
        query_lengths,
        gathered,
        scales,
        pools,
        cache.stride(0),
        token_table.stride(0),
        token_page_size,
        index_page_size,
        capacity,
        64,
        num_warps=4,
    )
    return gathered, scales, pools
