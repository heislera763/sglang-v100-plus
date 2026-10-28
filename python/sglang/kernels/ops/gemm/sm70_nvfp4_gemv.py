"""FP32-accumulating GEMV over the SM70 Marlin NVFP4 packed layout.

Each active routing block must contain exactly one valid row, as in single-token
expert routing or a batch-one dense projection. Scales are Marlin's encoded
FP16 scale bytes, not native E4M3 values. No tensor cores or FP8 instructions
are required.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(
    A,
    W,
    S,
    Routes,
    Experts,
    P,
    COUNT,
    HAS_COUNT: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    MACRO: tl.constexpr,
):
    route = tl.program_id(2)
    row = tl.load(Routes + route * ROUTE_BLOCK)
    expert = tl.load(Experts + route)
    live = route * ROUTE_BLOCK < tl.load(COUNT) if HAS_COUNT else True
    if live & (expert >= 0):
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        k = tl.program_id(1) * BK + tl.arange(0, BK)
        group_tiles = MACRO // 64
        n_tile = n // 64
        word = (
            expert * (K // 16) * (N * 2)
            + (k[:, None] // 16) * (N * 2)
            + (n_tile[None, :] // group_tiles) * group_tiles * 128
            + ((k[:, None] % 16) * 8 + (n[None, :] % 64) // 8) * group_tiles
            + n_tile[None, :] % group_tiles
        )
        packed = tl.load(W + word).to(tl.uint32)
        shift = ((n % 8) // 2 + 4 * (n % 2)) * 4
        code = (packed >> shift[None, :]) & 15
        fp4_bits = ((code & 7) << 9) | ((code & 8) << 12)
        small_fp4 = fp4_bits.to(tl.uint16).to(tl.float16, bitcast=True)
        scale_n = (n // 4) * 4 + (n % 4) // 2 + 2 * (n % 2)
        scale_byte = tl.load(
            S + expert * (K // 16) * N + (k[:, None] // 16) * N + scale_n[None, :]
        ).to(tl.uint16)
        scale = (scale_byte << 7).to(tl.float16, bitcast=True)
        # Match Marlin's FP16 dequantization before its FP32 accumulation.
        weight = (small_fp4.to(tl.float32) * scale.to(tl.float32)).to(tl.float16)
        a = tl.load(A + (row // TOPK) * K + k).to(tl.float32)
        accum = tl.sum(a[:, None] * weight.to(tl.float32), axis=0)
        tl.store(P + (route * SPLITS + tl.program_id(1)) * N + n, accum)


@triton.jit
def _reduce(
    P,
    G,
    Routes,
    Experts,
    RouterWeights,
    Out,
    COUNT,
    HAS_COUNT: tl.constexpr,
    ZERO_INACTIVE: tl.constexpr,
    ROWS: tl.constexpr,
    ROUTE_POWER2: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    MUL_ROUTER: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
):
    route = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    live = route * ROUTE_BLOCK < tl.load(COUNT) if HAS_COUNT else True
    if live:
        expert = tl.load(Experts + route)
        if expert >= 0:
            splits = tl.arange(0, BS)
            partial = tl.load(
                P + (route * SPLITS + splits[:, None]) * N + n[None, :],
                (splits[:, None] < SPLITS) & (n[None, :] < N),
                0,
            )
            row = tl.load(Routes + route * ROUTE_BLOCK)
            value = tl.sum(partial, axis=0)
            if MUL_ROUTER:
                value *= tl.load(RouterWeights + row)
            value *= tl.load(G + expert).to(tl.float32)
            tl.store(Out + row * N + n, value, n < N)
        elif ZERO_INACTIVE:
            # Masked single-token routing can group several rows in one block.
            offsets = tl.arange(0, ROUTE_POWER2)
            rows = tl.load(
                Routes + route * ROUTE_BLOCK + offsets, offsets < ROUTE_BLOCK, ROWS
            )
            valid = (offsets < ROUTE_BLOCK) & (rows >= 0) & (rows < ROWS)
            tl.store(
                Out + rows[:, None] * N + n[None, :],
                0,
                valid[:, None] & (n[None, :] < N),
            )


def sm70_nvfp4_gemv(
    a,
    out,
    weight,
    scales,
    global_scale,
    sorted_ids,
    expert_ids,
    router_weights,
    block_size,
    topk,
    mul_router,
    partials=None,
    block_n=32,
    block_k=256,
    *,
    padded_count=None,
    zero_inactive_output=False,
):
    """Compute one GEMV per routing block using caller-owned FP32 scratch.

    All tensors must be contiguous and colocated on an SM70 CUDA device.
    Valid routing rows must be unique and the first element in each block;
    expert IDs may be -1 for inactive blocks. Such blocks may hold several
    routes and are zeroed only when requested. An optional device count excludes
    unused tail blocks. The caller guarantees these device-side values.
    Weight shape is [experts,K/16,2*N], encoded scales [experts,K/16,N],
    and output [a.rows*topk,N]. K is divisible by block_k and N by block_n.
    """
    m, k = a.shape
    experts, groups, twice_n = weight.shape
    n = twice_n // 2
    routes = expert_ids.numel()
    if (
        min(m, k, n, experts, topk, block_size, block_n, block_k) <= 0
        or block_n & (block_n - 1)
        or block_k & (block_k - 1)
        or twice_n != 2 * n
    ):
        raise ValueError("Invalid SM70 NVFP4 GEMV dimensions")
    if (
        a.dtype != torch.float16
        or out.dtype != torch.float16
        or weight.dtype != torch.int32
        or scales.dtype != torch.float8_e4m3fn
        or global_scale.dtype != torch.float32
        or router_weights.dtype != torch.float32
        or sorted_ids.dtype != torch.int32
        or expert_ids.dtype != torch.int32
        or a.device.type != "cuda"
        or torch.cuda.get_device_capability(a.device) != (7, 0)
        or k % block_k
        or n % block_n
        or n % 64
        or k % 16
        or groups != k // 16
        or scales.shape != (experts, groups, n)
        or out.shape != (m * topk, n)
        or routes != m * topk
        or sorted_ids.numel() != routes * block_size
        or global_scale.numel() != experts
        or router_weights.numel() != routes
    ):
        raise ValueError("Unsupported SM70 NVFP4 GEMV layout")
    tensors = (
        a,
        out,
        weight,
        scales,
        global_scale,
        sorted_ids,
        expert_ids,
        router_weights,
    )
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        raise ValueError("SM70 NVFP4 GEMV requires contiguous colocated tensors")
    if padded_count is not None and not (
        padded_count.device == a.device
        and padded_count.dtype == torch.int32
        and padded_count.is_contiguous()
        and padded_count.numel() in (1, 2)
    ):
        raise ValueError("Invalid GEMV routing count")
    splits = k // block_k
    if partials is None:
        partials = torch.empty(
            (routes, splits, n), device=a.device, dtype=torch.float32
        )
    if (
        partials.shape != (routes, splits, n)
        or partials.dtype != torch.float32
        or partials.device != a.device
        or not partials.is_contiguous()
    ):
        raise ValueError("Invalid GEMV scratch")
    macro = 256 if n % 256 == 0 else 128 if n % 128 == 0 else 64
    _partial[(n // block_n, splits, routes)](
        a,
        weight,
        scales.view(torch.uint8),
        sorted_ids,
        expert_ids,
        partials,
        padded_count,
        padded_count is not None,
        n,
        k,
        topk,
        block_size,
        block_n,
        block_k,
        splits,
        macro,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _reduce[(triton.cdiv(n, 128), routes)](
        partials,
        global_scale,
        sorted_ids,
        expert_ids,
        router_weights,
        out,
        padded_count,
        padded_count is not None,
        zero_inactive_output,
        m * topk,
        triton.next_power_of_2(block_size),
        n,
        splits,
        block_size,
        mul_router,
        128,
        triton.next_power_of_2(splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
