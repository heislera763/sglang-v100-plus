"""Volta MMA over routed SM70 Marlin NVFP4 weights and encoded scale bytes.

Keep FP16 dequantization, FP32 accumulation/router/global scaling and FP16
output. A 16-row tile avoids padding small expert groups to Marlin's 32 rows;
K-major shared weights avoid scattering each decoded word across distant rows.
"""

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl


@tilelang.jit(
    compile_flags=["--fmad=false"],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _routed_mma(n, k, topk, mul_router, splits=1):
    bm, bn, bk, block = 16, 128, 64, 8
    nt, nr = T.dynamic("nt"), T.dynamic("nr")
    ns, ne, nb = T.dynamic("ns"), T.dynamic("ne"), T.dynamic("nb")
    groups = 4 if n % 256 == 0 else 2

    @T.prim_func
    def nvfp4_routed_sm70(
        A: T.Tensor([nt, k], T.float16),
        W: T.Tensor([ne, k // 16, n * 2], T.int32),
        S: T.Tensor([ne, k // 16, n // 8], T.uint64),
        G: T.Tensor([ne], T.float32),
        Ids: T.Tensor([ns], T.int32),
        Experts: T.Tensor([nb], T.int32),
        Count: T.Tensor([1], T.int32),
        Router: T.Tensor([nr], T.float32),
        C: T.Tensor([splits, nr, n], T.float32 if splits > 1 else T.float16),
    ):
        with T.Kernel(nb, n // bn, splits, threads=128) as (row, col, part):
            ash = T.alloc_shared([bm, bk], T.float16)
            bsh = T.alloc_shared([bk, bn], T.float16)
            accum = T.alloc_fragment([bm, bn], T.float32)
            # Bind once outside the branch: inside it, TileLang inlines this
            # metadata load into every decoded weight address in the K loop.
            expert = Experts[row]
            if row * block < Count[0]:
                T.clear(accum)
                if expert >= 0:
                    for step in T.Pipelined(k // bk // splits, num_stages=2):
                        kk = part * (k // bk // splits) + step
                        for m, d in T.Parallel(bm, bk):
                            route = T.if_then_else(m < block, Ids[row * block + m], nr)
                            ash[m, d] = T.if_then_else(
                                (route >= 0) & (route < nr),
                                A[route // topk, kk * bk + d],
                                0,
                            )
                        # One packed word covers eight output columns. Scales
                        # are encoded Half exponents, NOT numeric E4M3 values.
                        for group, ng, ki in T.Parallel(bk // 16, bn // 8, 16):
                            ln = col * bn + ng * 8
                            lk = kk * bk + group * 16 + ki
                            word = (
                                (ln // 64 // groups) * groups * 128
                                + (ki * 8 + (ln % 64) // 8) * groups
                                + (ln // 64) % groups
                            )
                            packed = T.cast(W[expert, lk // 16, word], T.uint32)
                            scales = S[expert, lk // 16, ln // 8]
                            for ni in T.unroll(8):
                                shift = (ni // 2 + 4 * (ni % 2)) * 4
                                code = (packed >> shift) & 15
                                fp4 = T.reinterpret(
                                    T.float16,
                                    T.cast(
                                        ((code & 7) << 9) | ((code & 8) << 12), T.uint16
                                    ),
                                )
                                sp = (ni // 4) * 4 + (ni % 4) // 2 + 2 * (ni % 2)
                                scale_byte = T.cast(
                                    (scales >> (sp * 8)) & 255, T.uint16
                                )
                                scale = T.reinterpret(
                                    T.float16,
                                    T.cast(scale_byte << 7, T.uint16),
                                )
                                bsh[group * 16 + ki, ng * 8 + ni] = fp4 * scale
                        T.gemm(ash, bsh, accum, policy=T.GemmWarpPolicy.FullCol)
                # Valid route IDs occur once in the padded permutation; each
                # row/column tile has one writer, including zeroed expert -1.
                for m, j in T.Parallel(bm, bn):
                    if m < block:
                        route = Ids[row * block + m]
                        if (route >= 0) & (route < nr):
                            if splits > 1:
                                C[part, route, col * bn + j] = accum[m, j]
                            elif expert >= 0:
                                if mul_router:
                                    C[0, route, col * bn + j] = T.cast(
                                        (accum[m, j] * Router[route]) * G[expert],
                                        T.float16,
                                    )
                                else:
                                    C[0, route, col * bn + j] = T.cast(
                                        accum[m, j] * G[expert], T.float16
                                    )
                            else:
                                C[0, route, col * bn + j] = 0

    return nvfp4_routed_sm70


@triton.jit
def _finish_routed(
    PARTIALS,
    GLOBAL,
    ROUTER,
    IDS,
    EXPERTS,
    COUNT,
    OUTPUT,
    ROUTES: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    MUL_ROUTER: tl.constexpr,
):
    block, tile = tl.program_id(0), tl.program_id(1)
    if block * 8 < tl.load(COUNT):
        expert = tl.load(EXPERTS + block)
        route = tl.load(IDS + block * 8 + tl.arange(0, 8))
        col = tile * 128 + tl.arange(0, 128)
        valid = (route[:, None] >= 0) & (route[:, None] < ROUTES)
        value = tl.full((8, 128), 0, tl.float32)
        for split in tl.static_range(SPLITS):
            value += tl.load(
                PARTIALS + (split * ROUTES + route[:, None]) * N + col[None, :],
                valid,
                0,
            )
        if MUL_ROUTER:
            value *= tl.load(ROUTER + route, (route >= 0) & (route < ROUTES), 0)[
                :, None
            ]
        value *= tl.load(GLOBAL + expert, expert >= 0, 0)
        tl.store(OUTPUT + route[:, None] * N + col[None, :], value, valid)


def sm70_nvfp4_routed_gemm(
    a,
    output,
    weight,
    scales,
    global_scale,
    sorted_ids,
    expert_ids,
    padded_count,
    router,
    topk,
    mul_router,
):
    """Write route-major Half output for block-eight padded expert routing.

    Device metadata follows moe_align_block_size: each valid row appears once,
    padding is >=a.rows*topk, expert -1 produces zero, and padded_count gates
    unused tail metadata. Valid experts lie within the bank; count is bounded
    by sorted_ids. The caller guarantees these device-side values.
    Scale bytes use the encoded Marlin layout. Long, narrow contractions split
    K into FP32 partials before scaling and the only FP16 output store. Scratch
    is bounded by tensor capacity and owned by this invocation, including graph
    capture; no CPU count reads or process-global buffers are used.
    """
    if a.ndim != 2 or weight.ndim != 3:
        raise ValueError("SM70 NVFP4 routed GEMM requires matrices and expert banks")
    m, k = a.shape
    experts, groups, twice_n = weight.shape
    n = twice_n // 2
    if not (
        a.is_cuda
        and torch.cuda.get_device_capability(a.device) == (7, 0)
        and a.dtype == output.dtype == torch.float16
        and weight.dtype == torch.int32
        and scales.dtype == torch.float8_e4m3fn
        and global_scale.dtype == router.dtype == torch.float32
        and sorted_ids.dtype == expert_ids.dtype == padded_count.dtype == torch.int32
        and 0 < m <= 32
        and 0 < topk <= 8
        and m * topk <= 32
        and experts > 0
        and k > 0
        and k % 64 == 0
        and n > 0
        and n % 128 == 0
        and twice_n == n * 2
        and groups == k // 16
        and scales.shape == (experts, groups, n)
        and global_scale.numel() == experts
        and output.ndim in (2, 3)
        and output.shape[-1] == n
        and output.numel() == m * topk * n
        and router.numel() == m * topk
        and sorted_ids.ndim == expert_ids.ndim == 1
        and sorted_ids.numel() % 8 == 0
        and expert_ids.numel() == sorted_ids.numel() // 8
        and sorted_ids.numel() >= m * topk
        and padded_count.numel() == 1
    ):
        raise ValueError("Unsupported SM70 NVFP4 routed GEMM layout")
    tensors = (
        a,
        output,
        weight,
        scales,
        global_scale,
        sorted_ids,
        expert_ids,
        padded_count,
        router,
    )
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        raise ValueError("SM70 NVFP4 routed GEMM requires contiguous colocated tensors")
    # Small expert populations leave too few N tiles to fill 80 SMs. Eight
    # independent K partitions expose more work without decoding weights twice.
    # Wide, short-K down projections already have enough output parallelism.
    splits = 8 if k >= 2048 and k % 512 == 0 and n <= 1024 else 1
    target = (
        torch.empty((splits, m * topk, n), device=a.device, dtype=torch.float32)
        if splits > 1
        else output.view(1, -1, n)
    )
    _routed_mma(n, k, topk, mul_router, splits)(
        a,
        weight,
        scales.view(torch.uint64),
        global_scale.reshape(-1),
        sorted_ids,
        expert_ids,
        padded_count.reshape(1),
        router.reshape(-1),
        target,
    )
    if splits > 1:
        _finish_routed[(expert_ids.numel(), n // 128)](
            target,
            global_scale,
            router,
            sorted_ids,
            expert_ids,
            padded_count,
            output,
            m * topk,
            n,
            splits,
            mul_router,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
