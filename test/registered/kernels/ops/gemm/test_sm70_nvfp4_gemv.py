"""Guard packed-weight/scales decoding and dynamic routing during graph replay.

The repacker comes from the independent installed SM70 Marlin extension;
the reference multiplies checkpoint-native FP4 values and E4M3 scales on CPU.
Changing nibble ordering, exponent compensation, or cached routing breaks parity.
"""

import os

import pytest
import torch

from sglang.kernels.ops.gemm.sm70_nvfp4_gemv import sm70_nvfp4_gemv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="jit-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 Marlin encoded-layout coverage",
)


@pytest.mark.parametrize(
    "m,topk,n,k",
    [
        (1, 1, 64, 256),
        (1, 1, 128, 256),
        (1, 1, 512, 4096),
        (1, 1, 3072, 4096),
        (1, 1, 6144, 4096),
        (1, 1, 4096, 1024),
        (1, 1, 4096, 256),
        (1, 1, 4096, 512),
        (1, 8, 512, 4096),
        (1, 8, 1024, 4096),
        (8, 1, 4096, 256),
        (8, 1, 4096, 512),
    ],
)
def test_checkpoint_reference_and_graph_replay(m, topk, n, k):
    from sglang_v100_plus.quantization import (
        _dense_repack,
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    assert os.environ.get("SGLANG_V100_MARLIN_DIR"), "Install the V100 Marlin extension"
    torch.manual_seed(531)
    routes = m * topk
    experts = 1 if routes == 1 else 9
    packed = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda"
    )
    scales = torch.rand(experts, n, k // 16, device="cuda").to(torch.float8_e4m3fn)
    # Include zero block scales and non-power-of-two global scales.
    scales.view(torch.uint8)[:, :, ::7] = 0
    global_native = torch.linspace(0.0625, 0.123456, experts, device="cuda")
    encoded, factor = sm70_nvfp4_marlin_process_scales(
        scales.transpose(1, 2).contiguous(), torch.float16
    )
    global_scale = (
        sm70_nvfp4_marlin_process_global_scale(global_native, torch.float16) / factor
    )
    weight = _dense_repack(packed)
    codes = torch.stack((packed.cpu().int() & 15, packed.cpu().int() >> 4), -1).reshape(
        experts, n, k
    )
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    reference_weight = (
        lookup[codes]
        * scales.cpu().float().repeat_interleave(16, -1)
        * global_native.cpu()[:, None, None]
    )
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    out = torch.full((routes, n), float("nan"), dtype=torch.float16, device="cuda")
    partials = torch.empty((routes, k // 128, n), dtype=torch.float32, device="cuda")
    block = 8
    sorted_ids = torch.full((routes, block), routes, dtype=torch.int32, device="cuda")
    sorted_ids[:, 0] = torch.arange(routes, device="cuda", dtype=torch.int32)
    sorted_ids = sorted_ids.flatten()
    expert_ids = torch.arange(routes, device="cuda", dtype=torch.int32)
    router = torch.linspace(0.125, 1, routes, device="cuda")
    mul_router = m > 1

    def run():
        return sm70_nvfp4_gemv(
            a,
            out,
            weight,
            encoded,
            global_scale,
            sorted_ids,
            expert_ids,
            router,
            block,
            topk,
            mul_router,
            partials,
            block_k=128,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        selected = torch.randperm(experts)[:routes]
        row_ids = torch.randperm(routes)
        expert_ids.copy_(selected)
        sorted_ids.view(routes, block)[:, 0].copy_(row_ids)
        a.copy_(torch.randn_like(a))
        router.copy_(torch.rand_like(router))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.empty(routes, n)
        cpu_a = a.cpu().float()
        cpu_router = router.cpu()
        for index, row in enumerate(row_ids.tolist()):
            expected[row] = cpu_a[row // topk] @ reference_weight[selected[index]].T
        if mul_router:
            expected *= cpu_router[:, None]
        torch.testing.assert_close(out.cpu().float(), expected, rtol=0.01, atol=0.015)

    # New routing producers coalesce masked routes and leave unused capacity
    # unwritten. Neither negative experts nor tail blocks may read weights.
    if routes == 8:
        count = torch.tensor([3 * block, 1], device="cuda", dtype=torch.int32)
        sorted_ids.fill_(routes)
        sorted_ids[:2] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        sorted_ids[block] = 2
        sorted_ids[2 * block] = 3
        expert_ids.fill_(123456)
        expert_ids[:3] = torch.tensor([-1, 2, -1], device="cuda", dtype=torch.int32)
        for zero in (False, True):
            out.fill_(17)

            def masked():
                return sm70_nvfp4_gemv(
                    a,
                    out,
                    weight,
                    encoded,
                    global_scale,
                    sorted_ids,
                    expert_ids,
                    router,
                    block,
                    topk,
                    mul_router,
                    partials,
                    block_k=128,
                    padded_count=count,
                    zero_inactive_output=zero,
                )

            masked()
            replay = torch.cuda.CUDAGraph()
            with torch.cuda.graph(replay):
                masked()
            out.fill_(17)
            replay.replay()
            torch.cuda.synchronize()
            expected = torch.full((routes, n), 17.0)
            expected[2] = a[2 // topk].cpu().float() @ reference_weight[2].T
            if mul_router:
                expected[2] *= router[2].item()
            if zero:
                expected[[0, 1, 3]] = 0
            torch.testing.assert_close(
                out.cpu().float(), expected, rtol=0.01, atol=0.015
            )
            count.zero_()
            out.fill_(17)
            replay.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, torch.full_like(out, 17))
            count[0] = 3 * block


@pytest.mark.parametrize(
    "m,topk,n,k,mul_router",
    [
        (1, 8, 128, 64, False),
        (1, 8, 512, 4096, False),
        (3, 8, 512, 4096, False),
        (4, 8, 512, 4096, False),
        (3, 8, 1024, 4096, False),
        (32, 1, 512, 4096, True),
        (24, 1, 4096, 256, True),
        (32, 1, 4096, 512, True),
    ],
)
def test_routed_mma_native_quant_reference_and_graph(m, topk, n, k, mul_router):
    """Pin encoded nibble/scale ordering, scatter and graph metadata lifetime.

    Reference weights decode checkpoint-native FP4/FP8 on CPU in FP64.
    Padded expert -1 writes zero; invalid unused tail experts must never load.
    Changing routes/inputs after capture must change results without recompiling.
    """
    from sglang_v100_plus.quantization import (
        _dense_repack,
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    from sglang.kernels.ops.gemm.sm70_nvfp4_routed import sm70_nvfp4_routed_gemm

    torch.manual_seed(531)
    e, routes = 9, m * topk
    packed = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
    scales_native = torch.rand(e, n, k // 16, device="cuda").to(torch.float8_e4m3fn)
    # E4M3 positive extremes, including scales with the sign-bit position set
    # after Marlin encoding. They must be decoded as unsigned metadata bytes.
    scales_native.view(torch.uint8)[:, :, 1::31] = 126  # 448, largest finite
    scales_native.view(torch.uint8)[:, :, 2::23] = 1  # smallest subnormal
    scales_native.view(torch.uint8)[:, :, ::7] = 0
    # Keep true weights in the same range when exercising the full FP8 scale
    # range. Global and block scales compensate each other in NVFP4.
    globals_native = torch.linspace(0.0625, 0.123456, e, device="cuda") / 448
    encoded, factor = sm70_nvfp4_marlin_process_scales(
        scales_native.transpose(1, 2).contiguous(), torch.float16
    )
    globals_encoded = (
        sm70_nvfp4_marlin_process_global_scale(globals_native, torch.float16) / factor
    )
    weight = _dense_repack(packed)
    codes = torch.stack((packed.cpu().long() & 15, packed.cpu().long() >> 4), -1)
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    reference_weight = lookup[
        codes.reshape(e, n, k)
    ] * scales_native.cpu().double().repeat_interleave(16, -1)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    router = torch.linspace(0.07, 0.93, routes, device="cuda")
    storage = torch.full((routes * n + 64,), 17.0, device="cuda", dtype=torch.float16)
    out = storage[: routes * n].view(routes, n)
    sorted_ids = torch.full((16 * 8,), routes, device="cuda", dtype=torch.int32)
    expert_ids = torch.full((16,), 123456, device="cuda", dtype=torch.int32)
    count = torch.zeros(2, device="cuda", dtype=torch.int32)

    def metadata(offset):
        selected = (torch.arange(routes) + offset) % e
        ids = torch.full((16 * 8,), routes, dtype=torch.int32)
        experts = torch.full((16,), 123456, dtype=torch.int32)
        cursor = 0
        for expert in range(e):
            live = torch.where(selected == expert)[0].int()
            if live.numel():
                ids[cursor * 8 : cursor * 8 + live.numel()] = live
                experts[cursor] = expert if expert != e - 1 else -1
                cursor += 1
        sorted_ids.copy_(ids)
        expert_ids.copy_(experts)
        count[0] = cursor * 8
        return selected

    def run(zero=True):
        return sm70_nvfp4_routed_gemm(
            a,
            out,
            weight,
            encoded,
            globals_encoded,
            sorted_ids,
            expert_ids,
            count,
            router,
            topk,
            mul_router,
            zero_inactive_output=zero,
        )

    selected = metadata(0)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for offset in (0, 2):
        if offset:
            selected = metadata(offset)
            a.mul_(0.625)
            router.copy_(router.flip(0).contiguous())
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.zeros(routes, n, dtype=torch.float64)
        cpu_a = a.cpu().double()
        for row, expert in enumerate(selected.tolist()):
            if expert != e - 1:
                expected[row] = cpu_a[row // topk] @ reference_weight[expert].T
                if mul_router:
                    expected[row] *= router[row].item()
                expected[row] *= globals_native[expert].item()
        torch.testing.assert_close(out.cpu(), expected.half(), rtol=0.001, atol=0.0002)
        assert torch.isfinite(out).all()
        assert torch.equal(storage[-64:], torch.full_like(storage[-64:], 17.0))

    # The caller may request preservation instead of clearing inactive rows.
    out.fill_(17)
    run(False)
    expected[selected == e - 1] = 17
    torch.testing.assert_close(out.cpu(), expected.half(), rtol=0.001, atol=0.0002)

    # Replaying an empty schedule must not read stale partials or invalid tail
    # experts, and must leave both the output and its canary untouched.
    count.zero_()
    expert_ids.fill_(123456)
    storage.fill_(17.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(storage, torch.full_like(storage, 17.0))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
