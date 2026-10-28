"""Upstream routing counts/zeroing must also hold on the native SM70 adapter."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Native SM70 Marlin adapter",
)


@pytest.mark.parametrize("sorted_prefix", [False, True])
def test_native_inactive_rows_and_graph_refresh(sorted_prefix):
    from sglang_v100_plus.kernels.moe_marlin import moe_wna16_marlin_gemm
    from sglang_v100_plus.quantization import (
        _dense_repack,
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    from sglang.srt.layers.quantization.utils import get_scalar_types
    from sglang.test.test_utils import published_topology

    m, k, n, topk, e, block = 4, 128, 256, 2, 3, 32
    a = torch.full((m, k), 0.5, device="cuda", dtype=torch.float16)
    packed = torch.full((e, n, k // 2), 0x22, device="cuda", dtype=torch.uint8)
    weight = _dense_repack(packed)
    scales, factor = sm70_nvfp4_marlin_process_scales(
        torch.ones((e, k // 16, n), device="cuda", dtype=torch.float8_e4m3fn),
        torch.float16,
    )
    global_scale = (
        sm70_nvfp4_marlin_process_global_scale(
            torch.full((e,), 0.125, device="cuda"), torch.float16
        )
        / factor
    )
    rows = m * topk
    # Upstream capacity need only align to int4, not to the expert block size.
    ids = torch.full((3 * block + 4,), rows, device="cuda", dtype=torch.int32)
    experts = torch.tensor(
        [-1, 0, 1 if sorted_prefix else -1, 123456], device="cuda", dtype=torch.int32
    )
    ids[:2] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    ids[block] = 2
    ids[2 * block] = 3
    count = torch.tensor([3 * block, 1], device="cuda", dtype=torch.int32)
    storage = torch.full((rows * n + 64,), 17.0, device="cuda", dtype=torch.float16)
    output = storage[: rows * n].view(rows, n)
    workspace = torch.zeros(4096, device="cuda", dtype=torch.int32)
    router = torch.ones(rows, device="cuda")

    def run(zero):
        return moe_wna16_marlin_gemm(
            a,
            output,
            weight,
            None,
            scales,
            global_scale,
            None,
            None,
            None,
            workspace,
            ids,
            experts,
            count,
            router,
            block,
            topk,
            False,
            False,
            get_scalar_types()[1].float4_e2m1f,
            m,
            n,
            k,
            expert_ids_are_sorted=sorted_prefix,
            zero_inactive_output=zero,
            has_inactive_prefix_count=True,
        )

    with published_topology(disable_overlap_schedule=True):
        for zero in (False, True):
            run(zero)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(zero)
            storage.fill_(17)
            graph.replay()
            expected = torch.full_like(output, 17)
            expected[2] = 8
            expected[3] = 8 if sorted_prefix else (0 if zero else 17)
            if zero:
                expected[:2] = 0
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert torch.equal(storage[-64:], torch.full_like(storage[-64:], 17))
            # Change the inactive route and live count after graph capture.
            ids[0] = 4
            count.copy_(torch.tensor([block, 1], device="cuda", dtype=torch.int32))
            storage.fill_(17)
            graph.replay()
            expected.fill_(17)
            if zero:
                expected[[1, 4]] = 0
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            count.zero_()
            storage.fill_(17)
            graph.replay()
            assert torch.equal(storage, torch.full_like(storage, 17))
            ids[0] = 0
            count.copy_(torch.tensor([3 * block, 1], device="cuda", dtype=torch.int32))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
