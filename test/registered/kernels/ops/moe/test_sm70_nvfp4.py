"""Checkpoint-native MoE reference, clamping and live graph routing on SM70."""

import os
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.moe.sm70_nvfp4 import sm70_nvfp4_moe
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=80, stage="jit-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Volta encoded NVFP4 experts",
)


@pytest.mark.parametrize("rows,width", [(1, 256), (2, 512), (3, 256), (4, 512)])
def test_checkpoint_reference_and_fresh_graph_routes(rows, width):
    from sglang_v100_plus.quantization import (
        _dense_repack,
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    assert os.environ.get("SGLANG_V100_MARLIN_DIR"), (
        "Install the Volta Marlin extension"
    )
    torch.manual_seed(831)
    experts, hidden = 9, 4096
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )

    def bank(n, k):
        packed = torch.randint(
            0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
        )
        scales = (torch.rand(experts, n, k // 16, device="cuda") * 0.04).to(
            torch.float8_e4m3fn
        )
        scales.view(torch.uint8)[:, :, ::7] = 0
        globals_native = torch.linspace(0.25, 1.123456, experts, device="cuda")
        encoded, factor = sm70_nvfp4_marlin_process_scales(
            scales.transpose(1, 2).contiguous(), torch.float16
        )
        globals_encoded = (
            sm70_nvfp4_marlin_process_global_scale(globals_native, torch.float16)
            / factor
        )
        codes = torch.stack(
            (packed.cpu().int() & 15, packed.cpu().int() >> 4), -1
        ).reshape(experts, n, k)
        # Independent of the encoded layout's word/nibble addressing.
        reference = (
            lookup[codes].double()
            * scales.cpu().double().repeat_interleave(16, -1)
            * globals_native.cpu().double()[:, None, None]
        )
        return _dense_repack(packed), encoded, globals_encoded, reference

    w13, s13, g13, reference13 = bank(2 * width, hidden)
    w2, s2, g2, reference2 = bank(hidden, width)
    a = torch.randn(rows, hidden, device="cuda", dtype=torch.float16) * 4
    ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:8] for _ in range(rows)]
    ).int()
    router = torch.rand(rows, 8, device="cuda")
    router.div_(router.sum(-1, keepdim=True))

    def run():
        return sm70_nvfp4_moe(a, ids, router, w13, w2, s13, s2, g13, g2, 10.0, 2.5)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(3):
        a.normal_(std=4)
        ids.copy_(
            torch.stack(
                [torch.randperm(experts, device="cuda")[:8] for _ in range(rows)]
            )
        )
        router.copy_(torch.rand_like(router))
        router.div_(router.sum(-1, keepdim=True))
        before = (a.clone(), ids.clone(), router.clone())
        graph.replay()
        eager = run()
        expected = torch.empty(rows, hidden, dtype=torch.float64)
        clamped = False
        cpu_a, cpu_ids, cpu_router = a.cpu().double(), ids.cpu(), router.cpu().double()
        for row in range(rows):
            parts = []
            for route in range(8):
                expert = cpu_ids[row, route]
                up = (cpu_a[row] @ reference13[expert].T).half()
                clamped |= bool((up.abs() > 10).any())
                gate, linear = up.chunk(2)
                # Keep both independent Half rounding boundaries.
                silu = torch.nn.functional.silu(gate.clamp(max=10).float()).half()
                activation = (silu.float() * linear.clamp(-10, 10).float()).half()
                down = (
                    activation.double() @ reference2[expert].T * cpu_router[row, route]
                ).half()
                parts.append(down.double())
            expected[row] = torch.stack(parts).sum(0) * 2.5
        assert clamped
        for result in (actual, eager):
            torch.testing.assert_close(
                result.cpu().double(), expected, rtol=0.01, atol=0.015
            )
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))
        for live, saved in zip((a, ids, router), before):
            assert torch.equal(live, saved)


class TestSM70NVFP4PrefillGather(CustomTestCase):
    def test_live_routes_and_unaligned_inputs(self):
        self._check_live_routes_and_unaligned_inputs(256, 64)

    def test_tp8_shards(self):
        for n, k, top_k in (
            (512, 4096, 8),
            (4096, 256, 8),
            (768, 6144, 4),
            (6144, 384, 4),
        ):
            with self.subTest(n=n, k=k):
                self._check_live_routes_and_unaligned_inputs(n, k, top_k)

    def _check_live_routes_and_unaligned_inputs(self, n, k, gate_top_k=8):
        """Cached rows must refresh on replay; wide loads must respect alignment.

        Rotating expert tails, negative expert IDs and a shrinking live count
        expose stale route caches and writes through padded or masked rows.
        """
        from sgl_kernel.scalar_type import scalar_types
        from sglang_v100_plus.kernels.moe_marlin import _load_marlin_v100_op
        from sglang_v100_plus.quantization import (
            _dense_repack,
            sm70_nvfp4_marlin_process_global_scale,
            sm70_nvfp4_marlin_process_scales,
        )

        self.enterContext(
            patch.dict(
                os.environ,
                SM70_MARLIN_MOE_CTA_GEOMETRY="64x256x32x4x64x64x32",
                SM70_MARLIN_MOE_SPLIT_K="1",
                SM70_MARLIN_MOE_METADATA_CACHE="vector_words",
            )
        )
        op = _load_marlin_v100_op()
        torch.manual_seed(651)
        experts, routes = 12, 512
        packed = torch.randint(
            0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda"
        )
        native_scales = (
            torch.rand(experts, n, k // 16, device="cuda") * 0.03 + 0.002
        ).to(torch.float8_e4m3fn)
        native_global = torch.linspace(0.3, 0.9, experts, device="cuda")
        scales, factor = sm70_nvfp4_marlin_process_scales(
            native_scales.transpose(1, 2).contiguous(), torch.float16
        )
        global_scale = (
            sm70_nvfp4_marlin_process_global_scale(native_global, torch.float16)
            / factor
        )
        lookup = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float64,
        )
        codes = torch.stack((packed.cpu().int() & 15, packed.cpu().int() >> 4), -1)
        reference = (
            lookup[codes.reshape(experts, n, k)]
            * native_scales.cpu().double().repeat_interleave(16, -1)
            * native_global.cpu().double()[:, None, None]
        )
        weight = _dense_repack(packed)
        workspace = torch.zeros(320, dtype=torch.int32, device="cuda")
        counts = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 175]
        for top_k in (gate_top_k, 1):
            m = routes // top_k
            sorted_ids = torch.full(
                (routes + experts * 64,), routes, dtype=torch.int32, device="cuda"
            )
            expert_ids = torch.full(
                (sorted_ids.numel() // 64,), 123456, dtype=torch.int32, device="cuda"
            )
            count = torch.zeros(1, dtype=torch.int32, device="cuda")
            router = torch.rand(routes, device="cuda")
            storage = torch.full(
                (routes * n + 128,), 17.0, dtype=torch.float16, device="cuda"
            )
            out = storage[64:-64].view(routes, n)

            def update(shift, masked):
                perm = torch.randperm(routes)
                route_expert = torch.empty(routes, dtype=torch.long)
                ids, banks, offset = [], [], 0
                for expert, rows in enumerate(counts[shift:] + counts[:shift]):
                    live = perm[offset : offset + rows]
                    offset += rows
                    route_expert[live] = -1 if expert == masked else expert
                    padded = (rows + 63) // 64 * 64
                    if padded:
                        block = torch.full((padded,), routes, dtype=torch.int32)
                        block[:rows] = live.int()
                        ids.append(block)
                        banks.extend(
                            [-1 if expert == masked else expert] * (padded // 64)
                        )
                ids = torch.cat(ids)
                sorted_ids.fill_(routes)
                sorted_ids[: ids.numel()].copy_(ids)
                expert_ids.fill_(123456)
                expert_ids[: len(banks)].copy_(torch.tensor(banks, dtype=torch.int32))
                count.fill_(ids.numel())
                return route_expert

            # An offset view remains contiguous but cannot use aligned uint4 loads.
            for extra, offset in ((0, 0), (0, 1)):
                with self.subTest(top_k=top_k, stride_extra=extra, offset=offset):
                    backing = (
                        torch.randn(
                            m * (k + extra) + offset, device="cuda", dtype=torch.float16
                        )
                        * 0.25
                    )
                    a = backing.as_strided((m, k), (k + extra, 1), offset)

                    def run():
                        out.zero_()
                        op(
                            a,
                            out,
                            weight,
                            None,
                            scales,
                            None,
                            global_scale,
                            None,
                            None,
                            None,
                            workspace,
                            sorted_ids,
                            expert_ids,
                            count,
                            router,
                            64,
                            top_k,
                            top_k == 1,
                            scalar_types.float4_e2m1f.id,
                            m,
                            n,
                            k,
                            True,
                            False,
                            False,
                            False,
                            -1,
                            -1,
                            -1,
                        )

                    update(0, -1)
                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    for shift, masked in ((0, -1), (3, 2), (7, -1), (1, 0)):
                        route_expert = update(shift, masked)
                        a.normal_(std=0.25)
                        router.uniform_(0.1, 0.9)
                        saved = [
                            t.clone()
                            for t in (backing, sorted_ids, expert_ids, count, router)
                        ]
                        graph.replay()
                        actual = out.clone()
                        expected = torch.zeros(routes, n, dtype=torch.float64)
                        cpu_a, cpu_router = a.cpu().double(), router.cpu().double()
                        for expert in range(experts):
                            selected = torch.where(route_expert == expert)[0]
                            if selected.numel():
                                values = cpu_a[selected // top_k] @ reference[expert].T
                                if top_k == 1:
                                    values *= cpu_router[selected, None]
                                expected[selected] = values
                        torch.testing.assert_close(
                            actual.cpu().double(), expected, rtol=0.001, atol=0.0002
                        )
                        run()
                        self.assertTrue(
                            torch.equal(actual.view(torch.uint8), out.view(torch.uint8))
                        )
                        for live, before in zip(
                            (backing, sorted_ids, expert_ids, count, router), saved
                        ):
                            self.assertTrue(torch.equal(live, before))
                        self.assertTrue(
                            bool(
                                (storage[:64] == 17).all()
                                and (storage[-64:] == 17).all()
                            )
                        )
                    count.zero_()
                    expert_ids.fill_(123456)
                    graph.replay()
                    self.assertTrue(bool((out == 0).all()))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
