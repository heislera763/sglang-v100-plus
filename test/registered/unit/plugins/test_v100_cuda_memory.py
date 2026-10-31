"""External module loads must not OOM while idle Torch blocks are reclaimable."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import (
    ParallelContext,
    RuntimeContext,
    get_context,
    get_parallel,
    use_context,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus import cuda_memory


class _Allocator:
    """Model independent driver allocation and reclaimable cached segments."""

    def __init__(self, *, capturing=False, free=786432):
        self.capturing = capturing
        self.free = free
        self.allocated = 30500569088
        self.reserved = 31358713856
        self.non_releasable = 88489984

    def mem_get_info(self):
        return self.free, 34072559616

    def is_current_stream_capturing(self):
        return self.capturing

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return self.allocated

    def empty_cache(self):
        if self.capturing:
            raise RuntimeError("Cannot release allocator blocks during capture")
        released = self.reserved - self.allocated - self.non_releasable
        self.free += released
        self.reserved -= released

    def load_module(self):
        # An external driver allocation cannot invoke Torch's OOM retry.
        if self.free < 32 << 20:
            raise RuntimeError("CUDA module load out of memory")
        self.free -= 32 << 20
        return "same kernel"


class TestV100ModuleMemory(CustomTestCase):
    def test_direct_kernel_without_serving_context_needs_no_allocator_policy(self):
        with use_context(RuntimeContext(parallel=ParallelContext())):
            cuda_memory.reclaim_for_triton_load(None, None, "kernel", {}, "hash")

    def test_cold_module_fits_after_reclaim_without_changing_live_allocations(self):
        allocator = _Allocator()
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            allocator.load_module()
        live = allocator.allocated
        config = SimpleNamespace(features=SimpleNamespace(enable_memory_saver=False))
        with (
            get_context().override_server_args(model_path="dummy"),
            patch.object(cuda_memory.torch, "cuda", allocator),
            patch.object(cuda_memory, "get_exec", return_value=config),
        ):
            cuda_memory.reclaim_for_triton_load(None, None, "kernel", {}, "hash")
        self.assertEqual(allocator.load_module(), "same kernel")
        self.assertEqual(allocator.allocated, live)
        self.assertEqual(allocator.reserved - live, allocator.non_releasable)

    def test_capture_custom_arena_and_plenty_of_headroom_preserve_reservations(self):
        for capture, custom, free in (
            (True, False, 786432),
            (False, True, 786432),
            (False, False, 2 << 30),
        ):
            with self.subTest(capture=capture, custom=custom, free=free):
                allocator = _Allocator(capturing=capture, free=free)
                config = SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=custom)
                )
                before = allocator.reserved, allocator.free
                with (
                    get_context().override_server_args(model_path="dummy"),
                    patch.object(cuda_memory.torch, "cuda", allocator),
                    patch.object(cuda_memory, "get_exec", return_value=config),
                ):
                    cuda_memory.reclaim_for_triton_load(
                        None, None, "kernel", {}, "hash"
                    )
                self.assertEqual((allocator.reserved, allocator.free), before)


class TestSingleSessionReservation(CustomTestCase):
    """Explicit pools used to be silently clamped, reducing session capacity."""

    def test_native_context_tail_needs_admission_headroom(self):
        for context, page_size, chunk_size, capacities in (
            (262144, 64, 4352, ((262208, 0), (262272, 32))),
            # DCP2 widens a 256-token physical page to 512 logical tokens.
            # The first full GLM prefill succeeded but clipped generation to0.
            (524288, 512, 2048, ((524800, 0), (525312, 32))),
        ):
            allocator = SimpleNamespace(page_size=page_size)
            for capacity, expected in capacities:
                with self.subTest(context=context, capacity=capacity):
                    self.assertEqual(
                        BaseTokenToKVPoolAllocator.max_new_tokens_for_memory(
                            allocator,
                            context - 38,  # 32 output + 4 draft + 2 guard positions
                            32,
                            token_capacity=capacity,
                            sliding_window_size=None,
                            chunk_size=chunk_size,
                        ),
                        expected,
                    )

    def test_shortfall_rejected_after_pipeline_minimum(self):
        def remote_stage_minimum(tensor, **_):
            tensor.fill_(8192)

        for cached in (False, True):
            with (
                self.subTest(cached=cached),
                get_context().override_server_args(
                    model_path="dummy",
                    max_running_requests=1,
                    disable_radix_cache=not cached,
                    max_total_tokens=12288,
                    page_size=64,
                    pp_size=2,
                ),
                get_parallel().override(
                    world_group=SimpleNamespace(cpu_group=object())
                ),
                patch("torch.distributed.all_reduce", remote_stage_minimum),
                self.assertRaisesRegex(RuntimeError, "requires 12288.*only 8192"),
            ):
                cuda_memory.require_single_session_capacity(
                    KVCacheConfigurator._apply_token_constraints, None, 65536
                )

    def test_page_rounding_cannot_hide_a_shortfall(self):
        with (
            get_context().override_server_args(
                model_path="dummy",
                max_running_requests=1,
                disable_radix_cache=True,
                max_total_tokens=12289,
                page_size=64,
                pp_size=1,
            ),
            self.assertRaisesRegex(RuntimeError, "requires 12289.*only 12288"),
        ):
            cuda_memory.require_single_session_capacity(
                KVCacheConfigurator._apply_token_constraints, None, 65536
            )

    def test_explicit_cap_and_other_serving_policies(self):
        for requests, cached, requested, budget, expected in (
            (1, False, 262208, 1048576, 262208),
            (2, False, 12288, 8192, 8192),
            (1, True, 262208, 1048576, 262208),
            (1, False, None, 8192, 8192),
        ):
            with (
                self.subTest(requests=requests, cached=cached, requested=requested),
                get_context().override_server_args(
                    model_path="dummy",
                    max_running_requests=requests,
                    disable_radix_cache=not cached,
                    max_total_tokens=requested,
                    page_size=64,
                    pp_size=1,
                ),
            ):
                self.assertEqual(
                    cuda_memory.require_single_session_capacity(
                        KVCacheConfigurator._apply_token_constraints, None, budget
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
