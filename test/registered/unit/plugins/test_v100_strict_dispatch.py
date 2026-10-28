"""Reject silent V100 fallbacks while retaining opt-out dispatch semantics."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

# The fork plugin is separately packaged; CPU CI need not install that wheel.
with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    import sglang_v100_plus
    from sglang_v100_plus import glm_mhc, mqa, qsa, quantization, runtime
    from sglang_v100_plus.dispatch import (
        V100FallbackError,
        eager_extend,
        in_prefill,
        prefill_scope,
    )
    from sglang_v100_plus.kernels import sm70_fp8_kv


class TestV100StrictDispatch(CustomTestCase):
    def test_prefill_scope_excludes_verify_and_restores_after_errors(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        def check(owner, batch):
            return in_prefill()

        self.assertFalse(in_prefill())
        with prefill_scope(True):
            self.assertTrue(
                eager_extend(
                    check, None, SimpleNamespace(forward_mode=ForwardMode.EXTEND)
                )
            )
            self.assertFalse(
                eager_extend(
                    check, None, SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
                )
            )
            self.assertTrue(in_prefill())
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with prefill_scope(False):
                    raise RuntimeError("test failure")
            self.assertTrue(in_prefill())
        self.assertFalse(in_prefill())

    def test_inactive_plugin_cannot_silently_ignore_strict_mode(self):
        with patch.object(
            sglang_v100_plus, "os", SimpleNamespace(environ={"SGLANG_V100_PLUS": "0"})
        ):
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
                self.assertIsNone(sglang_v100_plus.register())
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                with self.assertRaisesRegex(V100FallbackError, "plugin.activation"):
                    sglang_v100_plus.register()

    def test_uncovered_hc_rows_fail(self):
        """Extending small-batch coverage must not reopen silent fallback."""
        x = torch.ones(5, 10240, dtype=torch.float16)
        original = lambda owner, tensor: tensor.sum(-1)
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
            torch.testing.assert_close(runtime.mix(original, None, x), x.sum(-1))
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
            with self.assertRaisesRegex(
                V100FallbackError, r"qwen.hc_mix.*rows in \(1, 2, 3, 4\).*(5, 10240)"
            ):
                runtime.mix(original, None, x)

    def test_guarded_fallbacks_fail_before_delegating(self):
        """A missing check at any separate dispatch boundary permits silent work."""
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        weight = torch.ones(4, 2)
        residual = torch.ones(3, 4, 2)
        q = torch.ones(3, 4, 128)
        k = torch.ones(3, 1, 128)
        starts = torch.zeros(3, dtype=torch.int32)
        ends = torch.full((3,), 3, dtype=torch.int32)
        cases = [
            (
                "gemm.unquantized_linear",
                lambda: runtime.apply_unquant(
                    lambda owner, layer, a, bias: a @ layer.weight.T,
                    None,
                    SimpleNamespace(weight=weight),
                    x,
                ),
                x @ weight.T,
            ),
            (
                "moe.top10_router",
                lambda: runtime.route_top10(
                    lambda scores, *a, **kw: scores.sum(-1),
                    x,
                    None,
                    10,
                    scoring_func="softmax",
                ),
                x.sum(-1),
            ),
            (
                "qwen.hc_combine",
                lambda: runtime.combine(
                    lambda owner, a, state: a + state[0], None, x, (x, x)
                ),
                x + x,
            ),
            (
                "qwen.fp8_kv_store",
                lambda: runtime.store(
                    lambda owner, layer, loc, key, value, *a: key + value,
                    SimpleNamespace(dtype=torch.float16),
                    None,
                    None,
                    x,
                    x,
                ),
                x + x,
            ),
            (
                "glm.mhc_pre",
                lambda: glm_mhc.mhc_pre(lambda a: a.sum(-1), residual),
                residual.sum(-1),
            ),
            (
                "glm.mhc_post",
                lambda: glm_mhc.mhc_post(
                    lambda a, *args: a + 1, x, residual, None, None
                ),
                x + 1,
            ),
            (
                "qsa.indexer_decode",
                lambda: qsa.mqa_decode(
                    lambda a, *args, min_scored_len=0: a.sum(-1) + min_scored_len,
                    q,
                    k,
                    None,
                    None,
                    3,
                    min_scored_len=2,
                ),
                q.sum(-1) + 2,
            ),
            (
                "qsa.indexer_prefill",
                lambda: mqa.qsa_mqa_prefill(q, k, starts, ends),
                mqa.torch_qsa_mqa_prefill(q, k, starts, ends),
            ),
        ]
        for operation, call, expected in cases:
            with self.subTest(operation=operation):
                with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
                    torch.testing.assert_close(call(), expected)
                with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                    with self.assertRaises(V100FallbackError) as caught:
                        call()
                    self.assertIn(operation, str(caught.exception))
                    self.assertIn("device=cpu", str(caught.exception))

    def test_native_moe_alternate_dispatch_is_explicit(self):
        h = torch.ones(3, 2)
        dispatch = SimpleNamespace(
            hidden_states=h,
            topk_output=SimpleNamespace(topk_ids=torch.zeros(3, 10, dtype=torch.int32)),
        )
        quant = SimpleNamespace(w13_qweight=torch.zeros(1), w2_qweight=torch.zeros(1))
        original = lambda *args: (
            lambda dispatch, quant, config: dispatch.hidden_states * 2
        )
        call = quantization.moe_runner(original, None, "none", "marlin")
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
            torch.testing.assert_close(call(dispatch, quant, None), h * 2)
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
            with self.assertRaisesRegex(V100FallbackError, "moe.nvfp4_decode"):
                call(dispatch, quant, None)

    def test_cached_missing_cache_operator_cannot_silently_fall_back(self):
        x = torch.zeros(3, 1, 256, dtype=torch.float16)
        cache = torch.zeros(4, 1, 256, dtype=torch.uint8)
        locations = torch.arange(3)
        with patch.object(sm70_fp8_kv, "_get_op", return_value=None):
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
                self.assertFalse(
                    sm70_fp8_kv.write_fp8_e5m2_cache_sm70(x, x, cache, cache, locations)
                )
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                with self.assertRaisesRegex(
                    V100FallbackError, "operator is unavailable"
                ):
                    sm70_fp8_kv.write_fp8_e5m2_cache_sm70(x, x, cache, cache, locations)

    def test_qsa_attention_fallback_boundaries(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(qsa.QwenSparseAttnBackend)
        cache = torch.zeros(4, 1, 256, dtype=torch.uint8)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer: cache,
            get_value_buffer=lambda layer: cache,
        )
        backend.qsa_profile = None
        backend._resolve_metadata = lambda batch: None
        x = torch.ones(3, 6, 256)
        layer = SimpleNamespace(tp_q_head_num=6, head_dim=256, layer_id=0)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=None,
            extend_seq_lens_cpu=None,
        )
        indices = torch.zeros(3, 2, dtype=torch.int32)
        original = lambda owner, q, *args, **kwargs: q.flatten(1)
        with (
            patch.object(qsa.BaseQSA, "forward_extend", original),
            patch.object(qsa.BaseQSA, "_forward_paged_attention", original),
        ):
            cases = [
                (
                    "qsa.prefill",
                    lambda: backend.forward_extend(
                        x, None, None, layer, batch, topk_indices=indices
                    ),
                ),
                (
                    "qsa.paged_attention",
                    lambda: backend._forward_paged_attention(x, layer, batch, indices),
                ),
            ]
            for operation, call in cases:
                with self.subTest(operation=operation):
                    with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
                        torch.testing.assert_close(call(), x.flatten(1))
                    with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                        with self.assertRaisesRegex(V100FallbackError, operation):
                            call()

    def test_speculative_qsa_parent_delegation_is_not_a_fallback(self):
        """The base extend method routes verification into our paged backend."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(qsa.QwenSparseAttnBackend)
        backend.token_to_kv_pool = None
        x = torch.ones(3, 6, 256)
        layer = SimpleNamespace(tp_q_head_num=6, head_dim=256)
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        indices = torch.zeros(3, 2, dtype=torch.int32)
        with patch.object(
            qsa.BaseQSA, "forward_extend", lambda owner, q, *a, **kw: q.sum(-1)
        ):
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                torch.testing.assert_close(
                    backend.forward_extend(
                        x, None, None, layer, batch, topk_indices=indices
                    ),
                    x.sum(-1),
                )

    def test_qsa_prefill_dispatch_is_independent_of_model_layout(self):
        """Compatible prefill reuses its query/cache contract across model layouts."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(qsa.QwenSparseAttnBackend)
        cache = torch.zeros(1, 1, 256, dtype=torch.float8_e5m2)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: cache, get_value_buffer=lambda _: cache
        )
        backend.req_to_token_pool = SimpleNamespace(req_to_token=object())
        backend.qsa_profile = SimpleNamespace(budget=2048)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=[8192],
            extend_seq_lens_cpu=[128],
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8192], dtype=torch.int32),
        )
        layer = SimpleNamespace(
            tp_q_head_num=6, head_dim=256, layer_id=0, scaling=0.0625
        )
        x = torch.ones(128, 6, 256, dtype=torch.float16)
        indices = torch.arange(128, dtype=torch.int32)[:, None]

        def tensor_core(query, keys, values, table, requests, selected, length, scale):
            self.assertIs(keys, cache)
            self.assertIs(values, cache)
            self.assertEqual(length, 8192)
            self.assertEqual(scale, layer.scaling)
            self.assertTrue(torch.equal(selected, indices))
            return query * 2

        modules = {
            "sglang_v100_plus.kernels.qsa_prefill": SimpleNamespace(
                qsa_masked_prefill=tensor_core
            ),
            "sglang_v100_plus.kernels.qsa_cuda": SimpleNamespace(
                sm70_cuda_qsa_prefill=lambda query, *args: query * 3
            ),
        }
        fields = dict(
            model_path="dummy",
            quantization="fp8",
            tp_size=4,
            pp_size=2,
            disable_overlap_schedule=True,
            max_running_requests=1,
        )
        cases = [
            ({}, True),
            ({"speculative_algorithm": "EAGLE"}, True),
            ({"speculative_algorithm": "EAGLE", "pp_size": 1}, True),
            ({"pp_size": 1}, True),
            ({"tp_size": 8}, True),
            ({"tp_size": 8, "speculative_algorithm": "EAGLE"}, True),
            ({"quantization": "modelopt_fp4", "pp_size": 1}, True),
            ({"speculative_algorithm": "STANDALONE"}, True),
            ({"disable_overlap_schedule": False}, False),
            ({"speculative_algorithm": "EAGLE", "max_running_requests": 2}, False),
        ]
        with (
            patch.dict(sys.modules, modules),
            patch.object(backend, "_can_use_sm70_sparse_prefill", return_value=True),
            envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True),
        ):
            for changes, optimized in cases:
                with (
                    self.subTest(changes=changes),
                    get_context().override_server_args(**(fields | changes)),
                ):
                    actual = backend.forward_extend(
                        x,
                        None,
                        None,
                        layer,
                        batch,
                        save_kv_cache=False,
                        topk_indices=indices,
                    )
                    torch.testing.assert_close(
                        actual, (x * (2 if optimized else 3)).flatten(1), rtol=0, atol=0
                    )

    def test_nvfp4_prefill_geometry_does_not_leak_or_replace_user_tuning(self):
        """Process-wide Marlin overrides must end at the qualified shape.

        A stale prefill override can alter unrelated captured kernels; an
        explicit vendor override belongs to the caller, including split-K.
        Native bank comparisons cover the actual GPU geometry and arithmetic.
        """
        from sglang_v100_plus.kernels import moe_marlin

        names = moe_marlin._SM70_MARLIN_MOE_ENV_NAMES
        configured = moe_marlin._configure_sm70_nvfp4_stage
        fp4 = SimpleNamespace(dtype=torch.float8_e4m3fn)
        with (
            patch.dict(os.environ),
            patch.object(moe_marlin, "_IS_SM70", True),
            patch.object(moe_marlin, "_sm70_marlin_user_tuning", False),
            patch.object(moe_marlin, "_sm70_nvfp4_tuning_stage", None),
        ):
            for shape in (
                (2048, 512, 4096, 8),
                (2048, 1024, 4096, 8),
                (14032, 4096, 256, 1),
                (14032, 4096, 512, 1),
            ):
                m, n, k, topk = shape
                for block, scales in (
                    (32, fp4),
                    (64, SimpleNamespace(dtype=torch.float32)),
                ):
                    configured(fp4, 64, topk, m, n, k)
                    self.assertTrue(all(os.environ.get(name) for name in names))
                    configured(scales, block, topk, m, n, k)
                    self.assertTrue(all(name not in os.environ for name in names))
            configured(fp4, 64, 8, 2048, 1024, 4096)
            configured(fp4, 8, 8, 4, 1024, 4096)
            self.assertTrue(all(name not in os.environ for name in names))
            explicit = dict(zip(names, ("32x128x32x4x32x32x32", "2", "lane_vectors")))
            os.environ.update(explicit)
            with patch.object(moe_marlin, "_sm70_marlin_user_tuning", True):
                configured(fp4, 64, 8, 2048, 1024, 4096)
                self.assertEqual({name: os.environ[name] for name in names}, explicit)

    def test_nvfp4_routed_mma_preserves_vendor_override_and_epilogues(self):
        """A tuned replacement must not swallow unsupported Marlin contracts.

        Vendor overrides and optional bias/zero/EP semantics must still
        reach the declared Marlin backend; otherwise shape matching silently
        computes a different operation. GPU implementations are mock boundaries.
        """
        from sglang_v100_plus.kernels import moe_marlin

        a = torch.zeros(4, 4096, dtype=torch.float16)
        c = torch.empty(32, 1024, dtype=torch.float16)
        scalar = SimpleNamespace(id=31)
        fake_bank = SimpleNamespace(shape=(288, 256, 2048))
        scales = SimpleNamespace(dtype=torch.float8_e4m3fn)
        routed = SimpleNamespace(
            sm70_nvfp4_routed_gemm=lambda *args, **kwargs: args[1].fill_(7)
        )
        raw = lambda *args: args[1].fill_(3)
        base = dict(
            a=a,
            c_or_none=c,
            b_q_weight=fake_bank,
            b_bias_or_none=None,
            b_scales=scales,
            global_scale_or_none=torch.ones(288),
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=None,
            sorted_token_ids=torch.empty(256, dtype=torch.int32),
            expert_ids=torch.empty(32, dtype=torch.int32),
            num_tokens_post_padded=torch.empty(1, dtype=torch.int32),
            topk_weights=torch.ones(32),
            moe_block_size=8,
            top_k=8,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=scalar,
            size_m=4,
            size_n=1024,
            size_k=4096,
        )
        with (
            patch.object(moe_marlin, "_IS_SM70", True),
            patch.object(moe_marlin, "_load_marlin_v100_op", return_value=raw),
            patch.object(moe_marlin, "_configure_sm70_nvfp4_stage"),
            patch(
                "sglang.srt.layers.quantization.utils.get_scalar_types",
                return_value=(None, SimpleNamespace(float4_e2m1f=scalar)),
            ),
            patch.dict(
                sys.modules, {"sglang.kernels.ops.gemm.sm70_nvfp4_routed": routed}
            ),
            envs.SGLANG_OPT_SM70_NVFP4_GEMV.override(False),
        ):
            with patch.object(moe_marlin, "_sm70_marlin_user_tuning", False):
                torch.testing.assert_close(
                    moe_marlin.moe_wna16_marlin_gemm(**base), torch.full_like(c, 7)
                )
                # TP8 checkpoint shards have half the routed output/reduction
                # width. Missing either selection silently keeps padded M32.
                for m, topk, n, k in ((4, 8, 512, 4096), (32, 1, 4096, 256)):
                    result = moe_marlin.moe_wna16_marlin_gemm(
                        **dict(
                            base,
                            a=torch.zeros(m, k, dtype=torch.float16),
                            c_or_none=torch.empty(m * topk, n, dtype=torch.float16),
                            b_q_weight=SimpleNamespace(shape=(288, k // 16, n * 2)),
                            size_m=m,
                            size_n=n,
                            size_k=k,
                            top_k=topk,
                        )
                    )
                    torch.testing.assert_close(result, torch.full_like(result, 7))
                for change in (
                    dict(b_bias_or_none=torch.ones(1024)),
                    dict(b_zeros_or_none=torch.ones(1)),
                    dict(is_ep=True),
                    dict(is_k_full=False),
                    dict(is_zp_float=True),
                    dict(b_q_type=SimpleNamespace(id=17)),
                    dict(b_scales=SimpleNamespace(dtype=torch.float16)),
                    dict(global_scale_or_none=None),
                    dict(moe_block_size=16),
                    dict(size_m=8),
                ):
                    with self.subTest(change=tuple(change)):
                        torch.testing.assert_close(
                            moe_marlin.moe_wna16_marlin_gemm(**dict(base, **change)),
                            torch.full_like(c, 3),
                        )
                # True permits a split-K kernel, it does not require adding
                # into existing output. A full-K implementation is valid.
                torch.testing.assert_close(
                    moe_marlin.moe_wna16_marlin_gemm(**dict(base, use_atomic_add=True)),
                    torch.full_like(c, 7),
                )
            with patch.object(moe_marlin, "_sm70_marlin_user_tuning", True):
                torch.testing.assert_close(
                    moe_marlin.moe_wna16_marlin_gemm(**base), torch.full_like(c, 3)
                )


class TestFP16ExpertDispatchContract(CustomTestCase):
    def test_upstream_positional_options_stay_aligned(self):
        import inspect

        from sglang_v100_plus.fp16_moe import _OPTIONS

        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            invoke_fused_moe_kernel,
        )

        self.assertEqual(
            tuple(inspect.signature(invoke_fused_moe_kernel).parameters)[15:], _OPTIONS
        )

    def test_unqualified_fp16_epilogue_fails_before_kernel(self):
        import triton.language as tl
        from sglang_v100_plus.fp16_moe import invoke_fp16_moe

        a = torch.zeros(1, 16, dtype=torch.float16)
        b = torch.zeros(1, 16, 16, dtype=torch.float16)
        output = torch.zeros(1, 16, dtype=torch.float16)
        dummy = torch.zeros(1, dtype=torch.int32)
        args = (
            a,
            b,
            None,
            output,
            None,
            None,
            None,
            torch.ones(1, 1),
            dummy,
            dummy,
            dummy,
            dummy,
            False,
            1,
            {"BLOCK_SIZE_M": 16},
        )
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
            for options in ({"fuse_swiglu": True}, {"a_output_slices": 2}):
                with (
                    self.subTest(options=options),
                    self.assertRaises(V100FallbackError),
                ):
                    invoke_fp16_moe(
                        lambda *a, **kw: None,
                        *args,
                        compute_type=tl.float16,
                        **options,
                    )


if __name__ == "__main__":
    unittest.main()
