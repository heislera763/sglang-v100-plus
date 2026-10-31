"""Checkpoint FP8 KV metadata must not bypass GLM's SM70 cache policy."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.glm_dsa import sm70_glm_kv_cache_dtype


class TestV100KVCacheDtype(CustomTestCase):
    def test_minimax_target_and_native_draft_use_compute_dtype_cache(self):
        for architecture in (
            "MiniMaxM3SparseForConditionalGeneration",
            "DSparkMiniMaxDraftModel",
        ):
            model = SimpleNamespace(
                config=SimpleNamespace(architectures=[architecture]),
                quant_config=SimpleNamespace(kv_cache_quant_algo="FP8"),
            )
            args = dict(
                model=model,
                server_args_kv_cache_dtype="auto",
                model_dtype=torch.float16,
                is_draft_worker=architecture.startswith("DSpark"),
                is_dflash=True,
                speculative_draft_attention_backend="triton",
            )
            self.assertEqual(
                sm70_glm_kv_cache_dtype(configure_kv_cache_dtype, **args),
                ("auto", torch.float16),
            )
            self.assertEqual(model.quant_config.kv_cache_quant_algo, "FP8")

    def test_auto_fp8_recipe_keeps_glm_fp16_without_mutating_weights(self):
        for model_type in ("glm5_next", "glm5_next_text"):
            for draft in (False, True):
                with self.subTest(model_type=model_type, draft=draft):
                    quant = SimpleNamespace(kv_cache_quant_algo="FP8")
                    args = dict(
                        model=SimpleNamespace(
                            config=SimpleNamespace(model_type=model_type),
                            quant_config=quant,
                        ),
                        server_args_kv_cache_dtype="auto",
                        model_dtype=torch.float16,
                        is_draft_worker=draft,
                        is_dflash=False,
                        speculative_draft_attention_backend="triton",
                    )
                    self.assertEqual(
                        configure_kv_cache_dtype(**args)[1], torch.float8_e4m3fn
                    )
                    self.assertEqual(
                        sm70_glm_kv_cache_dtype(configure_kv_cache_dtype, **args),
                        ("auto", torch.float16),
                    )
                    self.assertEqual(quant.kv_cache_quant_algo, "FP8")

    def test_explicit_or_other_model_policies_keep_original_resolution(self):
        base = dict(
            model=SimpleNamespace(
                config=SimpleNamespace(model_type="glm5_next"),
                quant_config=SimpleNamespace(kv_cache_quant_algo="FP8"),
            ),
            server_args_kv_cache_dtype="auto",
            model_dtype=torch.float16,
            is_draft_worker=False,
            is_dflash=False,
            speculative_draft_attention_backend="triton",
        )
        for overrides in (
            dict(server_args_kv_cache_dtype="fp8_e4m3"),
            dict(model_dtype=torch.bfloat16),
            dict(model=None),
            dict(
                model=SimpleNamespace(
                    config=SimpleNamespace(model_type="qwen4_exp"),
                    quant_config=SimpleNamespace(kv_cache_quant_algo="FP8"),
                )
            ),
            dict(
                model=SimpleNamespace(
                    config=SimpleNamespace(model_type="glm5_next"),
                    quant_config=SimpleNamespace(kv_cache_quant_algo=None),
                )
            ),
            dict(is_draft_worker=True, speculative_draft_kv_cache_dtype="fp8_e4m3"),
        ):
            with self.subTest(overrides=overrides):
                args = base | overrides
                self.assertEqual(
                    sm70_glm_kv_cache_dtype(configure_kv_cache_dtype, **args),
                    configure_kv_cache_dtype(**args),
                )


if __name__ == "__main__":
    unittest.main()
