"""PEFT parameter adapters must retain both factors and every declared target."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.lora import adapter_factors, load_adapter


class TestV100LoRA(CustomTestCase):
    def test_dense_only_adapter_retains_declared_stage_targets(self):
        """A dense-only adapter must load without inventing expert targets."""
        from sglang.srt.layers.linear import RowParallelLinear

        model = type("Qwen4ExpForConditionalGeneration", (torch.nn.Module,), {})()
        model.model = torch.nn.Module()
        model.model.start_layer, model.model.end_layer = 24, 25
        layer = torch.nn.Module()
        layer.linear_attn = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.shared_expert = torch.nn.Module()
        model.model.layers = torch.nn.ModuleDict({"24": layer})
        factors = {}
        for parent, name, key in (
            (layer.linear_attn, "out_proj", "linear_attn.out_proj"),
            (layer.mlp.shared_expert, "down_proj", "mlp.shared_expert.down_proj"),
        ):
            module = RowParallelLinear.__new__(RowParallelLinear)
            torch.nn.Module.__init__(module)
            module.input_size, module.output_size = 8, 4
            module.input_size_per_partition = 2
            module.tp_group = SimpleNamespace(rank_in_group=1, world_size=4)
            module.weight = torch.nn.Parameter(torch.zeros(4, 2))
            module.quant_method = SimpleNamespace(apply=lambda *args: None)
            setattr(parent, name, module)
            factors[f"model.layers.24.{key}"] = {
                "A": torch.arange(8, dtype=torch.float32).view(1, 8),
                "B": torch.arange(4, dtype=torch.float32).view(4, 1),
            }
        # The other pipeline stage owns this declared projection.
        factors["model.layers.0.linear_attn.out_proj"] = factors[
            "model.layers.24.linear_attn.out_proj"
        ]
        config = SimpleNamespace(
            hf_config=SimpleNamespace(v100_static_lora="adapter"),
            num_hidden_layers=48,
            is_draft_model=False,
            quantization="fp8",
        )
        with (
            patch("sglang_v100_plus.lora.adapter_factors", return_value=factors),
            patch(
                "sglang.srt.runtime_context.get_lora",
                return_value=SimpleNamespace(enable_lora=False),
            ),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
            ),
            patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=SimpleNamespace(disable_overlap_schedule=True),
            ),
            patch(
                "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
                return_value=None,
            ),
        ):
            load_adapter(
                lambda *args, **kwargs: model,
                None,
                model_config=config,
                device_config=None,
            )
            expected = {name for name in factors if ".24." in name}
            self.assertEqual(set(model._sm70_static_lora_targets), expected)
            for name in expected:
                module = dict(model.named_modules())[name]
                torch.testing.assert_close(
                    module._sm70_lora_a, factors[name]["A"][:, 2:4], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    module._sm70_lora_b, factors[name]["B"].t(), rtol=0, atol=0
                )
            factors["model.embed_tokens"] = factors[
                "model.layers.0.linear_attn.out_proj"
            ]
            with self.assertRaisesRegex(ValueError, "unsupported projection targets"):
                load_adapter(
                    lambda *args, **kwargs: model,
                    None,
                    model_config=config,
                    device_config=None,
                )

    def test_packed_parameter_targets_and_missing_factors(self):
        prefix = "model.language_model.layers.24."
        config = {
            "r": 1,
            "lora_alpha": 1,
            "base_model_name_or_path": "Qwen/Qwen3.8-Flash-Next",
            "target_modules": [
                prefix + "linear_attn.out_proj",
                prefix + "self_attn.o_proj",
            ],
            "target_parameters": [prefix + "mlp.experts.down_proj"],
        }
        tensors = {
            "base_model.model."
            + prefix
            + "linear_attn.out_proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model."
            + prefix
            + "linear_attn.out_proj.lora_B.weight": torch.tensor([[3.0], [4.0]]),
            "base_model.model." + prefix + "mlp.experts.lora_A.weight": torch.arange(
                6.0
            ).view(3, 2),
            "base_model.model." + prefix + "mlp.experts.lora_B.weight": torch.arange(
                12.0
            ).view(4, 3),
        }
        for factor in ("A", "B"):
            tensors[
                "base_model.model." + prefix + f"self_attn.o_proj.lora_{factor}.weight"
            ] = torch.ones(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text(json.dumps(config))
            save_file(tensors, root / "adapter_model.safetensors")
            factors = adapter_factors(root)
            self.assertEqual(
                set(factors),
                {
                    "model.layers.24.linear_attn.out_proj",
                    "model.layers.24.mlp.experts.down_proj",
                    "model.layers.24.o_proj",
                },
            )
            for name, tensor in tensors.items():
                if ".mlp.experts." in name:
                    factor = name.rsplit(".lora_", 1)[1][0]
                    torch.testing.assert_close(
                        factors["model.layers.24.mlp.experts.down_proj"][factor],
                        tensor,
                    )
            tensors.pop(next(k for k in tensors if "experts.lora_B" in k))
            save_file(tensors, root / "adapter_model.safetensors")
            with self.assertRaisesRegex(ValueError, "paired factor"):
                adapter_factors(root)


if __name__ == "__main__":
    unittest.main()
