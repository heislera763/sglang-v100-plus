"""PEFT parameter adapters must retain both factors and every declared target."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.lora import adapter_factors


class TestV100LoRA(CustomTestCase):
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
                        tensor.half(),
                    )
            tensors.pop(next(k for k in tensors if "experts.lora_B" in k))
            save_file(tensors, root / "adapter_model.safetensors")
            with self.assertRaisesRegex(ValueError, "paired factor"):
                adapter_factors(root)


if __name__ == "__main__":
    unittest.main()
