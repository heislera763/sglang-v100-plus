"""Fixed rank-one Qwen adapters over the ordinary packed FP8 V100 model."""

import json
import logging
from copy import copy
from functools import partial
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _runtime_name(name):
    return (
        name.removeprefix("base_model.model.")
        .replace("model.language_model.", "model.")
        .replace(".self_attn.", ".")
    )


def adapter_factors(path):
    """Normalize PEFT's packed expert A/B layout without changing its factors."""
    from safetensors import safe_open

    path = Path(path)
    config = json.loads((path / "adapter_config.json").read_text())
    if (
        config["r"] != 1
        or config["lora_alpha"] != 1
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
        or config.get("use_dora", False)
        or config.get("fan_in_fan_out", False)
        or config.get("bias", "none") != "none"
        or config["base_model_name_or_path"] != "Qwen/Qwen3.8-Flash-Next"
    ):
        raise ValueError("Static SM70 Qwen adapters require ordinary rank-one factors")
    targets = set(config["target_modules"] + config.get("target_parameters", []))
    targets = {_runtime_name(name) for name in targets}
    factors = {}
    with safe_open(path / "adapter_model.safetensors", framework="pt") as file:
        tensor_names = file.keys()
        for key in tensor_names:
            name = _runtime_name(key)
            module, suffix = name.rsplit(".lora_", 1)
            if suffix not in ("A.weight", "B.weight"):
                raise ValueError(f"Unsupported adapter tensor {key}")
            if module.endswith(".mlp.experts"):
                module += ".down_proj"
            if module not in targets:
                raise ValueError(f"Adapter tensor has no declared target {key}")
            bank = factors.setdefault(module, {})
            factor = suffix[0]
            if factor in bank:
                raise ValueError(f"Duplicate adapter factor {key}")
            tensor = file.get_tensor(key)
            dtype = torch.float32 if tensor.dtype == torch.float32 else torch.float16
            bank[factor] = tensor.to(dtype)
            if bank[factor].ndim != 2 or not torch.isfinite(bank[factor]).all():
                raise ValueError(f"Adapter factor is not a finite matrix {key}")
    if set(factors) != targets or any(
        set(bank) != {"A", "B"} for bank in factors.values()
    ):
        raise ValueError("Adapter targets and paired factor tensors disagree")
    return factors


def load_adapter(original, self, *, model_config, device_config):
    path = getattr(model_config.hf_config, "v100_static_lora", None)
    model = original(self, model_config=model_config, device_config=device_config)
    if path is None or model_config.is_draft_model:
        return model

    from sglang.srt.layers.linear import RowParallelLinear, get_group_rank_size
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.runtime_context import get_exec, get_lora, get_schedule

    if get_lora().enable_lora:
        raise ValueError(
            "v100_static_lora and --enable-lora select incompatible adapter managers"
        )

    if (
        type(model).__name__ != "Qwen4ExpForConditionalGeneration"
        or model_config.quantization != "fp8"
        or get_exec().moe.enable_eplb
        or not get_schedule().disable_overlap_schedule
    ):
        raise ValueError(
            "Static Qwen LoRA requires packed FP8 with fixed expert placement and serial forwards"
        )

    factors = adapter_factors(path)
    supported = {
        f"model.layers.{layer}.{projection}"
        for layer in range(model_config.num_hidden_layers)
        for projection in (
            "o_proj",
            "linear_attn.out_proj",
            "mlp.shared_expert.down_proj",
            "mlp.experts.down_proj",
        )
    }
    if not factors or not set(factors).issubset(supported):
        raise ValueError("Static Qwen adapter has unsupported projection targets")
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    placement = get_global_expert_location_metadata()
    modules = dict(model.named_modules())
    start, end = model.model.start_layer, model.model.end_layer
    expected = {
        name
        for name in factors
        if name.startswith("model.layers.") and start <= int(name.split(".")[2]) < end
    }
    applied = []
    for name in sorted(expected):
        expert = name.endswith(".mlp.experts.down_proj")
        module_name = name.removesuffix(".down_proj") if expert else name
        module = modules.get(module_name)
        a, b = factors[name]["A"], factors[name]["B"]
        device = next(module.parameters()).device if module is not None else None
        if expert:
            if (
                not isinstance(module, FusedMoE)
                or a.shape != (512, 640)
                or b.shape != (2560, 512)
            ):
                raise ValueError(f"Unsupported packed expert adapter {name}")
            if (
                module.moe_tp_size != 1
                or module.num_local_experts * module.moe_ep_size != 512
            ):
                raise ValueError(
                    "Static Qwen expert adapters require whole experts under EP"
                )
            offset = module._expert_storage_rank * module.num_local_experts
            stop = offset + module.num_local_experts
            layer_id = int(name.split(".")[2])
            logical = (
                torch.arange(offset, stop)
                if placement is None
                else placement.physical_to_logical_map_cpu[layer_id, offset:stop].long()
            )
            a = a[logical].contiguous().to(device)
            b = b.t()[logical].contiguous().to(device)
            module.register_buffer("_sm70_lora_a", a)
            module.register_buffer("_sm70_lora_b", b)
            module._sm70_lora_down = (a, b)
        else:
            if (
                not isinstance(module, RowParallelLinear)
                or a.shape != (1, module.input_size)
                or b.shape != (module.output_size, 1)
            ):
                raise ValueError(f"Unsupported row-parallel adapter {name}")
            rank, _ = get_group_rank_size(module.tp_group)
            k = module.input_size_per_partition
            module.register_buffer(
                "_sm70_lora_a", a[:, rank * k : (rank + 1) * k].contiguous().to(device)
            )
            module.register_buffer("_sm70_lora_b", b.t().contiguous().to(device))
            # Quantization methods can be shared. Keep the adapter's callable
            # local to this projection and retain the ordinary FP8 dispatch.
            module.quant_method = copy(module.quant_method)
            module.quant_method.apply = partial(apply_dense, module.quant_method.apply)
        applied.append(name)
    if set(applied) != expected:
        raise ValueError("Static Qwen adapter did not cover all local projections")
    model._sm70_static_lora_targets = tuple(applied)
    logger.info(
        "Loaded static Qwen LoRA %s on %d projections for layers [%d,%d)",
        path,
        len(applied),
        start,
        end,
    )
    return model


def apply_dense(original, layer, x, bias=None):
    from sglang.kernels.ops.lora.sm70_rank_one import rank_one_add

    output = original(layer, x, bias)
    rank_one_add(x, output, layer._sm70_lora_a, layer._sm70_lora_b)
    return output


def install_hooks(hooks, hook_type):
    hooks.extend(
        [
            (
                "sglang.srt.model_loader.loader.DefaultModelLoader.load_model",
                load_adapter,
                hook_type.AROUND,
            ),
        ]
    )
