"""Volta primary operations and block-FP8 loading for DFlash2 drafts."""

import torch
from torch import nn

from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.models.dflash import DFlashGroupedConv, _grouped_conv

from .dispatch import reject_fallback


def initialize_attention(original, self, model_runner, *args, **kwargs):
    result = original(self, model_runner, *args, **kwargs)
    architecture = model_runner.model_config.hf_config.architectures[0]
    self._v100_minimax_draft = (
        model_runner.is_draft_worker
        and architecture
        in ("DSparkMiniMaxDraftModel", "MiniMaxDFlash2DraftModel", "DFlash2DraftModel")
        and (self.num_head, self.num_kv_head, self.dcp_size) in ((8, 1, 1), (16, 1, 1))
    )
    self._v100_dflash = self._v100_minimax_draft or (
        model_runner.is_draft_worker
        and model_runner.model_config.hf_config.architectures[0] == "DFlash2DraftModel"
    )
    if self._v100_dflash:
        if not self._v100_minimax_draft and (
            self.num_head,
            self.num_kv_head,
            self.dcp_size,
        ) != (4, 1, 1):
            raise ValueError(
                "V100 DFlash2 window attention requires TP8 without draft DCP"
            )
        self._v100_dflash_workspaces = {}
    return result


def forward_attention(original, self, layer, *args, **kwargs):
    if not self._v100_dflash:
        return original(self, layer, *args, **kwargs)
    from sglang.kernels.ops.attention.v100 import window_attention_v100

    q, k, v, out, kc, vc, qo, ki, ids, mask, causal, mi, mx, ks, vs, scale = args
    if getattr(self, "_v100_minimax_draft", False):
        from sglang.kernels.ops.attention.minimax_sparse.sm70 import (
            minimax_window_attention_sm70,
        )

        if (
            qo.numel() != 2
            or mask is not None
            or ks != 1.0
            or vs != 1.0
            or scale != 128**-0.5
            or kwargs.get("logit_cap", 0) != 0
            or kwargs.get("sinks") is not None
            or kwargs.get("score_mod") is not None
            or kwargs.get("aux_tensors") is not None
            or kwargs.get("xai_temperature_len", -1) > 0
        ):
            raise ValueError(
                "Unsupported SM70 MiniMax draft mask or scaling: "
                f"q={tuple(q.shape)}, causal={causal}, mask={mask is not None}, "
                f"scales={(ks, vs, scale)}, left_window={kwargs.get('sliding_window_size')}"
            )
        return minimax_window_attention_sm70(
            q,
            k,
            v,
            kc,
            vc,
            ids,
            ki,
            kwargs.get("sliding_window_size", -1),
            out,
            causal=causal,
        )
    if (
        q.shape != (8, 4, 128)
        or qo.numel() != 2
        or mask is not None
        or causal
        or ks != 1.0
        or vs != 1.0
        or kwargs.get("logit_cap", 0) != 0
        or not 0 < kwargs.get("sliding_window_size", -1) <= 2048
        or kwargs.get("sinks") is not None
        or kwargs.get("score_mod") is not None
        or kwargs.get("aux_tensors") is not None
        or kwargs.get("xai_temperature_len", -1) > 0
    ):
        raise ValueError(
            "Unsupported V100 DFlash2 bidirectional window geometry/mask: "
            f"q={tuple(q.shape)}, causal={causal}, mask={mask is not None}, "
            f"left_window={kwargs.get('sliding_window_size', -1)}"
        )
    stream = int(torch.cuda.current_stream(q.device).cuda_stream)
    workspace = self._v100_dflash_workspaces.setdefault(stream, {})
    return window_attention_v100(
        q,
        k,
        v,
        kc,
        vc,
        ids,
        ki,
        scale,
        left_window=kwargs["sliding_window_size"],
        workspace=workspace,
        out=out,
    )


class SM70DFlashGroupedConv(DFlashGroupedConv):
    def finish(self, hidden_states, coefficients):
        # Real GLM features overflow FP16 at the first output convolution.
        # Retain FP32 through its residual addition and the following norm.
        return _grouped_conv(
            hidden_states.float(),
            coefficients.float(),
            self.base_kernel[1].float(),
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )


class SM70DFlashResidualNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        self.weight = norm.weight
        self.variance_epsilon = norm.variance_epsilon
        self.gemma = isinstance(norm, GemmaRMSNorm)

    def forward(self, hidden_states, residual=None):
        summed = hidden_states.float()
        if residual is not None:
            summed = summed + residual.float()
        inv_rms = torch.rsqrt(
            summed.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        weight = self.weight.float()
        if self.gemma:
            weight = 1.0 + weight
        output = (summed * inv_rms * weight).to(torch.float16)
        return output if residual is None else (output, summed)


class SM70DFlashContextProjection(ColumnParallelLinear):
    """Shard the large context projection while keeping nn.Linear's interface.

    Each output column still reduces the full input K. Gather before hidden
    normalization; MiniMax avoids 315/378MiB of replicated FP16 weights per rank.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            gather_output=True,
            params_dtype=torch.float16,
        )
        self.in_features, self.out_features = in_features, out_features

    def forward(self, x):
        return super().forward(x)[0]


def candidate_topk(original, scores, k):
    if (
        scores.is_cuda
        and scores.ndim == 2
        and scores.dtype in (torch.float16, torch.float32)
        and 0 < k <= scores.shape[1]
        and torch.cuda.get_device_capability(scores.device) == (7, 0)
    ):
        # FlashInfer rejects SM70. CUDA Torch top-k is the selected Volta
        # backend, with the same sorted values/indices interface.
        return torch.topk(scores, k, dim=-1, largest=True, sorted=True)
    reject_fallback(
        "dflash.candidate_topk",
        "Volta CUDA FP16/FP32 score rows are required",
        scores=scores,
    )
    return original(scores, k)


def initialize_draft(original, self, config, quant_config=None, prefix=""):
    if config.hidden_size == 6144 and quant_config is None:
        self.context_projection_cls = SM70DFlashContextProjection
    if quant_config is not None:
        if quant_config.get_name() != "fp8" or quant_config.weight_block_size != [
            128,
            128,
        ]:
            raise ValueError(
                "SM70 DFlash2 supports unquantized or 128x128 block-FP8 draft weights"
            )
        # Mainline already builds quant-aware transformer projections. Enable
        # its replicated quant-aware context projection and weight prefixes as
        # well, so fc.weight_scale_inv is loaded instead of silently ignored.
        self.supports_quantization = True
    result = original(self, config, quant_config=quant_config, prefix=prefix)
    self.norm = SM70DFlashResidualNorm(self.norm)
    for layer in self.layers:
        layer.input_layernorm = SM70DFlashResidualNorm(layer.input_layernorm)
        layer.post_attention_layernorm = SM70DFlashResidualNorm(
            layer.post_attention_layernorm
        )
    return result


def forward_draft(
    original, self, input_ids, positions, forward_batch, input_embeds=None, **kwargs
):
    # EagerRunner's supplied-embedding kwargs currently round through BF16.
    # DFlash keeps the original embeddings on ForwardBatch; use those just as
    # the FP16 graph buffers do, without discarding FP16 mantissa bits first.
    if forward_batch.input_embeds is not None:
        input_embeds = forward_batch.input_embeds
    if input_embeds is not None:
        input_embeds = input_embeds.to(torch.float16)
    return original(
        self, input_ids, positions, forward_batch, input_embeds=input_embeds, **kwargs
    )
