"""Eager, single-request GLM control: repeat one FFN/mHC update, not attention.

Load with SGLang's existing forward-hook interface on model.layers.23. All arms
use the same hook and no-op logit processor. Request custom_params selects off
or repeat and its strength; separate cache_salt values isolate changed histories.
This deliberately experimental module is not installed by the V100 plugin.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

logger = logging.getLogger(__name__)


class PassThrough(CustomLogitProcessor):
    """Carry request custom_params through the ordinary sampling interface."""

    def __call__(self, logits, custom_param_list=None):
        return logits


def blend(anchor, refined, strength):
    """Keep zero strength exactly identical, including the caller's tensor."""
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Repeat strength must be finite and between zero and one")
    if anchor.shape != refined.shape or anchor.dtype != refined.dtype:
        raise ValueError("Repeat must preserve all residual streams and dtype")
    if strength == 0:
        return anchor
    return (anchor.float() + strength * (refined.float() - anchor.float())).to(
        anchor.dtype
    )


def request_policy(batch):
    info = batch.sampling_info
    params = None if info is None else info.custom_params
    if params is None:
        return None  # Ordinary startup/warmup has no research request.
    if len(params) != 1 or batch.batch_size != 1:
        raise ValueError("The frozen FFN pilot requires one request")
    policy = (params[0] or {}).get("frozen_ffn")
    if policy is None:
        return None
    if policy.get("mode") not in ("off", "repeat"):
        raise ValueError("Unknown frozen FFN mode")
    strength = float(policy.get("strength", 0))
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Repeat strength must be finite and between zero and one")
    return policy, strength


class RepeatFFN:
    def __init__(self, config):
        self.layer_id = int(config.get("layer", 23))
        self.observation_path = Path(config["observations"])
        self.observed = set()
        self.announced = set()

    def __call__(self, layer, args, output):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.moe.utils import sum_post_experts_output
        from sglang.srt.runtime_context import get_forward, get_parallel, get_spec

        batch = args[2]
        parsed = request_policy(batch)
        if parsed is None:
            return output
        policy, strength = parsed
        parallel = get_parallel()
        if get_spec().speculative_algorithm is not None:
            raise ValueError("MTP must be disabled for the frozen FFN pilot")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("The frozen FFN pilot requires eager execution")
        if not (
            parallel.tp_size == parallel.moe_tp_size == 8
            and parallel.moe_ep_size == parallel.pp_size == 1
            and layer.layer_id == self.layer_id
            and layer.config.mhc
            and layer.config.hc_mult == 4
            and not layer.mlp.reduce_results
        ):
            raise ValueError("Unexpected frozen FFN model/parallel layout")
        hidden, topk_indices = output
        stream = residual_batch.stream_of(batch)
        stream.check(hidden)
        if stream.pending is not None or hidden.shape[-1] != 16384:
            raise ValueError("Observe a completed four-stream FFN boundary")
        if hidden.dtype != torch.float16:
            raise ValueError("The frozen FFN pilot expects FP16 residuals")
        phase = "prefill" if batch.forward_mode.is_extend() else "decode"
        case = str(policy["case"])
        key = (case, phase)
        if key not in self.announced:
            logger.info(
                "Frozen FFN case=%s phase=%s mode=%s strength=%s rank=%s",
                case,
                phase,
                policy["mode"],
                strength,
                parallel.tp_rank,
            )
            self.announced.add(key)
        if policy["mode"] == "off":
            return output

        # Both the ordinary expert output and hc_post may borrow residual storage.
        # Keep the caller's original state untouched for the zero-strength control.
        working = hidden.clone()
        norm = layer.post_attention_layernorm
        x, h_res, h_post, norm_fused = layer.hc_ffn_pre(
            working, norm.weight.data, norm.variance_epsilon
        )
        if not norm_fused:
            x = norm(x)
        with get_forward().scoped(
            moe_output_buffer=None, defer_moe_finalize=False, mlp_reduce_scatter=False
        ):
            partial = layer.mlp(x, batch, None)
            if not isinstance(partial, torch.Tensor):
                raise TypeError("Expected an ordinary local expert output")
            complete = sum_post_experts_output(partial)
        refined = layer.hc_post(complete, working, h_res, h_post)
        mixed = blend(hidden, refined, strength)

        # These synchronizing observations run only in explicitly untimed probes.
        if policy.get("observe") and key not in self.observed and parallel.tp_rank == 0:
            a, b, c = hidden[-1].float(), refined[-1].float(), mixed[-1].float()
            record = dict(
                case=case,
                phase=phase,
                strength=strength,
                position=int(args[0][-1]),
                relative_delta=float((b - a).norm() / a.norm()),
                cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                mixed_relative_delta=float((c - a).norm() / a.norm()),
                finite=bool(torch.isfinite(b).all()),
                zero_identity=bool(torch.equal(hidden, mixed))
                if strength == 0
                else None,
            )
            self.observation_path.parent.mkdir(parents=True, exist_ok=True)
            with self.observation_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            self.observed.add(key)
        if strength == 0:
            return output
        residual_batch.set_written(mixed, batch)
        return mixed, topk_indices


def create_hook(config):
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )
    from sglang.srt.runtime_context import get_spec

    if get_spec().speculative_algorithm is not None or not all(
        check_cuda_graph_backend(phase, Backend.DISABLED)
        for phase in (Phase.PREFILL, Phase.DECODE)
    ):
        raise ValueError("Disable MTP and both CUDA-graph phases for this eager hook")
    return RepeatFFN(config)
