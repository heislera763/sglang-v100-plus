"""Exact generated-history penalties for linear EAGLE verification."""

from weakref import WeakKeyDictionary

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_buffer


def _history(req):
    cache = get_buffer("v100_eagle_penalty_histories", WeakKeyDictionary)
    coefficient = np.float32(req.sampling_params.frequency_penalty)
    state = cache.get(req)
    history = req.output_ids
    if (
        state is None
        or state[0] != coefficient
        or list(history[: len(state[1])]) != state[1]
    ):
        state = coefficient, [], {}
        cache[req] = state
    _, consumed, frequencies = state
    for token in history[len(consumed) :]:
        frequencies[token] = np.float32(
            frequencies.get(token, np.float32(0)) + coefficient
        )
        consumed.append(token)
    return frequencies


def prepared_penalties(info):
    """Keep only the immutable penalty kinds/stop mask across forward copies."""
    # Upstream post-verify graphs carry a compact sampling view without an
    # orchestrator. Forward copies may carry our prepared penalty metadata.
    orchestrator = getattr(info, "penalizer_orchestrator", None)
    if orchestrator is None:
        return getattr(info, "_v100_linear_penalties", ())
    return tuple(
        (type(penalizer), getattr(penalizer, "stop_token_penalties", None))
        for penalizer in orchestrator.penalizers.values()
        if penalizer.is_prepared()
    )


def copy_penalty_metadata(original, info):
    result = original(info)
    penalties = prepared_penalties(info)
    if penalties:
        result._v100_linear_penalties = penalties
    return result


def apply_linear_penalties(verify_input, batch, logits):
    """Apply the ordinary penalizer order to each row's actual causal prefix."""
    from sglang.srt.sampling.penaltylib.frequency_penalty import (
        BatchedFrequencyPenalizer,
    )
    from sglang.srt.sampling.penaltylib.min_new_tokens import (
        BatchedMinNewTokensPenalizer,
    )
    from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        BatchedRepetitionPenalizer,
    )

    if (
        verify_input.tree_topk != 1
        or verify_input.draft_token_num != verify_input.max_tree_depth
    ):
        raise NotImplementedError(
            "Exact EAGLE penalties require a complete linear chain"
        )
    width = verify_input.draft_token_num
    bs, vocab = len(batch.reqs), logits.shape[-1]
    values = logits.view(bs, width, vocab)
    candidates = verify_input.draft_token.view(bs, width).long()
    positions = verify_input.positions.view(bs, width)
    order = positions.argsort(dim=1)
    ordered_positions = positions.gather(1, order)
    seen = torch.zeros_like(values, dtype=torch.bool)
    frequency = torch.zeros_like(values, dtype=torch.float32)
    coefficients = torch.tensor(
        [r.sampling_params.frequency_penalty for r in batch.reqs],
        device=logits.device,
        dtype=torch.float32,
    )
    for row, req in enumerate(batch.reqs):
        frequencies = _history(req)
        if frequencies:
            ids = torch.tensor(
                list(frequencies), device=logits.device, dtype=torch.long
            )
            seen[row, :, ids] = True
            frequency[row, :, ids] = torch.tensor(
                list(frequencies.values()), device=logits.device, dtype=torch.float32
            )
    # Root is already present in Req.output_ids. Include each subsequent input
    # token in the rows at or after its position; physical rows may be reordered.
    for depth in range(1, width):
        ids = candidates.gather(1, order[:, depth : depth + 1])[:, None, :].expand(
            bs, width, 1
        )
        active = (positions >= ordered_positions[:, depth : depth + 1])[:, :, None]
        frequency.scatter_add_(
            2, ids, active.to(torch.float32) * coefficients[:, None, None]
        )
        seen.scatter_(2, ids, seen.gather(2, ids) | active)
    params = [r.sampling_params for r in batch.reqs]
    for kind, stop_penalties in prepared_penalties(batch.sampling_info):
        if kind is BatchedFrequencyPenalizer:
            values.sub_(frequency)
        elif kind is BatchedPresencePenalizer:
            factors = values.new_tensor([p.presence_penalty for p in params])[
                :, None, None
            ]
            values.sub_(seen * factors)
        elif kind is BatchedRepetitionPenalizer:
            factors = values.new_tensor([p.repetition_penalty for p in params])[
                :, None, None
            ]
            scale = torch.where(seen, factors, 1.0)
            values.copy_(torch.where(values < 0, values * scale, values / scale))
        elif kind is BatchedMinNewTokensPenalizer:
            lengths = positions - ordered_positions[:, :1]
            lengths = (
                lengths
                + positions.new_tensor([len(r.output_ids) for r in batch.reqs])[:, None]
            )
            limits = positions.new_tensor([p.min_new_tokens for p in params])[:, None]
            values.add_(
                torch.where(
                    (lengths < limits)[:, :, None],
                    stop_penalties[:, None, :],
                    0.0,
                )
            )
        else:
            raise NotImplementedError(
                "Exact EAGLE penalties do not support this penalizer"
            )


def exact_eagle_sample(
    original,
    verify_input,
    batch,
    logits_output,
    grammar_mask=None,
    uno_target_max_top_k=None,
    *,
    watermark_state=None,
    watermark_full_mask=True,
    simulated_acc_len=None,
    simulated_acc_len_rng=None,
    sparse_bonus_argmax=False,
):
    if batch.forward_mode.is_idle():
        return original(
            verify_input,
            batch,
            logits_output,
            grammar_mask,
            uno_target_max_top_k=uno_target_max_top_k,
            watermark_state=watermark_state,
            watermark_full_mask=watermark_full_mask,
            simulated_acc_len=simulated_acc_len,
            simulated_acc_len_rng=simulated_acc_len_rng,
            sparse_bonus_argmax=sparse_bonus_argmax,
        )
    info = batch.sampling_info
    if not prepared_penalties(info):
        # The graph does not capture request-owned watermark state.
        if (
            watermark_state is None
            and simulated_acc_len is None
            and simulated_acc_len_rng is None
            and not sparse_bonus_argmax
            and envs.SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH.get()
        ):
            from .sample_graph import sample_graph

            return sample_graph(
                original,
                verify_input,
                batch,
                logits_output,
                grammar_mask,
                uno_target_max_top_k=uno_target_max_top_k,
            )
        return original(
            verify_input,
            batch,
            logits_output,
            grammar_mask,
            uno_target_max_top_k=uno_target_max_top_k,
            watermark_state=watermark_state,
            watermark_full_mask=watermark_full_mask,
            simulated_acc_len=simulated_acc_len,
            simulated_acc_len_rng=simulated_acc_len_rng,
            sparse_bonus_argmax=sparse_bonus_argmax,
        )
    apply_linear_penalties(verify_input, batch, logits_output.next_token_logits)
    additive, scaling = info.acc_additive_penalties, info.acc_scaling_penalties
    info.acc_additive_penalties = info.acc_scaling_penalties = None
    try:
        return original(
            verify_input,
            batch,
            logits_output,
            grammar_mask,
            uno_target_max_top_k=uno_target_max_top_k,
            watermark_state=watermark_state,
            watermark_full_mask=watermark_full_mask,
            simulated_acc_len=simulated_acc_len,
            simulated_acc_len_rng=simulated_acc_len_rng,
            sparse_bonus_argmax=sparse_bonus_argmax,
        )
    finally:
        info.acc_additive_penalties, info.acc_scaling_penalties = additive, scaling
