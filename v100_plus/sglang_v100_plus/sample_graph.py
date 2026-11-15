"""Capture linear rejection verification without retaining request objects."""

from copy import copy
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from sglang.srt.runtime_context import (
    get_buffer,
    get_disagg,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
)

_PLAN_INPUTS = (
    "draft_token",
    "retrieve_index",
    "retrieve_next_token",
    "retrieve_next_sibling",
    "draft_probs",
)
_SAMPLING_INPUTS = ("temperatures", "top_ps", "top_ks", "min_ps")


def eligible(verify_input, batch, grammar_mask, uno_target_max_top_k):
    info = batch.sampling_info
    if batch.forward_mode.is_idle() or info is None:
        return False
    schedule, spec, parallel = get_schedule(), get_spec(), get_parallel()
    return (
        schedule.disable_overlap_schedule
        and schedule.max_running_requests == 1
        and not get_exec().overlap.enable_two_batch_overlap
        and not get_exec().overlap.enable_single_batch_overlap
        and parallel.pp_async_batch_depth == 0
        and parallel.tp_size == parallel.attn_tp_size
        and get_disagg().disaggregation_mode == "null"
        and spec.speculative_use_rejection_sampling
        and batch.seq_lens.numel() == 1
        and verify_input.tree_topk == 1
        and verify_input.draft_token_num in (2, 3, 4)
        and verify_input.draft_token_num == verify_input.max_tree_depth
        and verify_input.draft_probs is not None
        and grammar_mask is None
        and not batch.has_grammar
        and uno_target_max_top_k is None
        and not info.is_any_greedy
        and not info.need_min_p_sampling
        and info.sampling_seed is None
        and info.acc_additive_penalties is None
        and info.acc_scaling_penalties is None
        and info.logit_bias is None
        and not info.has_custom_logit_processor
        and not batch.return_logprob
        and not any(info.return_sampling_masks or [])
    )


def _clone_fields(source, names):
    return {name: getattr(source, name).clone() for name in names}


def _refresh(destination, source, names):
    for name in names:
        getattr(destination, name).copy_(getattr(source, name))


class SampleGraph:
    def __init__(self, original, verify_input, batch, logits_output, group):
        from sglang.srt.distributed import graph_capture

        self.group = group
        info = copy(batch.sampling_info)
        for name, value in _clone_fields(info, _SAMPLING_INPUTS).items():
            setattr(info, name, value)
        self.batch = SimpleNamespace(
            device=batch.device,
            forward_mode=batch.forward_mode,
            seq_lens=batch.seq_lens.clone(),
            sampling_info=info,
        )
        self.verify_input = SimpleNamespace(
            tree_topk=verify_input.tree_topk,
            draft_token_num=verify_input.draft_token_num,
            max_tree_depth=verify_input.max_tree_depth,
            **_clone_fields(verify_input, _PLAN_INPUTS),
        )
        self.logits = LogitsProcessorOutput(
            next_token_logits=logits_output.next_token_logits.clone()
        )
        device = self.logits.next_token_logits.device
        rng = torch.cuda.get_rng_state(device)
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(self.stream):
                for _ in range(2):
                    original(self.verify_input, self.batch, self.logits, None)
            torch.cuda.current_stream(device).wait_stream(self.stream)
            torch.cuda.synchronize(device)
            self.graph = torch.cuda.CUDAGraph()
            with graph_capture(stream=self.stream):
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.outputs = original(
                        self.verify_input, self.batch, self.logits, None
                    )
        finally:
            # Warmup/capture must not consume the caller's real sampling coins.
            torch.cuda.set_rng_state(rng, device)

    def run(self, verify_input, batch, logits_output):
        _refresh(self.verify_input, verify_input, _PLAN_INPUTS)
        _refresh(self.batch.sampling_info, batch.sampling_info, _SAMPLING_INPUTS)
        self.batch.seq_lens.copy_(batch.seq_lens)
        self.logits.next_token_logits.copy_(logits_output.next_token_logits)
        self.graph.replay()
        # PP can retain a result into a later postprocessing turn. Graph-owned
        # outputs must not overwrite that earlier packet on the next replay.
        return tuple(value.clone() for value in self.outputs)


def sample_graph(
    original,
    verify_input,
    batch,
    logits_output,
    grammar_mask=None,
    uno_target_max_top_k=None,
):
    if not eligible(verify_input, batch, grammar_mask, uno_target_max_top_k):
        return original(
            verify_input,
            batch,
            logits_output,
            grammar_mask,
            uno_target_max_top_k=uno_target_max_top_k,
        )
    logits = logits_output.next_token_logits
    q = verify_input.draft_probs
    if (
        getattr(logits_output, "next_token_logits_vocab_shard", None) is not None
        or not logits.is_cuda
        or torch.cuda.get_device_capability(logits.device) != (7, 0)
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
        or q.device != logits.device
        or q.dtype != torch.float32
        or not q.is_contiguous()
        or q.shape != (1, verify_input.draft_token_num - 1, logits.shape[-1])
    ):
        return original(
            verify_input,
            batch,
            logits_output,
            grammar_mask,
            uno_target_max_top_k=uno_target_max_top_k,
        )
    group = get_parallel().tp_group
    spec = get_spec()
    key = (
        id(group),
        logits.device,
        verify_input.draft_token_num,
        logits.shape[-1],
        batch.sampling_info.need_top_k_sampling,
        batch.sampling_info.need_top_p_sampling,
        spec.speculative_accept_threshold_single,
        spec.speculative_accept_threshold_acc,
        spec.speculative_use_block_verification,
    )
    graphs = get_buffer("v100_spec_sample_graphs", dict)
    if key not in graphs:
        graphs[key] = SampleGraph(original, verify_input, batch, logits_output, group)
    return graphs[key].run(verify_input, batch, logits_output)
