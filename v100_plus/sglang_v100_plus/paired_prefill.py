"""GLM eager prefill: overlap unchanged TP sums with the next token chunk.

A chunk publishes its same-layer KDA state and DSA keys before yielding to the
next chunk. Both run on one compute stream; global sums use a side stream.
Metadata, four-stream mHC coefficients, forward flags and activations belong to
the chunk. Decode/verification and attention/expert ownership are unchanged.
"""

import copy
import dataclasses
from contextvars import ContextVar, copy_context

import torch

from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch
from sglang.srt.runtime_context import (
    get_buffer,
    get_exec,
    get_forward,
    get_parallel,
    get_schedule,
    get_spec,
    mamba_cache_chunk_size,
)

from .quad_collectives import all_reduce as quad_all_reduce


def _slot():
    return get_buffer(
        "v100_glm_paired_prefill",
        lambda: ContextVar("v100_glm_paired_prefill", default=None),
    )


def is_active():
    """Whether this eager forward belongs to the two-chunk compute schedule."""
    return not torch.compiler.is_compiling() and _slot().get() is not None


def _states_of(model):
    from sglang.srt.layers.layer_boundary.residual.mhc import MHCState

    states = {}
    seen = set()

    def visit(x):
        if id(x) in seen:
            return
        seen.add(id(x))
        if isinstance(x, MHCState):
            states[id(x)] = x
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif dataclasses.is_dataclass(x):
            for f in dataclasses.fields(x):
                visit(getattr(x, f.name))
        elif hasattr(x, "__struct_fields__"):
            for key in x.__struct_fields__:
                visit(getattr(x, key))
        elif type(x).__module__.startswith("sglang.srt.layers.layer_boundary"):
            visit(vars(x))

    # Keep traversal containers alive: a shared id-based seen set must not
    # mistake recycled temporary-list IDs for already visited layers.
    visit([[layer.attn_boundary, layer.ffn_boundary] for layer in model.layers])
    return list(states.values())


def _metadata_of(backend):
    refs = []
    seen = set()

    def visit(x):
        if id(x) in seen:
            return
        seen.add(id(x))
        for key in ("forward_metadata", "sm70_request_ids"):
            if hasattr(x, key):
                refs.append((x, key, getattr(x, key)))
        for key in (
            "full_attn_backend",
            "linear_attn_backend",
            "prefill_backend",
            "decode_backend",
        ):
            child = getattr(x, key, None)
            if child is not None:
                visit(child)

    visit(backend)
    return refs


def _owned(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_owned(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_owned(v) for v in value)
    if isinstance(value, dict):
        return {k: _owned(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                f.name: _owned(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if f.init
            },
        )
    return value


def _split_batch(batch):
    assert batch.batch_size == 1 and not batch.can_run_tbo
    count = batch.input_ids.numel()
    # Keep the existing 2048-token FP16 recurrent-state persistence boundary.
    # The admission window contains at most 4096 tokens.
    assert 2048 < count <= 4096
    cut = 2048
    prefix = batch.extend_prefix_lens_cpu[0]
    children = []
    for start, stop in ((0, cut), (cut, count)):
        child = copy.copy(batch)
        for key in (
            "input_ids",
            "positions",
            "out_cache_loc",
            "origin_out_cache_loc",
            "out_cache_loc_virtual",
            "input_embeds",
            "mm_input_embeds",
        ):
            value = getattr(batch, key, None)
            if value is not None:
                setattr(child, key, value[start:stop])
        if batch.mrope_positions is not None:
            child.mrope_positions = batch.mrope_positions[:, start:stop]
        child.residual_stream = None
        child.extend_num_tokens = stop - start
        child.extend_seq_lens_cpu = [stop - start]
        child.extend_prefix_lens_cpu = [prefix + start]
        child.seq_lens_cpu = torch.tensor(
            [prefix + stop], dtype=batch.seq_lens_cpu.dtype
        )
        child.seq_lens_sum = prefix + stop
        child.seq_lens = batch.seq_lens.new_tensor([prefix + stop])
        child.extend_seq_lens = batch.extend_seq_lens.new_tensor([stop - start])
        child.extend_prefix_lens = batch.extend_prefix_lens.new_tensor([prefix + start])
        child.extend_start_loc = batch.extend_start_loc.new_zeros(1)
        child.num_token_non_padded = batch.input_ids.new_tensor(
            stop - start, dtype=torch.int32
        )
        child.global_num_token_non_padded_cpu = stop - start
        child.global_num_token_non_padded = None
        child.tbo_parent_token_range = None
        if batch.iteration_ids is not None:
            child.iteration_cols = batch.iteration_ids.cols_slice(
                batch.iteration_cols, slice(start, stop)
            )
        if batch.mamba_track_seqlens_cpu is not None:
            target = batch.mamba_track_seqlens_cpu[0]
            cache_chunk = mamba_cache_chunk_size()
            aligned_target = prefix + (target - prefix) // cache_chunk * cache_chunk
            track = (
                batch.mamba_prefill_track_mask_cpu[0]
                and prefix + start < aligned_target <= prefix + stop
            )
            child.mamba_prefill_track_mask_cpu = [track]
            child.mamba_track_mask = batch.mamba_track_mask.new_tensor([track])
            if track and aligned_target == prefix + stop:
                child.mamba_track_seqlens_cpu = [aligned_target]
                child.mamba_track_seqlens = batch.mamba_track_seqlens.new_tensor(
                    [aligned_target]
                )
        children.append(child)
    return children


def _merge(a, b):
    if isinstance(a, torch.Tensor):
        return torch.cat((a, b), dim=0)
    if isinstance(a, tuple):
        return tuple(_merge(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, list):
        return [_merge(x, y) for x, y in zip(a, b, strict=True)]
    raise TypeError(type(a))


class _Scheduler:
    def __init__(self, original, model, batch, embeds):
        from greenlet import getcurrent, greenlet

        self.original = original
        self.parent = getcurrent()
        self.states = _states_of(model)
        assert len(self.states) == len(model.layers), (
            "Every GLM layer must own an mHC state"
        )
        self.comm = torch.cuda.Stream()
        self.original_metadata = _metadata_of(get_attn_backend())
        self.children = _split_batch(batch)
        self.tasks = []
        self.metadata = []
        self.mhc = [[(None, None) for _ in self.states] for _ in self.children]
        self.outputs = [None, None]
        self.original_flags = {
            key: getattr(get_forward(), key) for key in get_forward()._GRAPH_VISIBLE
        }
        self.flags = [dict(self.original_flags) for _ in self.children]
        offset = 0
        for child in self.children:
            if embeds is not None:
                child.input_embeds = embeds[offset : offset + child.extend_num_tokens]
            offset += child.extend_num_tokens
            get_attn_backend().init_forward_metadata(child)
            self.metadata.append(
                [
                    (obj, key, _owned(value))
                    for obj, key, value in _metadata_of(get_attn_backend())
                ]
            )
        for index, child in enumerate(self.children):

            def run(index=index, child=child):
                with set_forward_batch(child):
                    self.outputs[index] = self.original(
                        model,
                        child.input_ids,
                        child.positions,
                        child,
                        input_embeds=child.input_embeds if embeds is not None else None,
                    )

            task = greenlet(run, parent=self.parent)
            self.tasks.append(task)

    def activate(self, index):
        self.index = index
        for key, value in self.flags[index].items():
            get_forward().set(key, value)
        for state, values in zip(self.states, self.mhc[index]):
            state.h_res, state.h_post = values
        for obj, key, value in self.metadata[index]:
            setattr(obj, key, value)

    def save(self):
        self.mhc[self.index] = [(x.h_res, x.h_post) for x in self.states]
        self.flags[self.index] = {
            key: getattr(get_forward(), key) for key in self.original_flags
        }

    def run(self):
        token = _slot().set(self)
        try:
            for task in self.tasks:
                task.gr_context = copy_context()
            while not all(task.dead for task in self.tasks):
                for index, task in enumerate(self.tasks):
                    if task.dead:
                        continue
                    self.activate(index)
                    task.switch()
            torch.cuda.current_stream().wait_stream(self.comm)
            return _merge(*self.outputs)
        finally:
            _slot().reset(token)
            for key, value in self.original_flags.items():
                get_forward().set(key, value)
            for obj, key, value in self.original_metadata:
                setattr(obj, key, value)
            # A task's Context holds this scheduler. Drop task references so
            # completed requests release their activation storage immediately.
            self.tasks.clear()
            self.outputs.clear()


def all_reduce(original, group, x):
    # The paired scheduler is eager-only. Keep compiled models on the native
    # collective without asking Dynamo to trace a ContextVar/resource lookup.
    if torch.compiler.is_compiling():
        return original(group, x)
    scheduler = _slot().get()
    if scheduler is None or group.world_size != 8 or not x.is_cuda:
        return quad_all_reduce(original, group, x)
    compute = torch.cuda.current_stream()
    ready = torch.cuda.Event()
    ready.record()
    with torch.cuda.stream(scheduler.comm):
        scheduler.comm.wait_event(ready)
        result = quad_all_reduce(original, group, x)
        x.record_stream(scheduler.comm)
        done = torch.cuda.Event()
        done.record()
    scheduler.save()
    scheduler.parent.switch()
    compute.wait_event(done)
    result.record_stream(compute)
    return result


def model_forward(
    original,
    self,
    input_ids,
    positions,
    forward_batch,
    input_embeds=None,
    pp_proxy_tensors=None,
):
    if torch.compiler.is_compiling():
        return original(
            self, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
    parallel, schedule = get_parallel(), get_schedule()
    eligible = (
        forward_batch.forward_mode.is_extend_without_speculative()
        and 2048 < forward_batch.input_ids.numel() <= 4096
        and forward_batch.batch_size == 1
        and parallel.tp_size == parallel.attn_tp_size == 8
        and parallel.pp_size == parallel.moe_ep_size == 1
        and parallel.attn_dcp_size == 4
        and schedule.chunked_prefill_size == 4096
        and schedule.disable_overlap_schedule
        and schedule.max_running_requests == 1
        and get_exec().graph.disable_prefill_cuda_graph
        and get_spec().speculative_algorithm in (None, "EAGLE")
        and self.config.mhc
        and self.config.hc_mult == 4
        and self.config.hidden_size == 4096
    )
    if not eligible:
        return original(
            self, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
    return _Scheduler(original, self, forward_batch, input_embeds).run()
