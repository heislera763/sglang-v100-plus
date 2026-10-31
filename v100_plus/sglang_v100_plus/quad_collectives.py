"""MiniMax TP8 sums across two NVLink quads with three PCIe bridges.

The first quad sum includes every rank, including a slow PCIe endpoint. Three
disjoint stripes cross through the fastest endpoints, then a second NVLink sum
distributes them. Tiny messages keep the existing latency-oriented collective.
"""

import logging

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_buffer, get_parallel

logger = logging.getLogger(__name__)


def _slot():
    return get_buffer("v100_minimax_quad_collectives", dict)


def bridge_plan(devices):
    """Validate two complete cliques and choose three fast endpoints per quad."""
    if len(devices) != 8 or len({d["bus"] for d in devices}) != 8:
        raise ValueError("MiniMax quad collectives require eight distinct GPUs")
    ends = []
    for base in (0, 4):
        ranks = list(range(base, base + 4))
        for rank in ranks:
            peers = {devices[r]["bus"] for r in ranks if r != rank}
            if not peers.issubset(devices[rank]["peers"]):
                raise ValueError(
                    "MiniMax TP8 ranks must form two complete NVLink quads"
                )
        # Higher rank wins ties, preserving deterministic choices on all ranks.
        fast = sorted(ranks, key=lambda r: (devices[r]["pcie_bw"], r))[-3:]
        ends.append(sorted(fast))
    return list(zip(*ends, strict=True))


def stripe_bounds(numel, slot):
    """Three aligned, non-overlapping stripes; the last owns any remainder."""
    step = numel // 3 // 16 * 16
    return slot * step, numel if slot == 2 else (slot + 1) * step


def _local_topology():
    import pynvml as nvml

    nvml.nvmlInit()
    try:
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        handle = nvml.nvmlDeviceGetHandleByUUID(uuid)

        def bus(handle):
            value = nvml.nvmlDeviceGetPciInfo(handle).busId
            return value.decode() if isinstance(value, bytes) else value

        peers = set()
        for link in range(6):
            if nvml.nvmlDeviceGetNvLinkState(handle, link):
                value = nvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link).busId
                peers.add(value.decode() if isinstance(value, bytes) else value)
        generation = nvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        width = nvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        return dict(
            bus=bus(handle),
            peers=peers,
            pcie_bw={1: 250, 2: 500, 3: 985}[generation] * width,
        )
    finally:
        nvml.nvmlShutdown()


class QuadCollectives:
    def __init__(self, group):
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.group = group
        self.native = group.require_pynccl_comm("MiniMax quad transport")
        records = [None] * 8
        dist.all_gather_object(records, _local_topology(), group=group.cpu_group)
        bridges = bridge_plan(records)
        ranks = group.ranks
        local = ranks.index(dist.get_rank())
        layouts = [list(range(4)), list(range(4, 8)), *bridges]
        # Every world rank constructs groups in the same order. Keep CPU groups
        # alive with their native communicators until the runtime is destroyed.
        self.cpu_groups = [
            dist.new_group([ranks[r] for r in layout], backend="gloo")
            for layout in layouts
        ]
        self.quad = PyNcclCommunicator(
            self.cpu_groups[local // 4], group.device, max_ctas=4
        )
        self.pair = None
        self.stripe = None
        for slot, endpoints in enumerate(bridges):
            if local in endpoints:
                self.pair = PyNcclCommunicator(
                    self.cpu_groups[2 + slot], group.device, max_ctas=4
                )
                self.stripe = slot
        # Eager gathers and tiny sums share the graph communicator. Creating a
        # second c10d NCCL transport would duplicate its channel/peer buffers.
        value = torch.zeros(1, device=group.device, dtype=torch.float16)
        gathered = torch.empty(8, device=group.device, dtype=value.dtype)
        with self.native.change_state(enable=True):
            self.native.all_gather(gathered, value)
        dist.barrier(group=group.cpu_group)
        if local == 0:
            logger.info("MiniMax NVLink quads: PCIe bridge TP ranks %s", bridges)

    def all_reduce(self, value):
        # Invocation-owned outputs are essential: callers retain activations and
        # captured graphs must not alias another layer's collective scratch.
        source = value.reshape(-1)
        partial, output = torch.empty_like(source), torch.empty_like(source)
        output.zero_()
        with self.quad.change_state(enable=True):
            self.quad.outplace_all_reduce(source, partial)
        if self.pair is not None:
            start, stop = stripe_bounds(source.numel(), self.stripe)
            output[start:stop].copy_(partial[start:stop])
            with self.pair.change_state(enable=True):
                self.pair.all_reduce(output[start:stop])
        with self.quad.change_state(enable=True):
            self.quad.all_reduce(output)
        return output.view_as(value)


def initialize(runner):
    parallel = get_parallel()
    if (
        parallel.tp_size != 8
        or parallel.pp_size != 1
        or parallel.moe_ep_size != 1
        or runner.is_draft_worker
    ):
        return
    state = _slot()
    if not state:
        from sglang.srt.distributed import get_tp_group

        state["transport"] = QuadCollectives(get_tp_group())


def select(group, value):
    if torch.compiler.is_compiling() or group.world_size != 8 or not value.is_cuda:
        return None
    transport = _slot().get("transport")
    if (
        transport is None
        or transport.group is not group
        or value.dtype != torch.float16
        or not value.is_contiguous()
        or value.numel() * value.element_size() < 64 * 1024
    ):
        return None
    return transport


def all_reduce(original, group, value):
    transport = select(group, value)
    if transport is not None:
        return transport.all_reduce(value)
    transport = _slot().get("transport")
    if transport is not None and transport.group is group and value.is_cuda:
        with transport.native.change_state(enable=True):
            return original(group, value)
    return original(group, value)


def all_gather(original, group, output, value):
    transport = _slot().get("transport")
    if transport is not None and transport.group is group and value.is_cuda:
        with transport.native.change_state(enable=True):
            return original(group, output, value)
    return original(group, output, value)
