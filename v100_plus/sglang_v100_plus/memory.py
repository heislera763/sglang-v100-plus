"""Explicit full-FP16 Qwen memory profile for one session on two V100 quads.

Select this plugin in place of v100_plus. Projections keep their complete K
reductions. KV readers retain the original logical layout and attention kernels.
CUDA VMM maps shared GPU storage into each consumer's local address space.
"""

import inspect
import math
import os

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.mem_cache.memory_pool import (
    GB,
    KVWriteLoc,
    MHATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.runtime_context import (
    get_buffer,
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils.cuda_vmm_utils import (
    VmmReservation,
    _get_cuda_driver,
    align_up,
    check_drv,
    exchange_posix_fds,
    get_allocation_granularity,
    import_peer_handle,
    make_device_allocation_prop,
    tensor_from_pointer,
)


def allocation_layout(nbytes, granularity, owners):
    """Return owner, offset and length for contiguous, balanced physical stripes."""
    if nbytes <= 0 or granularity <= 0 or not owners:
        raise ValueError("Shared buffers need positive capacity and owners")
    if len(set(owners)) != len(owners):
        raise ValueError("A shared-buffer owner may occur only once")
    pages = max(len(owners), align_up(nbytes, granularity) // granularity)
    offset = 0
    result = []
    for index, owner in enumerate(owners):
        size = (pages // len(owners) + (index < pages % len(owners))) * granularity
        result.append((owner, offset, size))
        offset += size
    return result


class SharedBuffer:
    """Own exported physical pages, local mappings and one non-owning Tensor."""

    def __init__(self, shape, dtype, group, owners):
        self.tensor = None
        self.reservation = None
        self.imported = []
        self.local_bytes = 0
        self.start = self.end = 0
        driver = _get_cuda_driver()
        device = torch.cuda.current_device()
        handle_type = (
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        prop = make_device_allocation_prop(
            device, handle_types=handle_type, gpu_direct_rdma=False
        )
        granularity = get_allocation_granularity(prop)
        nbytes = math.prod(shape) * dtype.itemsize
        layout = allocation_layout(nbytes, granularity, owners)
        rank = group.rank_in_group
        descriptor = next((part for part in layout if part[0] == rank), None)
        local_fds = []
        peer_fds = {}
        try:
            error = None
            try:
                self.reservation = VmmReservation(
                    sum(size for _, _, size in layout),
                    prop,
                    device,
                    alignment=granularity,
                )
                if descriptor is not None:
                    _, offset, self.local_bytes = descriptor
                    handle = self.reservation.map(
                        offset, self.local_bytes, retain_handle=True
                    )
                    local_fds.append(
                        int(
                            check_drv(
                                driver.cuMemExportToShareableHandle(
                                    handle, handle_type, 0
                                ),
                                "export shared KV allocation",
                            )
                        )
                    )
            except Exception as exc:
                error = str(exc)
            _check_allocation(group, error)
            peer_fds = exchange_posix_fds(
                group.cpu_group,
                rank,
                group.world_size,
                local_fds,
                [int(peer in owners) for peer in range(group.world_size)],
            )
            error = None
            try:
                for owner, offset, size in layout:
                    if owner == rank:
                        continue
                    handle = import_peer_handle(
                        None, peer_fds[(owner, 0)], use_fabric=False, peer_rank=owner
                    )
                    self.imported.append(handle)
                    self.reservation.map_existing(offset, size, handle)
                self.tensor = tensor_from_pointer(
                    self.reservation.base,
                    nbytes,
                    shape=shape,
                    dtype=dtype,
                    device_id=device,
                )
                if descriptor is not None:
                    row_bytes = math.prod(shape[1:]) * dtype.itemsize
                    self.start = descriptor[1] // row_bytes
                    self.end = (descriptor[1] + descriptor[2]) // row_bytes
                    self.tensor[self.start : min(shape[0], self.end)].zero_()
                torch.cuda.synchronize()
            except Exception as exc:
                error = str(exc)
            _check_allocation(group, error)
        except BaseException:
            self.close()
            raise
        finally:
            for fd in [*local_fds, *peer_fds.values()]:
                os.close(fd)

    def close(self):
        self.tensor = None
        if self.reservation is not None:
            self.reservation.close()
            self.reservation = None
        driver = _get_cuda_driver()
        while self.imported:
            check_drv(driver.cuMemRelease(self.imported.pop()), "release shared KV")

    def __del__(self):
        try:
            self.close()
        except Exception:
            # CUDA may already be unloaded during interpreter teardown.
            pass


def _check_allocation(group, error):
    errors = group.all_gather_object(error)
    if any(message is not None for message in errors):
        raise RuntimeError(f"Shared GPU allocation failed across ranks {errors}")


def _groups():
    def create():
        from sglang.srt.distributed import init_model_parallel_group

        from .prefill import quad_partition

        parallel = get_parallel()
        parent, world = parallel.attn_tp_group, parallel.world_group
        quad = quad_partition(1).group if parallel.tp_size == 8 else parent
        for peer in quad.ranks:
            if peer != world.rank_in_group and not torch.cuda.can_device_access_peer(
                torch.cuda.current_device(), peer
            ):
                raise ValueError(
                    "Memory sharing requires consecutive peer-access quads"
                )
        consumers = quad
        if parallel.tp_size == 4:
            consumers = init_model_parallel_group(
                [world.ranks[start : start + 2] for start in range(0, 8, 2)],
                local_rank=world.local_rank,
                backend=dist.get_backend(parent.device_group),
                use_pynccl=True,
                use_custom_allreduce=False,
                use_mscclpp=False,
                use_torch_symm_mem_allreduce=False,
                group_name="v100_memory_kv",
            )
        # Establish broadcast transports before sizing KV. Point-to-point
        # sends would allocate additional peer connections during prefill.
        value = torch.zeros(1, device=torch.cuda.current_device())
        for group in [parent] if parent is quad else [parent, quad]:
            with group.pynccl_comm.change_state(enable=True):
                for source in range(group.world_size):
                    group.broadcast(value, src=source)
        torch.cuda.synchronize()
        return quad, consumers

    return get_buffer("v100_memory_groups", create)


def _discard_replica(module):
    original = module.weight
    shape = original.shape

    def validate_shape(_parameter, loaded_weight):
        if loaded_weight.shape != shape:
            raise ValueError(
                f"Shared projection expected {shape}, got {loaded_weight.shape}"
            )

    module.weight = nn.Parameter(
        torch.empty(0, device=original.device, dtype=original.dtype),
        requires_grad=False,
    )
    module.weight.weight_loader = validate_shape


def _initialize_hc(original, self, config, *args, **kwargs):
    result = original(self, config, *args, **kwargs)
    group = get_parallel().attn_tp_group
    _groups()
    counter = get_buffer("v100_memory_hc_counter", lambda: [0])
    self._memory_owner = counter[0] % group.world_size
    self._memory_group = group
    counter[0] += 1
    if group.rank_in_group != self._memory_owner:
        for name in ("input_mix_weight_down", "input_mix_weight_up"):
            if hasattr(self, name):
                _discard_replica(getattr(self, name))
    return result


def _mix_hc(original, self, hyper_input, *, normalized_input=None):
    from .kernels.sm70_hc_mix import gate_supported
    from .prefill import _slot, local_input

    if normalized_input is not None:
        raise ValueError("The shared HC profile requires its ordinary normalization")
    group = self._memory_group
    partition = _slot().get()
    if partition is not None:
        hyper_input = local_input(hyper_input)
        normed = (
            self.hc_norm(hyper_input)
            if self.config.hc_per_branch_norm
            else self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        )
        output = hyper_input.new_empty((partition.rows, self.hidden_size))
        quad, _ = _groups()
        owner_quad = self._memory_owner // quad.world_size
        if group.rank_in_group // quad.world_size == owner_quad:
            # Both quads carry the same TP residual. One quad supplies its
            # normalized token shards to the full-weight owner one at a time.
            owner = self._memory_owner % quad.world_size
            comm = quad.pynccl_comm
            scratch = torch.empty_like(normed)
            with comm.change_state(enable=True):
                for peer in range(quad.world_size):
                    source = normed if peer == quad.rank_in_group else scratch
                    quad.broadcast(source, src=peer)
                    if quad.rank_in_group == owner:
                        first = peer * partition.local_rows
                        count = min(partition.local_rows, partition.rows - first)
                        if count > 0:
                            projected = self._mix_compute(
                                source,
                                self.input_mix_weight_down.weight,
                                self.input_mix_weight_up.weight,
                                self.hc_count,
                                self.hidden_size,
                            ).to(self.params_dtype)
                            output[first : first + count].copy_(projected[:count])
        with group.pynccl_comm.change_state(enable=True):
            group.broadcast(output, src=self._memory_owner)
        return output, (hyper_input, normed, None)
    if group.rank_in_group == self._memory_owner:
        output, residual = original(
            self, hyper_input, normalized_input=normalized_input
        )
    else:
        normed = (
            self.hc_norm(hyper_input)
            if self.config.hc_per_branch_norm
            else self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        )
        output = hyper_input.new_empty((hyper_input.shape[0], self.hidden_size))
        residual = (hyper_input, normed, None)
    with group.pynccl_comm.change_state(enable=True):
        group.broadcast(output, src=self._memory_owner)
        if getattr(self, "_split_combine_ok", False) and gate_supported(
            residual[1], self.block_inject_weight.weight
        ):
            gate = (
                residual[2]
                if group.rank_in_group == self._memory_owner
                else torch.empty(
                    (hyper_input.shape[0], 8, 4),
                    device=hyper_input.device,
                    dtype=torch.float32,
                )
            )
            group.broadcast(gate, src=self._memory_owner)
            residual = (residual[0], residual[1], gate)
    return output, residual


def _initialize_linear(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    prefix = (
        inspect.signature(original)
        .bind(self, *args, **kwargs)
        .arguments.get("prefix", "")
    )
    if not (
        ".layers." in prefix
        and prefix.endswith((".gate", ".index_qk_proj", ".key_proj", ".value_proj"))
        and self.weight.dtype == torch.float16
        and self.weight.numel() >= 512 * 2560
        and self.bias is None
        and self.quant_config is None
    ):
        return result
    quad, _ = _groups()
    loads = get_buffer("v100_memory_linear_loads", lambda: [0] * quad.world_size)
    self._memory_owner = min(range(quad.world_size), key=loads.__getitem__)
    self._memory_group = quad
    loads[self._memory_owner] += self.weight.nbytes
    if quad.rank_in_group != self._memory_owner:
        _discard_replica(self)
    return result


def _linear(original, self, inputs):
    if not hasattr(self, "_memory_group"):
        return original(self, inputs)
    group = self._memory_group
    if group.rank_in_group == self._memory_owner:
        output, bias = original(self, inputs)
    else:
        output = inputs.new_empty((*inputs.shape[:-1], self.output_size))
        bias = None
    with group.pynccl_comm.change_state(enable=True):
        group.broadcast(output, src=self._memory_owner)
    return output, bias


class SharedKVPool(MHATokenToKVPool):
    def _create_buffers_normal(self):
        _, self._memory_group = _groups()
        group = self._memory_group
        if (
            self.dtype != torch.float16
            or self.kv_cache_layout != "nhd"
            or self.head_num != 1
            or self.head_dim != 256
            or self.v_head_dim != 256
        ):
            raise ValueError("Shared QSA needs plain FP16 NHD storage with one KV head")
        shape = (self.size + self.page_size, 1, self.head_dim)
        self._memory_allocations = []
        self._memory_fence = torch.zeros(1, device=self.device)
        self.k_buffer, self.v_buffer = [], []
        self._memory_striped = self.layer_num == 1
        try:
            for index in range(self.layer_num):
                owners = (
                    list(range(group.world_size))
                    if self._memory_striped
                    else [index % group.world_size]
                )
                for buffers in (self.k_buffer, self.v_buffer):
                    allocation = SharedBuffer(shape, self.dtype, group, owners)
                    self._memory_allocations.append(allocation)
                    buffers.append(allocation.tensor)
        except BaseException:
            self.k_buffer, self.v_buffer = [], []
            for allocation in self._memory_allocations:
                allocation.close()
            raise

    def _fence(self):
        group = self._memory_group
        with group.pynccl_comm.change_state(enable=True):
            group.pynccl_comm.all_reduce(self._memory_fence)

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        dcp_kv_mask=None,
    ):
        index = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        ) - self.start_layer
        self._fence()
        if self._memory_striped:
            loc, _, _ = unwrap_write_loc(loc_info)
            allocation = self._memory_allocations[0]
            owned = (loc >= allocation.start) & (loc < allocation.end)
            dcp_kv_mask = owned if dcp_kv_mask is None else owned & dcp_kv_mask
        if cache_k.shape[0] and (
            self._memory_striped
            or index % self._memory_group.world_size == self._memory_group.rank_in_group
        ):
            super().set_kv_buffer(
                layer,
                loc_info,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_override,
                dcp_kv_mask=dcp_kv_mask,
            )
        self._fence()

    def set_kv_buffer_prefix_valid(
        self,
        layer,
        loc_2d,
        commit_lens,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
    ):
        if (
            loc_2d.ndim != 2
            or commit_lens.shape != (loc_2d.shape[0],)
            or cache_k.shape[0] != loc_2d.numel()
            or cache_v.shape[0] != loc_2d.numel()
        ):
            raise ValueError("Prefix commits require one KV row per fixed slot")
        loc_2d = loc_2d.to(device=self.device, dtype=torch.int64)
        commit_lens = commit_lens.to(device=self.device, dtype=torch.int32)
        columns = torch.arange(loc_2d.shape[1], device=self.device)
        valid = (columns[None, :] < commit_lens[:, None]).reshape(-1)
        self.set_kv_buffer(
            layer,
            KVWriteLoc(loc_2d.reshape(-1), physical=True),
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override,
            dcp_kv_mask=valid,
        )

    def _init_kv_copy_and_warmup(self):
        self._kv_copy_config = None

    def move_kv_cache(self, tgt_loc, src_loc):
        from sglang.srt.mem_cache.memory_pool import masked_set_kv_buffer_kernel

        self._fence()
        for index, (key, value) in enumerate(zip(self.k_buffer, self.v_buffer)):
            if self._memory_striped:
                allocation = self._memory_allocations[0]
                mask = (tgt_loc >= allocation.start) & (tgt_loc < allocation.end)
                source_k, source_v = key[src_loc.long()], value[src_loc.long()]
                if tgt_loc.numel():
                    masked_set_kv_buffer_kernel[(tgt_loc.numel(),)](
                        source_k,
                        source_v,
                        key,
                        value,
                        tgt_loc,
                        mask,
                        tgt_loc.numel(),
                        1,
                        self.head_dim,
                        128,
                        source_k.stride(0),
                        source_k.stride(1),
                        source_v.stride(0),
                        source_v.stride(1),
                        key.stride(0),
                        value.stride(0),
                    )
            elif (
                index % self._memory_group.world_size
                == self._memory_group.rank_in_group
            ):
                key[tgt_loc.long()] = key[src_loc.long()]
                value[tgt_loc.long()] = value[src_loc.long()]
        self._fence()

    def get_kv_size_bytes(self):
        return (
            sum(item.local_bytes for item in self._memory_allocations[::2]),
            sum(item.local_bytes for item in self._memory_allocations[1::2]),
        )

    def _clear_buffers(self):
        super()._clear_buffers()
        for allocation in self._memory_allocations:
            allocation.close()
        self._memory_allocations.clear()


def _initialize_qsa_pool(original, self, *args, **kwargs):
    kwargs["full_kv_pool_class"] = SharedKVPool
    original(self, *args, **kwargs)
    if self.qsa_compressed_dtype != torch.float16:
        raise ValueError("Shared QSA requires --qsa-indexer-dtype float16")
    quad, _ = _groups()
    self._memory_index_group = quad
    self._memory_index_fence = torch.zeros(1, device=self.device)
    self.qsa_compressed_k_buffer_pool = []
    self.qsa_compressed_flat = None
    torch.cuda.empty_cache()
    shape = (
        self.qsa_compressed_capacity,
        self.qsa_index_kv_heads,
        self.qsa_index_head_dim,
    )
    self._memory_index_allocations = []
    try:
        for index in range(self.full_layer_nums):
            allocation = SharedBuffer(
                shape, self.qsa_compressed_dtype, quad, [index % quad.world_size]
            )
            self._memory_index_allocations.append(allocation)
            self.qsa_compressed_k_buffer_pool.append(allocation.tensor)
    except BaseException:
        self.qsa_compressed_k_buffer_pool = []
        for allocation in self._memory_index_allocations:
            allocation.close()
        raise
    self.mem_usage = sum(self.get_kv_size_bytes()) / GB


def _qsa_size(original, self):
    if not hasattr(self, "_memory_index_allocations"):
        return original(self)
    key, value = self.full_kv_pool.get_kv_size_bytes()
    return key + sum(
        allocation.local_bytes for allocation in self._memory_index_allocations
    ) + sum(
        buffer.nbytes for buffer in self.qsa_key_state_buffer_pool
    ) + self.qsa_rope_position_buffer.nbytes, value


def _compress(original, self, *args, **kwargs):
    metadata = args[3] if len(args) > 3 else kwargs["metadata"]
    pool = metadata.token_to_kv_pool
    group = pool._memory_index_group
    index = pool._transfer_full_attention_id(self.layer_id)
    with group.pynccl_comm.change_state(enable=True):
        group.pynccl_comm.all_reduce(pool._memory_index_fence)
        if index % group.world_size == group.rank_in_group:
            original(self, *args, **kwargs)
        group.pynccl_comm.all_reduce(pool._memory_index_fence)


def _initialize_embedding(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if type(self).__name__ == "VocabParallelEmbedding" and self.embedding_dim == 2560:
        previous = self.weight
        self.weight = nn.Parameter(
            torch.empty(
                previous.shape, dtype=previous.dtype, device="cpu", pin_memory=True
            ),
            requires_grad=False,
        )
        self.weight.__dict__.update(previous.__dict__)
    return result


def _embed(original, self, inputs):
    if self.weight.device.type != "cpu":
        return original(self, inputs)
    from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
        vocab_parallel_embedding,
    )

    return vocab_parallel_embedding(
        inputs.contiguous(),
        self.weight,
        self.shard_indices.org_vocab_start_index,
        self.shard_indices.org_vocab_end_index,
        self.shard_indices.num_org_vocab_padding,
        self.shard_indices.added_vocab_start_index,
        self.shard_indices.added_vocab_end_index,
    )


def _share_mtp_weights(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if type(self.draft_runner.model).__name__ == "Qwen4ExpForCausalLMMTP":
        self.init_token_map()
        self.init_lm_head()
        torch.cuda.empty_cache()
    return result


def _load_model(original, self, *args, **kwargs):
    config = getattr(
        self.model_config.hf_config, "text_config", self.model_config.hf_config
    )
    parallel = get_parallel()
    if not (
        self.model_config.dtype == torch.float16
        and torch.cuda.get_device_capability() == (7, 0)
        and getattr(config, "model_type", None) == "qwen4_exp_text"
        and getattr(config, "quantization_config", None) is None
        and getattr(self.model_config.hf_config, "quantization_config", None) is None
        and config.hidden_size == 2560
        and config.hc_count == 4
        and config.hc_lowrank == 320
        and config.num_key_value_heads == 2
        and config.head_dim == 256
        and config.indexer_kv_heads == 1
        and config.indexer_head_dim == 128
        and config.indexer_compress_ratio == 4
        and parallel.launch_world_size == 8
        and (
            (parallel.tp_size, parallel.pp_size) in ((8, 1), (4, 2))
            or (
                self.is_draft_worker and parallel.tp_size == 4 and parallel.pp_size == 1
            )
        )
        and parallel.attn_tp_size == parallel.tp_size
        and parallel.moe_ep_size == parallel.tp_size
        and parallel.attn_dcp_size == 1
        and get_schedule().max_running_requests == 1
        and get_schedule().disable_overlap_schedule
        and get_exec().graph.disable_prefill_cuda_graph
        and not get_exec().features.enable_memory_saver
        and not get_memory().enable_unified_memory
        and not get_memory().enable_hierarchical_cache
        and get_disagg().disaggregation_mode == "null"
    ):
        raise ValueError(
            "v100_memory requires single-session full-FP16 Qwen4-Exp on TP8 or PP2xTP4"
        )
    result = original(self, *args, **kwargs)
    for module in self.model.modules():
        cache = getattr(module, "cos_sin_cache", None)
        if (
            not isinstance(cache, torch.Tensor)
            or cache.device.type != "cuda"
            or hasattr(module, "_memory_cos_sin_cpu")
        ):
            continue
        host = cache.cpu().pin_memory()
        module._memory_cos_sin_cpu = host
        module.cos_sin_cache = tensor_from_pointer(
            host.data_ptr(),
            host.nbytes,
            shape=host.shape,
            dtype=host.dtype,
            device_id=torch.cuda.current_device(),
        )
    return result


def _initialize_budget(original, self, configurator):
    original(self, configurator)
    from sglang.srt.configs.hybrid_arch import mambaish_config

    config = mambaish_config(configurator.model_config)
    layers = (
        1
        if configurator.is_draft_worker
        else sum(
            configurator.layer_info.start_layer
            <= layer
            < configurator.layer_info.end_layer
            for layer in config.full_attention_layer_ids
        )
    )
    quad, consumers = _groups()
    self._cell_size = (
        math.ceil(layers / consumers.world_size) * 1024
        + math.ceil(layers / quad.world_size) * 64
    )
    parallel = get_parallel()
    hosts_draft = parallel.pp_size == 1 or parallel.pp_group.is_last_rank
    if (
        not configurator.is_draft_worker
        and configurator.spec_algorithm.is_eagle()
        and hosts_draft
    ):
        self._cell_size += 1024 // consumers.world_size + 64
    # Each separately-backed K/V/index allocation can round up by one VMM page.
    # This bounds main and native single-layer draft padding independently of counts.
    self._fixed_overhead_bytes += (
        2 * math.ceil(layers / consumers.world_size)
        + math.ceil(layers / quad.world_size)
        + 3
    ) * (2 << 20)


def register():
    from sglang.srt.environ import envs

    selected = (envs.SGLANG_PLUGINS.get() or "").split(",")
    if "v100_memory" not in (name.strip() for name in selected):
        return
    from . import register as register_base
    from . import runtime

    register_base()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    hooks = (
        ("sglang.srt.model_executor.model_runner.ModelRunner.load_model", _load_model),
        ("sglang.srt.layers.hyperconnection.GatedResidual.__init__", _initialize_hc),
        ("sglang.srt.layers.hyperconnection.GatedResidual.mix", _mix_hc),
        ("sglang.srt.layers.linear.ReplicatedLinear.__init__", _initialize_linear),
        ("sglang.srt.layers.linear.ReplicatedLinear.forward", _linear),
        (
            "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__",
            _initialize_qsa_pool,
        ),
        (
            "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool.get_kv_size_bytes",
            _qsa_size,
        ),
        (
            "sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer.update_key_state_and_compress",
            _compress,
        ),
        (
            "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__",
            _initialize_embedding,
        ),
        (
            "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding._embed_local_shard",
            _embed,
        ),
        (
            "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.__init__",
            _share_mtp_weights,
        ),
        (
            "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator.__init__",
            _initialize_budget,
        ),
        (
            "sglang.srt.layers.attention.qsa.qsa_indexer._qsa_prefill_row_chunk_size",
            _indexer_rows,
        ),
    )
    for target, replacement in hooks:
        if not any(
            hook[1] is replacement for hook in HookRegistry._hooks.get(target, ())
        ):
            HookRegistry.register(target, replacement, HookType.AROUND)
    runtime.REQUIRED_HOOKS |= frozenset(target for target, _ in hooks)


def _indexer_rows(original, rows, keys, heads):
    native = original(rows, keys, heads)
    if keys <= 0:
        return native
    block_rows = max(1, 128 // heads)
    bounded = max(block_rows, (32 << 20) // (keys * 4) // block_rows * block_rows)
    return min(native, bounded)
