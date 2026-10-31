"""MiniMax M3.1 integration with packed SM70 W4A16/W8A16 execution."""

import logging
from copy import copy

import torch

logger = logging.getLogger(__name__)


def draft_graph_memory(original, available_mem_gb, *, draft_model, num_tokens_per_req):
    from sglang.srt.runtime_context import get_schedule

    if (
        get_schedule().max_running_requests == 1
        and 1 <= num_tokens_per_req <= 8
        and type(draft_model).__name__
        in ("DSparkMiniMaxDraftModel", "MiniMaxDFlash2DraftModel", "DFlash2DraftModel")
        and getattr(draft_model.config, "hidden_size", None) == 6144
    ):
        # DSpark's complete folded graph measured 20MiB/rank. These bounded
        # single-request drafts may attempt capture below the generic 1GiB
        # heuristic; a genuine CUDA allocation failure is never suppressed.
        return True
    return original(
        available_mem_gb, draft_model=draft_model, num_tokens_per_req=num_tokens_per_req
    )


def sparse_backend_init(original, self, runner):
    from .quad_collectives import initialize

    result = original(self, runner)
    initialize(runner)
    logger.info(
        "MiniMax SM70 uses FP16 Tensor Core index/sparse attention, FP32 "
        "reductions and FP16 main/index KV; Triton supplies metadata/top-k."
    )
    return result


def attention_init(original, self, config, *args, **kwargs):
    from sglang.srt.runtime_context import get_model, get_spec
    from sglang.srt.utils.hf_transformers_utils import get_rope_config

    limit = get_model().context_length
    if limit is not None and limit < config.max_position_embeddings:
        # Fixed-frequency MiniMax RoPE is independent of cache capacity. Keep
        # the chosen frequencies, but do not allocate a 1M-position table for
        # a much smaller serving session (256 MiB/rank at rotary_dim64).
        _, rope = get_rope_config(config)
        if rope and rope.get("rope_type", rope.get("type")) != "default":
            raise ValueError(
                "SM70 MiniMax RoPE capacity trimming requires fixed frequencies"
            )
        config = copy(config)
        # A final static verification block can evaluate lookahead positions
        # beyond the requested output limit. Keep their fixed-frequency rows.
        lookahead = get_spec().speculative_num_draft_tokens or 0
        config.max_position_embeddings = min(
            config.max_position_embeddings, limit + lookahead
        )
    return original(self, config, *args, **kwargs)


def draft_attention_init(original, self, config, *args, **kwargs):
    if getattr(config, "hidden_size", None) == 6144 and set(
        getattr(config, "architectures", ()) or ()
    ) & {"MiniMaxDFlash2DraftModel", "DFlash2DraftModel"}:
        return attention_init(original, self, config, *args, **kwargs)
    return original(self, config, *args, **kwargs)


def _select_blocks(
    q,
    k,
    req_to_token,
    slot_ids,
    prefix,
    lengths,
    max_length,
    topk,
    init_blocks,
    local_blocks,
):
    from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
        _topk_index_kernel,
    )
    from sglang.kernels.ops.attention.minimax_sparse.sm70 import (
        minimax_index_scores_sm70,
    )

    if slot_ids.numel() != 1:
        raise ValueError("SM70 MiniMax currently requires max-running-requests=1")
    # SGLang reserves padding request rows. Select its device-side request ID;
    # never read counts or IDs on the CPU, including during graph capture.
    mapping = req_to_token.index_select(0, slot_ids.long()).reshape(-1)
    rows = q.shape[0]
    if torch.cuda.is_current_stream_capturing():
        # Dummy capture histories are short. Every replay must score the live
        # history, including blocks that did not exist during graph capture.
        max_length = mapping.numel()
    scores = minimax_index_scores_sm70(
        q, k, mapping, prefix, lengths, (max_length + 127) // 128
    )
    selected = torch.full(
        (q.shape[1], rows, topk), -1, device=q.device, dtype=torch.int32
    )
    offsets = torch.arange(0, rows + 1, rows, device=q.device, dtype=torch.int32)
    # Preserve upstream's block-selection/tie ordering. Pin its elementwise
    # selector instead of autotuning Hopper-sized configurations on SM70.
    _topk_index_kernel.fn.fn[(rows, 1, q.shape[1])](
        scores,
        selected,
        1,
        128,
        offsets,
        offsets,
        prefix,
        topk,
        init_blocks,
        local_blocks,
        *scores.stride(),
        *selected.stride(),
        BLOCK_SIZE_K=128,
        BLOCK_SIZE_T=topk,
        MASK_INIT=False,
        MASK_LOCAL=False,
        num_warps=4,
        num_stages=1,
    )
    return mapping, selected


def sparse_prefill(
    original,
    q,
    k_cache,
    v_cache,
    sink,
    idx_q,
    idx_k_cache,
    idx_v_cache,
    idx_sink,
    req_to_token,
    slot_ids,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    max_seqlen_q,
    max_seqlen_k,
    block_size_q,
    block_size_k,
    topk,
    init_blocks,
    local_blocks,
    **kwargs,
):
    from sglang.kernels.ops.attention.minimax_sparse.sm70 import (
        minimax_sparse_attention_sm70,
    )

    _validate_sparse(
        q, k_cache, sink, idx_q, idx_sink, block_size_q, block_size_k, topk, kwargs
    )
    prefix_lens = prefix_lens.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    mapping, selected = _select_blocks(
        idx_q,
        idx_k_cache,
        req_to_token,
        slot_ids,
        prefix_lens,
        seq_lens,
        max_seqlen_k,
        topk,
        init_blocks,
        local_blocks,
    )
    output = minimax_sparse_attention_sm70(
        q, k_cache, v_cache, mapping, prefix_lens, seq_lens, selected
    )
    if kwargs.get("return_topk_idx", False):
        return None, output, selected
    return None, output


def sparse_decode(
    original,
    q,
    sink,
    k_cache,
    v_cache,
    idx_q,
    idx_sink,
    idx_k_cache,
    idx_v_cache,
    req_to_token,
    slot_ids,
    seq_lens,
    max_seqlen_k,
    block_size_q,
    block_size_k,
    topk,
    init_blocks,
    local_blocks,
    **kwargs,
):
    from sglang.kernels.ops.attention.minimax_sparse.sm70 import (
        minimax_sparse_attention_sm70,
    )

    _validate_sparse(
        q, k_cache, sink, idx_q, idx_sink, block_size_q, block_size_k, topk, kwargs
    )
    if q.shape[0] != 1:
        raise ValueError("SM70 MiniMax decode requires one request")
    seq_lens = seq_lens.to(torch.int32)
    prefix = seq_lens - 1
    mapping, selected = _select_blocks(
        idx_q,
        idx_k_cache,
        req_to_token,
        slot_ids,
        prefix,
        seq_lens,
        max_seqlen_k,
        topk,
        init_blocks,
        local_blocks,
    )
    return None, minimax_sparse_attention_sm70(
        q, k_cache, v_cache, mapping, prefix, seq_lens, selected
    )


def sparse_extend(original, self, q, k, v, layer, forward_batch, *args, **kwargs):
    if not forward_batch.forward_mode.is_target_verify():
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)
    # DFlash/DSpark seq_lens is the accepted prefix, unlike the ordinary
    # prefill batch's prefix+chunk. Build a causal linear verify window without
    # mutating the shared forward batch or relying on absent extend metadata.
    rows = q.shape[0]
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if (
        forward_batch.seq_lens.numel() != 1
        or not 1 <= rows <= 8
        or rows != spec.speculative_num_draft_tokens
        or spec.speculative_algorithm not in ("DSPARK", "DFLASH")
    ):
        raise ValueError(
            "SM70 MiniMax verification requires one static DSpark/DFlash block: "
            f"rows={rows}, requests={forward_batch.seq_lens.numel()}, "
            f"width={spec.speculative_num_draft_tokens}"
        )
    # CUDA-graph setup uses a generic verify-info object with placeholder tree
    # masks. The resolved DSpark/DFlash profile is a static causal chain; its
    # prefix and query positions, not that placeholder, define target visibility.
    idx_q, idx_k, idx_v = (kwargs[name] for name in ("idx_q", "idx_k", "idx_v"))
    if idx_v is not None or layer.layer_id not in self.disable_value_layer_ids:
        raise ValueError("SM70 MiniMax requires disabled index values")
    if not self._is_sparse_kv_cached_by_fusion(forward_batch, layer.layer_id):
        self.kv_pool.set_fused_kv_index_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
            idx_k,
            None,
            layer.k_scale_float,
            layer.v_scale_float,
            layer.idx_k_scale_float,
            layer.idx_v_scale_float,
        )
    kc, vc = self.kv_pool.layer_router.read(layer.layer_id).kv()
    prefix = forward_batch.seq_lens.to(torch.int32)
    lengths = prefix + rows
    offsets = torch.arange(0, rows + 1, rows, device=q.device, dtype=torch.int32)
    _, output = sparse_prefill(
        None,
        q,
        kc,
        vc,
        None,
        idx_q,
        self.kv_pool.get_index_k_buffer(layer.layer_id),
        None,
        None,
        self.req_to_token,
        forward_batch.req_pool_indices,
        offsets,
        lengths,
        prefix,
        rows,
        self.req_to_token.shape[1],
        self.block_size_q,
        self.block_size_k,
        self.topk_blocks,
        self.init_blocks,
        self.local_blocks,
        score_type=self.score_type,
        disable_index_value=True,
        q_scale=layer.q_scale_float,
        k_scale=layer.k_scale_float,
        v_scale=layer.v_scale_float,
        idx_q_scale=layer.idx_q_scale_float,
        idx_k_scale=layer.idx_k_scale_float,
        idx_v_scale=layer.idx_v_scale_float,
    )
    return None, output.reshape(rows, -1)


def _validate_sparse(q, keys, sink, idx_q, idx_sink, bq, bk, topk, kwargs):
    if not (
        q.dtype == keys.dtype == idx_q.dtype == torch.float16
        and q.shape[1:] in ((8, 128), (16, 128))
        and keys.shape[1:] == idx_q.shape[1:] == (1, 128)
        and bq == 1
        and bk == 128
        and topk == 16
        and sink is idx_sink is None
        and kwargs.get("disable_index_value")
        and kwargs.get("score_type") == "max"
        and kwargs.get("sm_scale") in (None, 128**-0.5)
        and kwargs.get("idx_sm_scale") in (None, 128**-0.5)
        and all(
            kwargs.get(s) in (None, 1.0)
            for s in (
                "q_scale",
                "k_scale",
                "v_scale",
                "idx_q_scale",
                "idx_k_scale",
                "idx_v_scale",
            )
        )
        and kwargs.get("cached_topk_idx") is None
        and kwargs.get("loc_mapping") is None
        and kwargs.get("indexer_cp") is None
    ):
        raise ValueError(
            "Unsupported SM70 MiniMax sparse geometry, cache or index policy"
        )


def quant_config_init(original, self, **kwargs):
    original(self, **kwargs)
    # Preserve the original group32 MXFP8 checkpoint. Converting it to block128
    # requantizes weights and selects FP8 Tensor Cores unavailable on Volta.
    self.convert_mxfp8_to_block_fp8 = False


def quant_method(original, config, layer, prefix):
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.utils import is_layer_skipped

    from .fp8 import SM70MXFP8LinearMethod

    if isinstance(layer, LinearBase) and not is_layer_skipped(
        prefix, config.ignored_layers, fused_mapping=config.packed_modules_mapping
    ):
        return SM70MXFP8LinearMethod(config)
    return original(config, layer, prefix)


def prepare_experts(original, method, layer):
    from .quantization import prepare_nvfp4_moe

    layer.w13_weight = layer.w13_weight_packed
    layer.w2_weight = layer.w2_weight_packed
    gate_up = layer.w13_weight_global_scale.float()
    if gate_up.ndim == 2:
        if not torch.equal(gate_up[:, 0], gate_up[:, 1]):
            raise ValueError("MiniMax SM70 requires equal gate/up global scales")
        gate_up = gate_up[:, 0]
    # MiniMax stores an encoding divisor, unlike ModelOpt's decode multiplier.
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        gate_up.reciprocal().contiguous(), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        layer.w2_weight_global_scale.float().reciprocal().contiguous(),
        requires_grad=False,
    )
    prepare_nvfp4_moe(None, layer, group_size=16)
    layer.register_parameter("w13_weight_packed", None)
    layer.register_parameter("w2_weight_packed", None)
    layer._mega_moe_nvfp4 = False
    layer._mega_moe_weights_built = False


def install_hooks(hooks, hook_type):
    from .quad_collectives import all_gather
    from .quantization import minimum_capability

    base = "sglang.srt.layers.quantization.minimax_m3_nvfp4."
    hooks.extend(
        [
            (
                "sglang.srt.distributed.parallel_state.GroupCoordinator._all_gather_into_tensor",
                all_gather,
                hook_type.AROUND,
            ),
            (
                "sglang.srt.speculative.draft_worker_common.draft_graph_has_memory",
                draft_graph_memory,
                hook_type.AROUND,
            ),
            (
                "sglang.srt.models.dflash.DFlashAttention.__init__",
                draft_attention_init,
                hook_type.AROUND,
            ),
            (
                "sglang.srt.models.minimax_m3.MiniMaxM3Attention.__init__",
                attention_init,
                hook_type.AROUND,
            ),
            (
                base + "MiniMaxM3Nvfp4Config.__init__",
                quant_config_init,
                hook_type.AROUND,
            ),
            (
                base + "MiniMaxM3Nvfp4Config.get_min_capability",
                minimum_capability,
                hook_type.AROUND,
            ),
            (
                base + "MiniMaxM3Nvfp4Config.get_quant_method",
                quant_method,
                hook_type.AROUND,
            ),
            (
                base + "MiniMaxM3Nvfp4MoE.get_min_capability",
                minimum_capability,
                hook_type.AROUND,
            ),
            (
                base + "MiniMaxM3Nvfp4MoE._process_marlin_weights",
                prepare_experts,
                hook_type.AROUND,
            ),
        ]
    )


def install_attention_hooks(hooks, hook_type):
    base = "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse."
    hooks.extend(
        [
            (
                "sglang.srt.layers.attention.minimax_sparse_backend.MiniMaxSparseAttnBackend.__init__",
                sparse_backend_init,
                hook_type.AROUND,
            ),
            (
                "sglang.srt.layers.attention.minimax_sparse_backend.MiniMaxSparseAttnBackend.forward_extend",
                sparse_extend,
                hook_type.AROUND,
            ),
            (base + "minimax_sparse_prefill", sparse_prefill, hook_type.AROUND),
            (base + "minimax_sparse_decode", sparse_decode, hook_type.AROUND),
        ]
    )
