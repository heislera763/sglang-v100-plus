"""Request-owned GLM pool4: eager prefill, decode and chain MTP on SM70."""

import logging

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsa.sm70_indexer import (
    fp8_quantize_sm70,
    kpool_compress_sm70,
)
from sglang.kernels.ops.attention.dsa.sm70_indexer_decode import (
    mqa_logits_decode_sm70,
)
from sglang.kernels.ops.attention.dsa.sm70_indexer_prefill import (
    mqa_logits_prefill_sm70,
)
from sglang.kernels.ops.attention.dsa.sm70_pool4_decode import (
    pool4_cache_page_size,
    pool4_decode_sm70,
    pool4_spec_sm70,
)
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    topk_from_pooled_history_logits,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
    get_token_to_kv_pool,
)

from .prefill import indexer_partition

logger = logging.getLogger(__name__)


def sm70_glm_kv_cache_dtype(original, **kwargs):
    """Keep GLM's supported compute-dtype KV when auto sees an FP8 recipe.

    Weight quantization and the separate FP8 indexer are unchanged. The plugin
    only registers this adapter on SM70; explicit cache requests stay explicit.
    """
    model = kwargs.get("model")
    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    quant = getattr(model, "quant_config", None)
    draft_dtype = kwargs.get("speculative_draft_kv_cache_dtype")
    effective = (
        draft_dtype
        if kwargs.get("is_draft_worker") and draft_dtype is not None
        else kwargs.get("server_args_kv_cache_dtype")
    )
    minimax = bool(
        set(getattr(getattr(model, "config", None), "architectures", ()) or ())
        & {
            "MiniMaxM3SparseForConditionalGeneration",
            "MiniMaxM3SparseForCausalLM",
            "DSparkMiniMaxDraftModel",
        }
    )
    if (
        (
            getattr(config, "model_type", None) in ("glm5_next", "glm5_next_text")
            or minimax
        )
        and kwargs.get("model_dtype") == torch.float16
        and effective == "auto"
        and str(getattr(quant, "kv_cache_quant_algo", "")).upper() == "FP8"
    ):
        logger.info(
            "SM70 auto KV uses FP16; checkpoint FP8 KV recipe requires "
            "separate backend qualification. Weight quantization is unchanged."
        )
        return "auto", torch.float16
    return original(**kwargs)


def sm70_dsa_cache_default(original, view):
    from sglang.srt.arg_groups.overrides import model_config_of

    config = model_config_of(view).hf_config
    if config.architectures[0] == "Glm5NextForConditionalGeneration":
        if view.attention_backend != "v100_plus":
            raise ValueError("GLM on SM70 requires --attention-backend v100_plus")
        if view.dtype not in ("float16", "half") or view.kv_cache_dtype != "auto":
            raise ValueError(
                "V100 GLM requires float16 weights/state and compute-dtype auto KV"
            )
        return {}
    return original(view)


def pooled_locations(token_table, pool_ids, *, token_page_size, index_page_size):
    # Each index page represents four tokens per pooled slot. The allocator's
    # token page size and the index cache's slot count are separate quantities.
    token_page_starts = (
        torch.div(pool_ids, index_page_size, rounding_mode="floor")
        * index_page_size
        * 4
    )
    token_slots = token_table[token_page_starts.long()].long()
    return (
        torch.div(token_slots, token_page_size, rounding_mode="floor") * index_page_size
        + pool_ids % index_page_size
    )


def write_pooled_cache(buf, locations, keys, scales):
    index_page_size = pool4_cache_page_size(buf)
    pages = torch.div(locations, index_page_size, rounding_mode="floor").long()
    offsets = locations.remainder(index_page_size).long()
    columns = offsets[:, None] * 128 + torch.arange(128, device=buf.device)[None, :]
    buf[pages[:, None], columns] = keys.contiguous().view(torch.uint8)
    scale_buffer = buf[:, index_page_size * 128 :].view(torch.float32)
    scale_buffer[pages, offsets] = scales.reshape(-1)


def read_pooled_cache(buf, locations):
    index_page_size = pool4_cache_page_size(buf)
    pages = torch.div(locations, index_page_size, rounding_mode="floor").long()
    offsets = locations.remainder(index_page_size).long()
    columns = offsets[:, None] * 128 + torch.arange(128, device=buf.device)[None, :]
    keys = buf[pages[:, None], columns].contiguous().view(torch.float8_e4m3fn)
    scales = buf[:, index_page_size * 128 :].view(torch.float32)[pages, offsets]
    return keys, scales


class SM70IndexerKPool(IndexerKPool):
    # Bound the reduced [query, pooled-history] FP32 score tensor.
    # A query's GEMM reduction and causal pool/tail lengths remain independent
    # of this launch batching; cache writes still happen once per input chunk.
    _prefill_query_chunk_size = 128

    def __init__(self, *args, **kwargs):
        kwargs["alt_stream"] = None
        super().__init__(*args, **kwargs)
        if self.index_kpool != 4 or self.head_dim != 128 or not self.skip_rope:
            raise ValueError("SM70 K-pool supports GLM's pool4/no-RoPE profile")

    def forward_cuda(
        self, x, q_lora, positions, forward_batch, layer_id, return_indices=True
    ):
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            return self._forward_spec(
                x, q_lora, positions, forward_batch, layer_id, return_indices
            )
        if torch.cuda.is_current_stream_capturing() and not mode.is_decode_or_idle():
            raise RuntimeError("SM70 GLM indexer supports decode graphs only")
        if mode.is_idle():
            return torch.full(
                (x.shape[0], self.index_topk + 3),
                -1,
                device=x.device,
                dtype=torch.int32,
            )
        if not (mode.is_extend_without_speculative() or mode.is_decode()):
            raise NotImplementedError(
                "SM70 eager GLM indexer supports ordinary prefill/decode only"
            )
        metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
        if metadata is None:
            return None
        query, key, _, _ = self._get_q_k_bf16(
            q_lora, x, positions, False, forward_batch
        )
        gates = F.linear(x, self.index_kpool_compress_gate)
        query_fp8, query_scale = fp8_quantize_sm70(query, self.scale_fmt is not None)
        head_weights = self.weights_proj(x.float())[0] * (
            self.n_heads**-0.5 * self.softmax_scale
        )
        weights = head_weights * query_scale.squeeze(-1)
        pool = get_token_to_kv_pool()
        buffer = pool.get_index_k_with_scale_buffer(layer_id)
        index_page_size = pool4_cache_page_size(buffer)
        # A replicated DCP draft has widened allocator pages. The packed
        # index buffer still addresses one index_page_size*kpool token span.
        tail_k, tail_s = pool.get_compress_tail_buffers(layer_id)
        table = get_req_to_token_pool().req_to_token
        if mode.is_decode():
            cached_keys, cached_scales, pools = pool4_decode_sm70(
                key,
                gates,
                self.index_kpool_compress_ape,
                tail_k,
                tail_s,
                buffer,
                table,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                self.scale_fmt is not None,
                retain_closed_tail=tail_k.shape[1] > 4,
                token_page_size=index_page_size * self.index_kpool,
            )
            if not return_indices:
                return None
            logits = mqa_logits_decode_sm70(
                query_fp8, cached_keys, cached_scales, weights, pools
            )
            return topk_from_pooled_history_logits(
                logits,
                pools,
                4,
                self.index_topk,
                page_table=table,
                seq_lens=forward_batch.seq_lens,
                page_table_row_index=forward_batch.req_pool_indices.to(torch.int32),
            )
        backend = get_attn_backend()
        sparse_backend = getattr(backend, "full_attn_backend", backend)
        request_ids = sparse_backend.sm70_request_ids
        seq_lens = forward_batch.seq_lens_cpu.tolist()
        chunk_lens = (
            forward_batch.extend_seq_lens_cpu
            if mode.is_extend_without_speculative()
            else [1] * len(request_ids)
        )
        outputs = []
        offset = 0
        for request, seq_len, count in zip(request_ids, seq_lens, chunk_lens):
            prefix = seq_len - count
            if count == 0:
                continue
            token_table = table[request]
            old_tail = prefix % 4
            tail_positions = (
                torch.arange(prefix - old_tail, prefix, device=x.device)
                % tail_k.shape[1]
            )
            all_keys = torch.cat(
                (
                    tail_k[request, tail_positions],
                    key[offset : offset + count].bfloat16(),
                )
            )
            all_scores = torch.cat(
                (
                    tail_s[request, tail_positions].float(),
                    gates[offset : offset + count].float(),
                )
            )
            closed = all_keys.shape[0] // 4
            if closed:
                compressed, scales = kpool_compress_sm70(
                    all_keys[: closed * 4].reshape(closed, 4, 128),
                    all_scores[: closed * 4].reshape(closed, 4, 128),
                    self.index_kpool_compress_ape,
                    self.scale_fmt is not None,
                )
                ids = prefix // 4 + torch.arange(closed, device=x.device)
                write_pooled_cache(
                    buffer,
                    pooled_locations(
                        token_table,
                        ids,
                        token_page_size=index_page_size * self.index_kpool,
                        index_page_size=index_page_size,
                    ),
                    compressed,
                    scales,
                )
            remainder = seq_len % 4
            if remainder:
                tail_positions = (
                    torch.arange(seq_len - remainder, seq_len, device=x.device)
                    % tail_k.shape[1]
                )
                tail_k[request, tail_positions] = all_keys[-remainder:]
                tail_s[request, tail_positions] = all_scores[-remainder:].to(
                    tail_s.dtype
                )
            if return_indices:
                max_pools = seq_len // 4
                ids = torch.arange(max_pools, device=x.device)
                cached_keys, cached_scales = read_pooled_cache(
                    buffer,
                    pooled_locations(
                        token_table,
                        ids,
                        token_page_size=index_page_size * self.index_kpool,
                        index_page_size=index_page_size,
                    ),
                )
                # This index bank is shared by every query subgroup. Convert
                # it once; latent KV and the resident FP8 index cache stay as-is.
                cached_keys = cached_keys.half()
                # The fused scorer writes only [query,history] FP32 scores.
                # Bound that actual scratch, without the eliminated head axis.
                score_row_bytes = max(max_pools, 1) * 4
                score_rows = max(1, (128 << 20) // score_row_bytes)
                score_rows = 1 << (score_rows.bit_length() - 1)
                query_chunk = min(
                    self._prefill_query_chunk_size,
                    128 if seq_len <= 8192 else 32,
                    score_rows,
                )
                # All ranks still publish the complete index cache above.
                # Only independent query scoring/top-k is divided within DCP;
                # exchange final physical IDs once, after the whole chunk.
                partition = indexer_partition(count, max_pools)
                first = (
                    0
                    if partition is None
                    else min(count, partition.rank * partition.local_rows)
                )
                last = (
                    count
                    if partition is None
                    else min(count, first + partition.local_rows)
                )
                selected_chunks = []
                for start in range(first, last, query_chunk):
                    stop = min(start + query_chunk, last)
                    lengths = torch.arange(
                        prefix + start + 1,
                        prefix + stop + 1,
                        device=x.device,
                        dtype=torch.int32,
                    )
                    pools = torch.div(lengths, 4, rounding_mode="floor")
                    logits = mqa_logits_prefill_sm70(
                        query_fp8[offset + start : offset + stop],
                        cached_keys,
                        cached_scales,
                        weights[offset + start : offset + stop],
                        pools,
                    )
                    if max_pools == 0:
                        selected = torch.full(
                            (stop - start, self.index_topk + 3),
                            -1,
                            device=x.device,
                            dtype=torch.int32,
                        )
                        for row, length in enumerate(
                            range(prefix + start + 1, prefix + stop + 1)
                        ):
                            selected[row, :length] = token_table[:length].int()
                    else:
                        selected = topk_from_pooled_history_logits(
                            logits,
                            pools,
                            4,
                            self.index_topk,
                            page_table=token_table[None, :].expand(stop - start, -1),
                            seq_lens=lengths,
                        )
                    selected_chunks.append(selected)
                selected = torch.cat(selected_chunks)
                if partition is not None:
                    selected = F.pad(
                        selected,
                        (0, 0, 0, partition.local_rows - (last - first)),
                        value=-1,
                    )
                    selected = partition.gather(selected)
                outputs.append(selected)
            offset += count
        if not return_indices or not outputs:
            return None
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def _forward_spec(
        self, x, q_lora, positions, forward_batch, layer_id, return_indices
    ):
        from sglang.srt.runtime_context import get_spec

        if get_spec().speculative_eagle_topk != 1:
            raise NotImplementedError(
                "SM70 pool4 speculation supports linear chains only"
            )
        metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
        if metadata is None:
            return None
        plan = metadata.attn_metadata.kpool_write_plan
        if plan is None:
            raise RuntimeError(
                "SM70 speculation requires the upstream k-pool write plan"
            )
        query, key, _, _ = self._get_q_k_bf16(
            q_lora, x, positions, False, forward_batch
        )
        gates = F.linear(x, self.index_kpool_compress_gate)
        pool = get_token_to_kv_pool()
        tail_k, tail_s = pool.get_compress_tail_buffers(layer_id)
        requests = plan.req.repeat_interleave(plan.num_draft_tokens)
        lengths = plan.seqlens_per_q
        cached_keys, cached_scales, pools = pool4_spec_sm70(
            key,
            gates,
            self.index_kpool_compress_ape,
            tail_k,
            tail_s,
            pool.get_index_k_with_scale_buffer(layer_id),
            get_req_to_token_pool().req_to_token,
            plan.req,
            plan.write_start,
            plan.tail_logical_start,
            plan.write_loc,
            forward_batch.out_cache_loc,
            requests,
            lengths,
            effective_num_tokens=plan.effective_n_per_batch,
            round_scale=self.scale_fmt is not None,
            token_page_size=pool4_cache_page_size(
                pool.get_index_k_with_scale_buffer(layer_id)
            )
            * self.index_kpool,
        )
        if not return_indices:
            return None
        query_fp8, query_scale = fp8_quantize_sm70(query, self.scale_fmt is not None)
        weights = (
            self.weights_proj(x.float())[0]
            * (self.n_heads**-0.5 * self.softmax_scale)
            * query_scale.squeeze(-1)
        )
        logits = mqa_logits_decode_sm70(
            query_fp8, cached_keys, cached_scales, weights, pools
        )
        return topk_from_pooled_history_logits(
            logits,
            pools,
            4,
            self.index_topk,
            page_table=get_req_to_token_pool().req_to_token,
            seq_lens=lengths,
            page_table_row_index=requests.to(torch.int32),
        )


class SM70SparseAttnBackend(DeepseekSparseAttnBackend):
    # Prefill and ordinary eager attention metadata still use the CPU mirror.
    # The decode indexer and graph replay use live device lengths exclusively.
    needs_cpu_seq_lens = True

    def init_forward_metadata(self, forward_batch):
        request_ids = forward_batch.req_pool_indices_cpu
        if request_ids is None:
            request_ids = forward_batch.req_pool_indices.cpu()
        # One synchronization per batch on decode, instead of one at each of
        # the eleven indexer layers; prefill uses the existing CPU mirror.
        self.sm70_request_ids = request_ids.tolist()
        return super().init_forward_metadata(forward_batch)

    def _build_kpool_paged_mqa_schedule_metadata(self):
        return False

    def _init_kpool_metadata(self, metadata, forward_batch, *args, **kwargs):
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            return super()._init_kpool_metadata(
                metadata, forward_batch, *args, **kwargs
            )
        # The adapter owns its cache writes and causal lengths, so no native
        # DeepGEMM/fused-write plan is needed.
        return metadata

    def _init_kpool_metadata_capture(self, metadata, bs, forward_mode):
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            return super()._init_kpool_metadata_capture(metadata, bs, forward_mode)
        # The SM70 decode kernel reads live device request/length tensors; it
        # does not consume native DeepGEMM schedules or pooled write plans.
        return metadata

    def _update_kpool_metadata_replay(
        self,
        metadata,
        seq_lens,
        req_pool_indices,
        forward_mode,
        effective_n_per_batch=None,
    ):
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            return super()._update_kpool_metadata_replay(
                metadata,
                seq_lens,
                req_pool_indices,
                forward_mode,
                effective_n_per_batch,
            )
        return

    def _update_kpool_metadata_from_precomputed(
        self, metadata, precomputed, forward_mode
    ):
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            return super()._update_kpool_metadata_from_precomputed(
                metadata, precomputed, forward_mode
            )
        # Multi-step draft graphs enter here rather than the ordinary replay
        # entry point. SM70 decode reads live lengths and owns its cache writes;
        # it has no native decode write plan to refresh or copy between steps.
        return

    def _refresh_paged_mqa_schedule_metadata(self, *args, **kwargs):
        return

    def _check_kpool_tail_backend(self, topk_indices, dsa_impl, phase):
        if dsa_impl == "v100_plus":
            return
        return super()._check_kpool_tail_backend(topk_indices, dsa_impl, phase)

    def _forward_v100_sparse(
        self, q_nope, q_rope, kv_cache, page_table_1, sm_scale, v_head_dim
    ):
        from sglang.kernels.ops.attention.v100 import sparse_attention_v100

        if q_rope.shape[-1] or v_head_dim != 512:
            raise ValueError(
                "v100_plus sparse attention requires latent512 and no separate RoPE tail"
            )
        if not hasattr(self, "v100_sparse_workspaces"):
            self.v100_sparse_workspaces = {}
        stream = int(torch.cuda.current_stream(q_nope.device).cuda_stream)
        workspace = self.v100_sparse_workspaces.setdefault(stream, {})
        lengths = None
        if self.dcp_size > 1:
            from sglang.kernels.ops.attention.dsa.transform_index import (
                transform_index_page_table_dcp,
            )

            # Index-K stays replicated: only latent KV rows are striped.
            # Select every owned key (including the live pool4 tail), keeping
            # score order and duplicate multiplicity. No full-history gather.
            page_table_1, lengths = transform_index_page_table_dcp(
                page_table_1.reshape(q_nope.shape[0], -1),
                self.dcp_size,
                self.dcp_rank,
            )
        return sparse_attention_v100(
            q_nope,
            kv_cache,
            page_table_1,
            sm_scale,
            lengths=lengths,
            workspace=workspace,
            return_lse=self.dcp_size > 1,
        )
