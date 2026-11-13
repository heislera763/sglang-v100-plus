"""SM70 QSA execution; all metadata and scheduling remain mainline."""

import os

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend as BaseQSA,
)
from sglang.srt.runtime_context import get_schedule

from .dispatch import reject_fallback


def scoring_dtype(original, q, k):
    # The relative-column prefill entry point uses this shared dtype selector.
    # Volta MMA requires FP16 operands even when the checkpoint stores BF16.
    if (
        q.is_cuda
        and q.dtype == k.dtype == torch.float16
        and torch.cuda.get_device_capability(q.device) == (7, 0)
    ):
        return torch.float16
    return original(q, k)


class QwenSparseAttnBackend(BaseQSA):
    @staticmethod
    def _forward_sm70_dense_prefill(q, k, v, extend_lens, softmax_scale):
        """Run no-prefix QSA prefill through the packaged TileLang D256 kernel.

        K/V are packed request-major when there is no cached prefix.  The
        dense D256 kernel handles one sequence per launch, so split ragged
        batches at their existing packed boundaries.  This keeps the V100
        image self-contained: it intentionally does not ship the legacy
        external ``flash_attn_v100`` wheel.
        """
        from sglang_v100_plus.kernels.attention import (
            get_dense_prefix_d256_kernel,
        )

        if sum(extend_lens) != q.shape[0]:
            raise ValueError(
                "QSA dense-prefill packed rows do not match extend lengths: "
                f"rows={q.shape[0]}, extend={sum(extend_lens)}"
            )
        kernel = get_dense_prefix_d256_kernel(q.shape[1], k.shape[1])
        outputs = []
        row_start = 0
        for extend_len in extend_lens:
            row_end = row_start + extend_len
            if extend_len:
                outputs.append(
                    kernel(
                        q[row_start:row_end].contiguous(),
                        k[row_start:row_end].contiguous(),
                        v[row_start:row_end].contiguous(),
                        0,
                        softmax_scale,
                    )
                )
            row_start = row_end
        if not outputs:
            return q.new_empty(q.shape)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    @staticmethod
    def _can_use_sm70_sparse_prefill(
        q, k_buffer, v_buffer, forward_batch, topk_indices
    ) -> bool:
        return (
            forward_batch.forward_mode.is_extend()
            and q.is_cuda
            and torch.cuda.get_device_capability(q.device) == (7, 0)
            and q.dtype == torch.float16
            and q.ndim == 3
            and q.shape[0] > 0
            and q.shape[1:] in ((3, 256), (6, 256))
            and topk_indices.ndim == 2
            and topk_indices.shape[0] == q.shape[0]
            and topk_indices.shape[1] > 0
            and k_buffer.dtype in (torch.float16, torch.float8_e5m2)
            and v_buffer.dtype == k_buffer.dtype
            and k_buffer.ndim == 3
            and v_buffer.shape == k_buffer.shape
            and k_buffer.shape[1:] == (1, 256)
            and forward_batch.req_pool_indices.numel() == 1
            and forward_batch.seq_lens.numel() == 1
        )

    @staticmethod
    def _can_use_sm70_sparse_decode(
        q,
        k_buffer,
        v_buffer,
        forward_batch,
        metadata,
        topk_indices,
    ) -> bool:
        """Exact direct-cache QSA decode/verify specialization for Qwen3.8 TP4/TP8."""
        forward_mode = forward_batch.forward_mode
        return (
            (
                forward_mode.is_decode()
                or QwenSparseAttnBackend._is_speculative_paged_mode(forward_mode)
            )
            and q.is_cuda
            and torch.cuda.get_device_capability(q.device) == (7, 0)
            and q.dtype == torch.float16
            and q.ndim == 3
            and q.shape[1:] in ((3, 256), (6, 256))
            and q.shape[0] == metadata.sequence_lengths.numel()
            and topk_indices.ndim == 2
            and topk_indices.shape[0] == q.shape[0]
            and topk_indices.shape[1] > 0
            and k_buffer.dtype in (torch.float16, torch.float8_e5m2)
            and v_buffer.dtype == k_buffer.dtype
            and k_buffer.ndim == 3
            and v_buffer.shape == k_buffer.shape
            and k_buffer.shape[1:] == (1, 256)
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        topk_indices=None,
        **kwargs,
    ):
        if topk_indices is None:
            raise ValueError("QSA sparse attention requires topk_indices")
        pool = self.token_to_kv_pool
        rows = topk_indices.shape[0]
        q3 = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        if not self._is_speculative_paged_mode(forward_batch.forward_mode):
            lengths = forward_batch.seq_lens_cpu
            extend = forward_batch.extend_seq_lens_cpu
            limit = int(
                os.environ.get("SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS", "8192")
            )
            # Dense attention equals QSA only while the entire prefix fits
            # within the checkpoint's selection budget (2048 for this model).
            if self.qsa_profile is not None:
                limit = min(limit, self.qsa_profile.budget)
            no_prefix = (
                lengths is not None
                and extend is not None
                and len(lengths) > 0
                and all(int(n) == int(e) for n, e in zip(lengths, extend))
            )
            if (
                no_prefix
                and max(int(n) for n in lengths) <= limit
                and q3.shape[1:] in ((3, 256), (6, 256))
            ):
                if save_kv_cache:
                    pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
                out = self._forward_sm70_dense_prefill(
                    q3[:rows],
                    k[:rows],
                    v[:rows],
                    [int(n) for n in extend],
                    layer.scaling,
                )
                return self._pad_extend_output(out, q3.shape[0])
            kb = pool.get_key_buffer(layer.layer_id)
            vb = pool.get_value_buffer(layer.layer_id)
            if self._can_use_sm70_sparse_prefill(
                q3[:rows], kb, vb, forward_batch, topk_indices
            ):
                if save_kv_cache:
                    pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
                if (
                    self.qsa_profile is not None
                    and lengths is not None
                    and len(lengths) == 1
                    and rows >= 128
                    and get_schedule().disable_overlap_schedule
                    and get_schedule().max_running_requests == 1
                    and self.qsa_profile.budget
                    <= int(lengths[0])
                    <= 6 * self.qsa_profile.budget
                ):
                    # Sparse per-query scalar work underuses Volta. Share K/V
                    # tiles across queries/heads and mask their exact selections
                    # while using FP16 Tensor Cores with FP32 accumulation.
                    # Query/cache geometry and per-query membership determine
                    # eligibility; weight format and TP/PP placement do not.
                    from .kernels.qsa_prefill import qsa_masked_prefill

                    out = qsa_masked_prefill(
                        q3[:rows],
                        kb,
                        vb,
                        self.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        topk_indices.to(torch.int32).contiguous(),
                        int(lengths[0]),
                        layer.scaling,
                    )
                    return self._pad_extend_output(out, q3.shape[0])
                from .kernels.qsa_cuda import sm70_cuda_qsa_prefill

                out = sm70_cuda_qsa_prefill(
                    q3[:rows].contiguous(),
                    kb,
                    vb,
                    self.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    topk_indices.to(torch.int32).contiguous(),
                    forward_batch.seq_lens,
                    layer.scaling,
                )
                return self._pad_extend_output(out, q3.shape[0])
            reject_fallback(
                "qsa.prefill",
                "neither native dense no-prefix prefill nor single-request "
                "SM70 FP16 QSA prefill with FP16/E5M2 cache is supported",
                query=q3,
                key_cache=kb,
                value_cache=vb,
                indices=topk_indices,
            )
        return super().forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, topk_indices, **kwargs
        )

    def _forward_paged_attention(self, q, layer, forward_batch, topk_indices):
        pool = self.token_to_kv_pool
        kb = pool.get_key_buffer(layer.layer_id)
        vb = pool.get_value_buffer(layer.layer_id)
        metadata = self._resolve_metadata(forward_batch)
        if self._can_use_sm70_sparse_decode(
            q, kb, vb, forward_batch, metadata, topk_indices
        ):
            from .kernels.qsa_cuda import sm70_cuda_qsa_decode

            requests = metadata.row_req_pool_indices
            if requests is None:
                requests = forward_batch.req_pool_indices
            out = sm70_cuda_qsa_decode(
                q,
                kb,
                vb,
                self.req_to_token_pool.req_to_token,
                requests,
                topk_indices.to(torch.int32).contiguous(),
                metadata.sequence_lengths,
                layer.scaling,
            )
            return out.reshape(q.shape[0], -1)
        reject_fallback(
            "qsa.paged_attention",
            "native decode/verify requires SM70 FP16 [rows, 3|6, 256], "
            "matching row metadata/indices and FP16/E5M2 [pages, 1, 256] cache",
            query=q,
            key_cache=kb,
            value_cache=vb,
            indices=topk_indices,
        )
        return super()._forward_paged_attention(q, layer, forward_batch, topk_indices)


def project_qk(original, self, hidden_states, positions, **kwargs):
    if (
        self.index_n_heads == 4
        and self.index_head_dim == 128
        and kwargs.get("q_heads_padded") is not None
    ):
        kwargs["q_heads_padded"] = 4
    return original(self, hidden_states, positions, **kwargs)


def mqa_decode(
    original,
    q,
    k_cache,
    page_table,
    context_lens,
    max_model_len,
    score_scale=None,
    *,
    min_scored_len=0,
):
    # This is an upstream work-skipping hint. The SM70 kernel computes every
    # valid score, including rows whose consumer can select every block.
    if (
        q.dtype == torch.float16
        and q.shape[1:] == (4, 128)
        and k_cache.dtype == torch.float16
        and k_cache.shape[1] in (4, 16)
        and k_cache.shape[2:] == (1, 128)
    ):
        from .kernels.qsa_cuda import sm70_cuda_qsa_indexer_decode

        return sm70_cuda_qsa_indexer_decode(
            q,
            k_cache,
            page_table,
            context_lens,
            max_model_len,
            score_scale or q.shape[-1] ** 0.5,
        )
    reject_fallback(
        "qsa.indexer_decode",
        "native indexer requires FP16 [rows, 4, 128] query and FP16 "
        "[pages, 4 or 16, 1, 128] key cache",
        query=q,
        key_cache=k_cache,
    )
    return original(
        q,
        k_cache,
        page_table,
        context_lens,
        max_model_len,
        score_scale,
        min_scored_len=min_scored_len,
    )
