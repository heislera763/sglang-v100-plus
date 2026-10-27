"""Opt-in SM70 integration using mainline plugin hooks."""

import os

import msgspec
import torch
from sgl_kernel.sampling import (
    _top_k_renorm_probs_internal,
    _top_p_renorm_probs_internal,
)
from sgl_kernel.utils import _to_tensor_scalar_tuple

from .dispatch import eager_extend, in_prefill, reject_fallback


def install():
    redact_logs()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("SGLANG_V100_PLUS=1 requires an SM70 CUDA device")
    from .cuda_memory import install as install_module_memory_guard
    from .cuda_memory import require_single_session_capacity

    install_module_memory_guard()
    import sgl_kernel.elementwise as norm_ops

    import sglang.srt.layers.layernorm as norms
    from sglang.srt.arg_groups.choices import add_linear_attn_kernel_backend_choices
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from .checkpoint import include_nested_shards, load_glm_csf_weights
    from .commit_graph import commit_relayed_states
    from .dflash import (
        SM70DFlashGroupedConv,
        candidate_topk,
        forward_attention,
        forward_draft,
        initialize_attention,
        initialize_draft,
    )
    from .fp8 import (
        apply_fp8_dense,
        apply_fp8_moe,
        create_fp8_moe_runner,
        fp8_marlin_scalar_type,
        fp8_minimum_capability,
        fp8_route_block_size,
        mixed_quant_method,
        prepare_fp8_dense,
        prepare_fp8_moe,
    )
    from .fp16_moe import invoke_fp16_moe
    from .pipeline import (
        initialize_local_output,
        receive_output,
        send_output,
        send_output_dict,
        set_local_relay,
    )
    from .quantization import (
        dense_marlin_gemm,
        marlin_gemm,
        minimum_capability,
        moe_runner,
        prepare_nvfp4_dense,
        prepare_nvfp4_moe,
    )
    from .sampling import copy_penalty_metadata, exact_eagle_sample

    norm_ops._has_flashinfer = False
    norms._flashinfer_layernorm_available = False
    norms._flashinfer_rmsnorm_quant_available = False
    QSATokenToKVPool.index_state_dtype = torch.float16
    add_linear_attn_kernel_backend_choices(["tilelang_v100"])
    from .attention import (
        kda_dispatcher_init,
        prepare_cache_checkpoint,
        register_profile,
        resolve_profile,
        supports_cache_snapshots,
    )

    register_profile()
    from .glm_dsa import (
        SM70IndexerKPool,
        SM70SparseAttnBackend,
        sm70_dsa_cache_default,
        sm70_glm_kv_cache_dtype,
    )
    from .glm_mhc import mhc_post, mhc_pre, mhc_pre_dispatch
    from .hc_state import SM70GatedResidualState
    from .mqa import qsa_mqa_prefill
    from .ple import (
        Qwen4ExpPinnedHostEmbedding,
        _gather_ple_embedding_from_pinned_kernel,
    )
    from .qsa import QwenSparseAttnBackend, mqa_decode, project_qk

    def legacy_dtype(original, model_config):
        if model_config.dtype != torch.float16:
            raise ValueError("The enabled SM70 profile requires float16 model weights")

    hooks = [
        (
            "sglang.srt.layers.attention.triton_backend.TritonAttnBackend.__init__",
            initialize_attention,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.attention.triton_backend.TritonAttnBackend._forward_extend_kernel",
            forward_attention,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.schedule_batch.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend",
            prepare_cache_checkpoint,
            HookType.AROUND,
        ),
        (
            "sglang.srt.arg_groups.overrides.supports_mamba_cache_extra_buffer",
            supports_cache_snapshots,
            HookType.AROUND,
        ),
        (
            "sglang.srt.arg_groups.overrides._dsa_split_backend_resolution",
            resolve_profile,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.attention.linear.kda_backend.KDAKernelDispatcher.__init__",
            kda_dispatcher_init,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.moe.fused_moe_triton_kernels.invoke_fused_moe_kernel",
            invoke_fp16_moe,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._apply_token_constraints",
            require_single_session_capacity,
            HookType.AROUND,
        ),
        (
            "sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo.copy_for_forward",
            copy_penalty_metadata,
            HookType.AROUND,
        ),
        (
            "sglang.srt.speculative.eagle_worker_common.eagle_sample",
            exact_eagle_sample,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.dflash.DFlashGroupedConv",
            SM70DFlashGroupedConv,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.models.dflash._radix_topk",
            candidate_topk,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.dflash.DFlash2DraftModel.__init__",
            initialize_draft,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.dflash.DFlashDraftModel.forward",
            forward_draft,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.qwen4_exp.GatedResidualState",
            SM70GatedResidualState,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin.init_pp_loop_state",
            initialize_local_output,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin._pp_send_output_to_next_stage",
            send_output,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin._pp_recv_dict_from_prev_stage",
            receive_output,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin._pp_send_dict_to_next_stage",
            send_output_dict,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin._pp_spec_set_relay",
            set_local_relay,
            HookType.AROUND,
        ),
        (
            "sglang.srt.speculative.spec_utils.commit_mamba_states_after_verify",
            commit_relayed_states,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.select_marlin_moe_block_size",
            fp8_route_block_size,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.glm5_next.Glm5NextForConditionalGeneration.load_weights",
            load_glm_csf_weights,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_loader.loader.DefaultModelLoader._prepare_weights",
            include_nested_shards,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.modelopt_quant.ModelOptMixedPrecisionConfig.get_quant_method",
            mixed_quant_method,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8Config.get_min_capability",
            fp8_minimum_capability,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading",
            prepare_fp8_dense,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.apply",
            apply_fp8_dense,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8MoEMethod.create_moe_runner",
            create_fp8_moe_runner,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading",
            prepare_fp8_moe,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.fp8.Fp8MoEMethod.apply",
            apply_fp8_moe,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.get_scalar_type",
            fp8_marlin_scalar_type,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.runner.eager_runner.EagerRunner._execute_extend",
            eager_extend,
            HookType.AROUND,
        ),
        ("sglang.kernels.ops.layernorm.mhc._mhc_pre_torch", mhc_pre, HookType.AROUND),
        (
            "sglang.kernels.ops.layernorm.mhc._mhc_pre_dispatch",
            mhc_pre_dispatch,
            HookType.AROUND,
        ),
        ("sglang.kernels.ops.layernorm.mhc._mhc_post_torch", mhc_post, HookType.AROUND),
        (
            "sglang.srt.layers.quantization.marlin_utils_fp4.prepare_nvfp4_layer_for_marlin",
            prepare_nvfp4_dense,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.gemm.gptq_marlin.gptq_marlin_gemm",
            dense_marlin_gemm,
            HookType.AROUND,
        ),
        (
            "sglang.srt.arg_groups.overrides._dsa_kv_cache_dtype_default",
            sm70_dsa_cache_default,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.kv_cache_dtype.configure_kv_cache_dtype",
            sm70_glm_kv_cache_dtype,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.IndexerKPool",
            SM70IndexerKPool,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend",
            SM70SparseAttnBackend,
            HookType.REPLACE,
        ),
        (
            "flashinfer.sampling.top_k_renorm_probs",
            top_k_renorm_probs,
            HookType.REPLACE,
        ),
        (
            "flashinfer.sampling.top_p_renorm_probs",
            top_p_renorm_probs,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.model_executor.model_runner_components.load_model_utils.maybe_downgrade_dtype_for_legacy_gpu",
            legacy_dtype,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.modelopt_quant.ModelOptFp4Config.get_min_capability",
            minimum_capability,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.quantization.modelopt_quant.prepare_moe_nvfp4_layer_for_marlin",
            prepare_nvfp4_moe,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.moe.moe_wna16_marlin.moe_wna16_marlin_gemm",
            marlin_gemm,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.attention.linear.gdn_backend.GDNKernelDispatcher.__init__",
            dispatcher_init,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.moe.moe_runner.base.FusedOpPool.get_fused_func",
            moe_runner,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate",
            route_top10,
            HookType.AROUND,
        ),
        (
            "sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update_kernel.run",
            round_verify_state,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool.set_kv_buffer",
            store,
            HookType.AROUND,
        ),
        ("sglang.srt.server_args.prepare_server_args", arguments, HookType.AROUND),
        (
            "sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend",
            QwenSparseAttnBackend,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer.project_qk",
            project_qk,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.attention.qsa.mqa.qsa_mqa_decode",
            mqa_decode,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.attention.qsa.mqa.qsa_mqa_prefill",
            qsa_mqa_prefill,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.models.qwen4_exp._gather_ple_embedding_from_pinned_kernel",
            _gather_ple_embedding_from_pinned_kernel,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.models.qwen4_exp.Qwen4ExpPinnedHostEmbedding",
            Qwen4ExpPinnedHostEmbedding,
            HookType.REPLACE,
        ),
        (
            "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply",
            apply_unquant,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.hyperconnection.GatedResidual.__init__",
            initialize,
            HookType.AROUND,
        ),
        ("sglang.srt.layers.hyperconnection.GatedResidual.mix", mix, HookType.AROUND),
        (
            "sglang.srt.layers.hyperconnection.GatedResidual.combine",
            combine,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.qwen4_exp.Qwen4ExpNGramEmbedding._finish_embedding_lookup",
            embedding_output,
            HookType.AROUND,
        ),
    ]
    from .paired_prefill import all_reduce as paired_all_reduce
    from .paired_prefill import model_forward as paired_model_forward
    from .prefill import model_forward, prepare_attention

    hooks.extend(
        (
            (
                "sglang.srt.models.glm5_next.Glm5NextModel.forward",
                paired_model_forward,
                HookType.AROUND,
            ),
            (
                "sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce",
                paired_all_reduce,
                HookType.AROUND,
            ),
            (
                "sglang.srt.models.qwen4_exp.Qwen4ExpModel.forward",
                model_forward,
                HookType.AROUND,
            ),
            (
                "sglang.srt.models.qwen4_exp.Qwen4ExpLayerExtensionMixin._prepare_attn_stage",
                prepare_attention,
                HookType.AROUND,
            ),
        )
    )
    for target, replacement, kind in hooks:
        HookRegistry.register(target, replacement, kind)
    global REQUIRED_HOOKS
    REQUIRED_HOOKS = frozenset(target for target, _, _ in hooks)


def arguments(original, *args, **kwargs):
    result = original(*args, **kwargs)
    if result.api_key is None:
        result.api_key = os.environ.get("LLAMA_API_KEY")
    return result


def redact_logs():
    import logging

    secret = os.environ.get("LLAMA_API_KEY")
    if not secret:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        message = record.getMessage()
        if secret in message:
            record.msg = message.replace(secret, "[REDACTED]")
            record.args = ()
        return record

    logging.setLogRecordFactory(factory)


def initialize(original, self, config, *args, **kwargs):
    config = msgspec.structs.replace(config, params_dtype=torch.float16)
    return original(self, config, *args, **kwargs)


def mix(original, self, hyper_input, *, normalized_input=None):
    if normalized_input is not None:
        reject_fallback("qwen.hc_mix", "prepared HC norms require SM70 qualification")
        return original(self, hyper_input, normalized_input=normalized_input)

    if (
        hyper_input.ndim == 2
        and hyper_input.shape[0] > 4
        and hyper_input.shape[1] == 10240
        and hyper_input.dtype == torch.float16
        and hyper_input.is_cuda
        and torch.cuda.get_device_capability(hyper_input.device) == (7, 0)
        and in_prefill()
    ):
        from .prefill import gather_input, local_input

        hyper_input = local_input(hyper_input)
        # Prefill deliberately uses FP16 cuBLAS, not the small persistent
        # decode kernel. Keep the upstream prefill equations and rounding.
        if self.config.hc_per_branch_norm:
            normed = self.hc_norm(hyper_input)
        else:
            normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        result = self._mix_compute(
            normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            self.hc_count,
            self.hidden_size,
        ).to(self.params_dtype)
        return gather_input(result), (hyper_input, normed, None)
    if (
        hyper_input.ndim != 2
        or hyper_input.shape[0] not in (1, 2, 3, 4)
        or hyper_input.shape[1] != 10240
        or hyper_input.dtype != torch.float16
    ):
        reject_fallback(
            "qwen.hc_mix",
            "native HC requires FP16 [rows, 10240] with rows in (1, 2, 3, 4)",
            hyper_input=hyper_input,
        )
        return original(self, hyper_input)
    if self.config.hc_per_branch_norm:
        normed = self.hc_norm(hyper_input)
    else:
        normed = self.hc_norm(
            hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)
    if not sm70_hc_down_gemv_silu_supported(
        normed, self.input_mix_weight_down.weight, self.input_mix_weight_up.weight
    ):
        reject_fallback(
            "qwen.hc_mix",
            "native HC requires contiguous, aligned SM70 FP16 input/weights "
            "with down/up shapes (320, 10240)/(10240, 320); batched HC "
            "requires SGLANG_SM70_HC_NATIVE=1 and SGLANG_SM70_MTP_HC=1",
            input=normed,
            down_weight=self.input_mix_weight_down.weight,
            up_weight=self.input_mix_weight_up.weight,
        )
        return original(self, hyper_input)
    from .kernels.sm70_hc_mix import gate_supported, hc_down, hc_down_with_gate, hc_up

    gate = None
    if getattr(self, "_split_combine_ok", False) and gate_supported(
        normed, self.block_inject_weight.weight
    ):
        down, gate = hc_down_with_gate(
            normed, self.input_mix_weight_down.weight, self.block_inject_weight.weight
        )
    else:
        down = hc_down(normed, self.input_mix_weight_down.weight)
    result = hc_up(down, normed, self.input_mix_weight_up.weight).to(self.params_dtype)
    return result, (hyper_input, normed, gate)


def combine(original, self, block_output, residuals):
    if in_prefill():
        from .prefill import local_input

        block_output = local_input(block_output)
    if len(residuals) == 3 and residuals[2] is not None:
        from .kernels.sm70_hc_mix import hc_apply_gate

        return hc_apply_gate(block_output, residuals[0], residuals[2])
    if (
        len(residuals) in (2, 3)
        and block_output.is_cuda
        and block_output.dtype == torch.float16
        and block_output.ndim == 2
        and block_output.shape[1] == 2560
        and torch.cuda.get_device_capability(block_output.device) == (7, 0)
        and all(
            x.shape == (block_output.shape[0], 10240)
            and x.dtype == block_output.dtype
            and x.device == block_output.device
            and x.is_contiguous()
            for x in residuals[:2]
        )
        and self.block_inject_weight.weight.shape == (4, 10240)
        and self.block_inject_weight.weight.dtype == block_output.dtype
        and self.block_inject_weight.weight.device == block_output.device
        and self.block_inject_weight.weight.is_contiguous()
    ):
        from sglang.kernels.ops.elementwise.hc_combine import (
            hc_combine,
            hc_combine_split,
        )

        # Both are native FP16 CUDA kernels supported on SM70. Preserve the
        # original split/unsplit policy without delegating backend selection.
        op = (
            hc_combine_split
            if self._split_combine_ok and block_output.shape[0] <= 32
            else hc_combine
        )
        return op(
            block_output,
            residuals[0],
            residuals[1],
            self.block_inject_weight.weight,
            4,
            2560,
        )
    reject_fallback(
        "qwen.hc_combine",
        "native combine requires the HC gate state or contiguous SM70 FP16 "
        "[rows, 2560] output, [rows, 10240] residuals and [4, 10240] injection weights",
        output=block_output,
    )
    return original(
        self, block_output, residuals if len(residuals) == 3 else (*residuals, None)
    )


def embedding_output(original, self, *args, **kwargs):
    return original(self, *args, **kwargs).to(torch.float16)


def sm70_hc_down_gemv_silu_supported(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and torch.cuda.get_device_capability(x.device) == (7, 0)
        and x.dtype == torch.float16
        and w_down.dtype == torch.float16
        and w_up.dtype == torch.float16
        and w_down.device == x.device
        and w_up.device == x.device
        and x.ndim == 2
        and x.shape[1] == 10240
        and (
            x.shape[0] == 1
            or (
                x.shape[0] in (2, 3, 4)
                and os.environ.get("SGLANG_SM70_HC_NATIVE", "1") == "1"
                and os.environ.get("SGLANG_SM70_MTP_HC", "1") == "1"
                and x.data_ptr() % 16 == 0
                and w_down.data_ptr() % 16 == 0
                and w_up.data_ptr() % 16 == 0
            )
        )
        and w_down.shape == (320, 10240)
        and w_up.shape == (10240, 320)
        and x.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )


def store(
    original,
    self,
    layer,
    loc_info,
    key,
    value,
    k_scale=None,
    v_scale=None,
    layer_id_override=None,
    dcp_kv_mask=None,
):
    import torch

    if self.dtype == torch.float16:
        from .kv_cache import fp16_store_supported

        if fp16_store_supported(self, key, value, k_scale, v_scale, dcp_kv_mask):
            # Preserve upstream slot validation, layer selection and padding;
            # its declared CUDA byte-copy kernel supports SM70 FP16 storage.
            return original(
                self,
                layer,
                loc_info,
                key,
                value,
                k_scale,
                v_scale,
                layer_id_override,
                dcp_kv_mask,
            )
    if self.dtype == torch.float8_e5m2 and not self.use_hnd and dcp_kv_mask is None:
        from sglang.srt.mem_cache.memory_pool import unwrap_write_loc

        from .kernels.sm70_fp8_kv import write_fp8_e5m2_cache_sm70

        loc, _, _ = unwrap_write_loc(loc_info)
        idx = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        ) - self.start_layer
        if write_fp8_e5m2_cache_sm70(
            key, value, self.k_buffer[idx], self.v_buffer[idx], loc, k_scale, v_scale
        ):
            return
    reject_fallback(
        "qwen.fp8_kv_store",
        "native cache writer requires E5M2 cache, non-HND layout, no DCP mask "
        "and supported FP16 key/value tensors and scales",
        key=key,
        value=value,
    )
    return original(
        self,
        layer,
        loc_info,
        key,
        value,
        k_scale,
        v_scale,
        layer_id_override,
        dcp_kv_mask,
    )


def apply_unquant(original, self, layer, x, bias=None):
    from .kernels.gemm import blas_supported, linear_dense, supported

    if supported(x, layer.weight, bias):
        return linear_dense(x, layer.weight)
    if blas_supported(x, layer.weight, bias):
        return torch.nn.functional.linear(x, layer.weight, bias)
    reject_fallback(
        "gemm.unquantized_linear",
        "no declared SM70 linear backend; native dense/small GEMM requires SGLANG_SM70_DENSE_GEMV=1, "
        "aligned contiguous SM70 FP16 input/weights, no bias and a tuned "
        "shape (rows 1, 2, 3 or 4); batched shapes require SGLANG_SM70_MTP_SMALL_GEMM=1",
        input=x,
        weight=layer.weight,
        bias=bias,
    )
    return original(self, layer, x, bias)


def dispatcher_init(
    original, self, decode_backend, prefill_backend, verify_backend=None
):
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    use_tilelang = (
        prefill_backend.is_custom()
        and (mamba.linear_attn_prefill_backend or mamba.linear_attn_backend)
        == "tilelang_v100"
    )
    original(
        self,
        decode_backend,
        LinearAttnKernelBackend.TRITON if use_tilelang else prefill_backend,
        verify_backend,
    )
    if use_tilelang:
        from .kernels.gdn import TileLangGDNKernel

        self.extend_kernel = TileLangGDNKernel()


def round_verify_state(original, *args, **kwargs):
    import torch

    states = kwargs.get("intermediate_states_buffer")
    kwargs["QUANTIZE_STATE_EACH_STEP"] = (
        states is not None
        and states.dtype != torch.float32
        and not kwargs.get("HAS_EAGLE_TREE_CUSTOM_ATTN_MASK", False)
    )
    grid = kwargs["grid"]
    split_grid = kwargs.get("SPLIT_N_HV_GRID", False)
    sequences = grid[1] if split_grid else grid[2] // kwargs["HV"]
    if (
        states is not None
        and states.dtype == torch.float16
        and kwargs["q"].dtype == torch.float16
        and (
            kwargs["B"],
            sequences,
            kwargs["H"],
            kwargs["HV"],
            kwargs["K"],
            kwargs["V"],
        )
        == (1, 1, 4, 12, 128, 128)
        and kwargs["T"] in (2, 3, 4)
        and kwargs["DISABLE_STATE_UPDATE"]
        and not kwargs["IS_KDA"]
        and not kwargs["HAS_EAGLE_TREE_CUSTOM_ATTN_MASK"]
    ):
        # Preserve the fork's TP4 verification tile and the full launch grid.
        kwargs["BV"] = 8
        kwargs["grid"] = (16, *grid[1:]) if split_grid else (grid[0], 16, grid[2])
    return original(*args, **kwargs)


def route_top10(original, scores, bias, topk, *args, **kwargs):
    defaults = dict(
        scoring_func="softmax",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
        num_token_non_padded=None,
        packed_out=None,
        sqrtsoftplus_log1p=False,
    )
    if (
        not args
        and bias is None
        and topk == 10
        and kwargs.get("scoring_func") == "softmax"
        and all(
            key in defaults and value == defaults[key] for key, value in kwargs.items()
        )
        and scores.is_cuda
        and scores.dtype in (torch.float16, torch.float32)
        and scores.ndim == 2
        and scores.shape[0] > 0
        and scores.shape[1] == 512
        and scores.is_contiguous()
    ):
        from .kernels.sm70_nvfp4_moe_decode import sm70_topk10_softmax

        return sm70_topk10_softmax(scores)
    if (
        scores.is_cuda
        and scores.dtype in (torch.float16, torch.float32)
        and scores.ndim == 2
        and scores.shape[1] == 288
        and topk == 8
        and bias is not None
        and bias.shape == (288,)
        and bias.device == scores.device
        and bias.dtype in (torch.float16, torch.float32)
        and torch.cuda.get_device_capability(scores.device) == (7, 0)
    ):
        # GLM's biased top-8 is the mainline FP32 Triton router on SM70.
        # Its SM100-only radix branch cannot activate on this device.
        return original(scores, bias, topk, *args, **kwargs)
    reject_fallback(
        "moe.top10_router",
        "native router requires contiguous CUDA FP16/FP32 [rows, 512], "
        "positive rows, topk=10, no bias and default softmax routing options",
        scores=scores,
        bias=bias,
    )
    return original(scores, bias, topk, *args, **kwargs)


def top_k_renorm_probs(probs, top_k):
    if (
        probs.ndim == 2
        and 1 <= probs.shape[0] <= 4
        and 131072 <= probs.shape[1] <= 262144
        and probs.dtype == torch.float32
        and probs.is_cuda
    ):
        from sglang.kernels.ops.sampling.renorm_triton import (
            top_k_renorm_probs_triton,
        )

        # Small batches of wide probability rows leave most Volta SMs idle in
        # the one-block-per-row implementation. This selection depends on the
        # probability tensor, not weight quantization or distributed placement.
        # Sorting plus parallel mask/reduction retains the same cutoff ties.
        return top_k_renorm_probs_triton(probs, top_k)
    return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))


def top_p_renorm_probs(probs, top_p):
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))
