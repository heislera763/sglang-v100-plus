"""Block/MX FP8 weights with FP16 activations on Volta.

Retain mainline's checkpoint loading and expert mapping. Only finalization
and execution change. Weights stay packed E4M3; dense prefill unpacks only
one transient projection, while small-row kernels reconstruct values in flight.
"""

import torch

from sglang.srt.layers.quantization.base_config import LinearMethodBase


class SM70MXFP8LinearMethod(LinearMethodBase):
    """Serialized E4M3/UE8M0 groups32 in SM70 Marlin's packed layout.

    Reuse upstream parameter loading and TP slicing. Keep weights packed FP8;
    convert only the exact power-of-two scales to Marlin's FP16 bias convention.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.use_mxfp8 = True

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        if params_dtype != torch.float16:
            raise ValueError("SM70 MXFP8 requires FP16 activations")
        Fp8LinearMethod.create_fp8_weight_(
            layer=layer,
            block_quant=True,
            quant_config=self.quant_config,
            use_mxfp8=True,
            output_size_per_partition=sum(output_partition_sizes),
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=extra_weight_attrs.pop("weight_loader", None),
            is_checkpoint_fp8_serialized=True,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer):
        prepare_mxfp8_dense(self, layer)

    def _process_mxfp8_linear_weight_scale(self, layer):
        # MiniMax concatenates QKV/index projections before the loader's final
        # processing pass. Its fused holder owns both packed weights and scales.
        prepare_mxfp8_dense(self, layer)

    def apply(self, layer, x, bias=None):
        return apply_fp8_dense(None, self, layer, x, bias)


def mixed_quant_method(original, config, layer, prefix):
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.layers.quantization.utils import is_layer_skipped
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    if (
        isinstance(layer, (LinearBase, ParallelLMHead))
        and config._resolve_quant_algo(prefix) == "MXFP8"
        and not config.is_layer_excluded(prefix)
        and not is_layer_skipped(
            prefix, config.exclude_modules, config.packed_modules_mapping
        )
    ):
        if prefix.endswith(".kv_b_proj"):
            # checkpoint.py reconstructs the small absorbed MLA projection for
            # upstream's FP16 w_kc/w_vc BMM preparation, before post_load_weights.
            return UnquantizedLinearMethod()
        return SM70MXFP8LinearMethod(config.mxfp8_config)
    return original(config, layer, prefix)


def prepare_mxfp8_dense(method, layer):
    from sglang.srt.layers.utils import copy_or_rebind_param

    from .quantization import _dense_repack

    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    weight, scales = layer.weight, layer.weight_scale_inv
    if (
        layer.orig_dtype != torch.float16
        or not method.use_mxfp8
        or k % 32
        or weight.dtype != torch.float8_e4m3fn
        or weight.shape != (n, k)
        or scales.dtype != torch.uint8
        or scales.shape != (n, k // 32)
    ):
        raise ValueError("SM70 MXFP8 requires serialized E4M3 [N,K], UE8M0 [N,K/32]")
    # UE8M0 scale 255 is NaN. The exponent-bias factor is exact; reject any
    # exponent that would overflow or underflow Marlin's FP16 scale storage.
    exponent = scales.int() - 127
    decoded = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), exponent + 8)
    encoded = decoded.half()
    if bool((scales == 255).any()) or not bool(
        (torch.isfinite(encoded) & (encoded > 0)).all()
    ):
        raise ValueError(
            "MXFP8 scales cannot be represented exactly by SM70 FP16 scaling"
        )
    pn, pk = (n + 127) // 128 * 128, (k + 127) // 128 * 128
    weight = torch.nn.functional.pad(weight.view(torch.uint8), (0, pk - k, 0, pn - n))
    encoded = torch.nn.functional.pad(encoded, (0, (pk - k) // 32, 0, pn - n))
    copy_or_rebind_param(layer, "weight", _dense_repack(weight[None], num_bits=8)[0])
    copy_or_rebind_param(layer, "weight_scale_inv", encoded.T.contiguous())
    layer.workspace = torch.zeros(
        torch.cuda.get_device_properties(weight.device).multi_processor_count * 4,
        device=weight.device,
        dtype=torch.int32,
    )
    layer._sm70_fp8_physical_shape = (pn, pk)
    layer._sm70_fp8_dense_ready = True


def fp8_minimum_capability(original, config):
    return 70


def fp8_quant_method(original, config, layer, prefix):
    """Keep native MXFP8 draft projections in the same W8A16 layout."""
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.utils import is_layer_skipped

    if (
        config.use_mxfp8
        and isinstance(layer, LinearBase)
        and not is_layer_skipped(
            prefix, config.ignored_layers, fused_mapping=config.packed_modules_mapping
        )
    ):
        return SM70MXFP8LinearMethod(config)
    return original(config, layer, prefix)


def prepare_fp8_dense(original, method, layer):
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
    )
    from sglang.srt.layers.utils import copy_or_rebind_param

    from .quantization import _dense_repack

    if (
        method.use_mxfp8
        or method.weight_block_size != [128, 128]
        or not method.quant_config.is_checkpoint_fp8_serialized
        or layer.orig_dtype != torch.float16
        or getattr(layer, "bias", None) is not None
    ):
        raise ValueError(
            "SM70 dense FP8 requires serialized 128x128 blocks, FP16 and no bias"
        )
    weight, block_scales = layer.weight, layer.weight_scale_inv
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if (
        weight.dtype != torch.float8_e4m3fn
        or tuple(weight.shape) != (n, k)
        or n % 128
        or k % 128
        or tuple(block_scales.shape) != (n // 128, k // 128)
    ):
        raise ValueError(
            "SM70 dense FP8 dimensions must preserve complete 128x128 blocks"
        )
    # The SM70 iterator consumes logical [K/group,N] scales, unlike the
    # SM80 dense Marlin wrapper's permuted scale layout.
    scales = fp8_fused_exponent_bias_into_scales(
        block_scales.T.repeat_interleave(128, dim=1).to(torch.float16)
    )
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("SM70 dense FP8 scales overflow FP16")
    copy_or_rebind_param(layer, "weight", _dense_repack(weight[None], num_bits=8)[0])
    copy_or_rebind_param(layer, "weight_scale_inv", scales)
    layer.workspace = torch.zeros(
        torch.cuda.get_device_properties(weight.device).multi_processor_count * 4,
        device=weight.device,
        dtype=torch.int32,
    )
    layer._sm70_fp8_dense_ready = True


def apply_fp8_dense(original, method, layer, x, bias=None):
    from sglang.srt.layers.quantization.utils import get_scalar_types

    from .quantization import dense_marlin_gemm

    if not getattr(layer, "_sm70_fp8_dense_ready", False) or (
        x.dtype != torch.float16 or not x.is_cuda
    ):
        raise ValueError(
            "SM70 dense FP8 requires prepared weights and FP16 activations"
        )
    a = x.reshape(-1, x.shape[-1]).contiguous()
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if a.shape[1] != k:
        raise ValueError(
            "SM70 dense FP8 activation width does not match the checkpoint"
        )
    pn, pk = getattr(layer, "_sm70_fp8_physical_shape", (n, k))
    # Decode/verification reuse the unpacked values across their small batch.
    if 1 <= a.shape[0] <= 8:
        from sglang.kernels.ops.gemm.sm70_fp8_gemv import sm70_fp8_gemv
        from sglang.srt.runtime_context import get_buffer

        rows = a.shape[0]
        stream = torch.cuda.current_stream(a.device).cuda_stream
        partials = get_buffer(
            f"v100_fp8_dense_gemv:{a.device}:{stream}:{rows}:{n}:{pk}",
            lambda: torch.empty(
                (rows, pk // 128, n), device=a.device, dtype=torch.float32
            ),
        )
        output = torch.empty((rows, n), device=a.device, dtype=torch.float16)
        sm70_fp8_gemv(a, layer.weight, layer.weight_scale_inv, output, partials)
        if bias is not None:
            if bias.dtype != torch.float16 or bias.shape != (n,):
                raise ValueError("SM70 FP8 bias must match the logical FP16 output")
            output.add_(bias)
        return output.reshape(*x.shape[:-1], n)
    if a.shape[0] >= 128:
        from sglang.kernels.ops.gemm.sm70_fp8_gemm import sm70_fp8_gemm

        # Only one projection is expanded transiently. Keeping this allocation
        # local lets the allocator reuse it across layers and shapes; there is
        # no model-sized FP16 weight cache or per-shape persistent scratch.
        scratch = torch.empty((pk, pn), device=a.device, dtype=torch.float16)
        output = torch.empty((a.shape[0], n), device=a.device, dtype=torch.float16)
        sm70_fp8_gemm(a, layer.weight, layer.weight_scale_inv, output, scratch)
        if bias is not None:
            if bias.dtype != torch.float16 or bias.shape != (n,):
                raise ValueError("SM70 FP8 bias must match the logical FP16 output")
            output.add_(bias)
        return output.reshape(*x.shape[:-1], n)
    if pk != k:
        a = torch.nn.functional.pad(a, (0, pk - k))
    output = dense_marlin_gemm(
        None,
        a,
        None,
        layer.weight,
        layer.weight_scale_inv,
        None,
        None,
        None,
        None,
        layer.workspace,
        get_scalar_types()[1].float8_e4m3fn,
        a.shape[0],
        pn,
        pk,
    )
    if pn != n:
        output = output[:, :n].contiguous()
    if bias is not None:
        if bias.dtype != torch.float16 or bias.shape != (n,):
            raise ValueError("SM70 FP8 bias must match the logical FP16 output")
        output.add_(bias)
    return output.reshape(*x.shape[:-1], n)


def _validate(method):
    from sglang.srt.layers.moe import get_moe_runner_backend

    if (
        method.use_mxfp8
        or method.is_fp4_expert
        or method.weight_block_size != [128, 128]
        or not method.quant_config.is_checkpoint_fp8_serialized
        or not get_moe_runner_backend().is_marlin()
    ):
        raise ValueError(
            "SM70 FP8 experts require serialized E4M3 blocks [128,128] and Marlin"
        )


def create_fp8_moe_runner(original, method, layer, config):
    from sglang.srt.layers.moe.moe_runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    _validate(method)
    method.moe_runner_config = config
    method._owns_moe_runner = True
    method.runner = MoeRunner(MoeRunnerBackend.MARLIN, config)


def prepare_fp8_moe(original, method, layer):
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
    )
    from sglang.srt.layers.utils import copy_or_rebind_param

    from .quantization import _dense_repack

    _validate(method)
    if layer.params_dtype != torch.float16:
        raise ValueError("SM70 FP8 experts require FP16 activations")
    if any(
        getattr(layer, n, None) is not None
        for n in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("SM70 FP8 expert bias is unsupported")
    for prefix in ("w13", "w2"):
        weight = getattr(layer, prefix + "_weight")
        block_scales = getattr(layer, prefix + "_weight_scale_inv")
        e, n, k = weight.shape
        if weight.dtype != torch.float8_e4m3fn or n % 128 or k % 128:
            raise ValueError(
                "SM70 FP8 expert dimensions must preserve complete 128x128 blocks"
            )
        if tuple(block_scales.shape) != (e, n // 128, k // 128):
            raise ValueError("Invalid block-FP8 expert scale shape")
        packed = _dense_repack(weight, num_bits=8)
        scales = torch.empty(
            (e, k // 128, n), device=weight.device, dtype=torch.float16
        )
        for i in range(e):
            logical = block_scales[i].T.repeat_interleave(128, dim=1).to(torch.float16)
            # Unlike mainline Marlin, the SM70 iterator consumes logical
            # [K/group,N] scale order, without the SM80 tile permutation.
            scales[i] = fp8_fused_exponent_bias_into_scales(logical)
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("SM70 FP8 scales overflow FP16")
        copy_or_rebind_param(layer, prefix + "_weight", packed)
        name = prefix + "_weight_scale_inv"
        copy_or_rebind_param(layer, name, scales)
        # The shared Marlin runner otherwise interprets eight-bit storage as
        # integer quantization. Preserve the floating-point format explicitly.
        getattr(layer, name)._sm70_fp8_scale = True


def fp8_marlin_scalar_type(original, num_bits, has_zp, scales=None, global_scale=None):
    if getattr(scales, "_sm70_fp8_scale", False):
        from sglang.srt.layers.quantization.utils import get_scalar_types

        if num_bits != 8 or has_zp or global_scale is not None:
            raise ValueError("Invalid SM70 FP8 Marlin metadata")
        return get_scalar_types()[1].float8_e4m3fn
    return original(num_bits, has_zp, scales, global_scale)


def fp8_route_block_size(original, hidden, ids, w1, w2, scales):
    """Align compatible packed FP8 routes to the native kernel's 32-row CTA."""
    from sglang.srt.runtime_context import get_schedule

    from .kernels.moe_marlin import _sm70_marlin_user_tuning

    if (
        not _sm70_marlin_user_tuning
        and get_schedule().disable_overlap_schedule
        and get_schedule().max_running_requests == 1
        and hidden.dtype == scales.dtype == torch.float16
        and hidden.is_cuda
        and hidden.ndim == 2
        and hidden.shape[0] >= 128
        and hidden.shape[1] == 2560
        and ids.shape == (hidden.shape[0], 10)
        and ids.dtype == torch.int32
        and len(w1.shape) == len(w2.shape) == 3
        and w1.shape[0] == w2.shape[0] > 0
        and tuple(w1.shape[1:]) == (160, 5120)
        and tuple(w2.shape[1:]) == (40, 10240)
        and getattr(scales, "_sm70_fp8_scale", False)
    ):
        return 32
    return original(hidden, ids, w1, w2, scales)


def apply_fp8_moe(original, method, layer, dispatch_output):
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.runtime_context import get_exec, get_schedule

    config = method.moe_runner_config
    hidden = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    if (
        # Scratch is shared by serialized forwards. TP/PP placement and the
        # number of locally stored experts do not change this operation.
        get_schedule().disable_overlap_schedule
        and not get_exec().overlap.enable_two_batch_overlap
        and not get_exec().overlap.enable_single_batch_overlap
        and hidden.ndim == 2
        and 1 <= hidden.shape[0] <= 4
        and hidden.shape[1] == 2560
        and hidden.dtype == torch.float16
        and TopKOutputChecker.format_is_standard(topk)
        and topk.topk_ids.shape == (hidden.shape[0], 10)
        and layer.w13_weight.shape[1:] == (160, 5120)
        and layer.w2_weight.shape[1:] == (40, 10240)
        and config.is_gated
        and config.activation == "silu"
        and not config.apply_router_weight_on_input
        and config.routed_scaling_factor is None
        and config.gemm1_alpha is None
        and config.gemm1_clamp_limit is None
        and config.swiglu_limit is None
    ):
        from .kernels.sm70_fp8_moe_decode import fp8_moe_decode

        return StandardCombineInput(
            hidden_states=fp8_moe_decode(
                hidden,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                topk.topk_ids,
                topk.topk_weights,
            )
        )

    mapping = layer.dispatcher.local_expert_mapping
    quant = MarlinMoeQuantInfo(
        w13_qweight=layer.w13_weight,
        w2_qweight=layer.w2_weight,
        w13_scales=layer.w13_weight_scale_inv,
        w2_scales=layer.w2_weight_scale_inv,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=8,
        expert_map=mapping,
        # StandardDispatcher has already translated global IDs to local IDs
        # (or -1). Alignment needs only the stored experts, not empty global
        # expert slots on every rank.
        global_num_experts=layer.w13_weight.shape[0] if mapping is not None else -1,
    )
    return method.runner.run(dispatch_output, quant)
