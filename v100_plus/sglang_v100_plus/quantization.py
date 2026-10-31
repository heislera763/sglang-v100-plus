"""SM70 NVFP4 layout and dispatch, retaining mainline loading and MoE APIs."""

import os
from pathlib import Path

import torch

from .dispatch import reject_fallback


def minimum_capability(original, cls):
    return 70


def prepare_nvfp4_dense(original, layer):
    from sglang.srt.layers.utils import copy_or_rebind_param

    if layer.params_dtype != torch.float16 or layer.quant_config.group_size != 16:
        raise ValueError("V100 dense NVFP4 requires FP16 activations and group_size=16")
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    padded_n, padded_k = min(
        (
            ((n + 63) // 64 * 64, (k + 127) // 128 * 128),
            ((n + 127) // 128 * 128, (k + 63) // 64 * 64),
        ),
        key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]),
    )
    weight = torch.nn.functional.pad(
        layer.weight, (0, (padded_k - k) // 2, 0, padded_n - n)
    )
    # FP8 indexing/padding is not native arithmetic on Volta; pad byte storage.
    scales = torch.nn.functional.pad(
        layer.weight_scale.view(torch.uint8), (0, (padded_k - k) // 16, 0, padded_n - n)
    ).view(torch.float8_e4m3fn)
    encoded, factor = sm70_nvfp4_marlin_process_scales(
        scales.T.contiguous()[None], torch.float16
    )
    global_scale = (
        sm70_nvfp4_marlin_process_global_scale(
            layer.weight_global_scale, torch.float16
        ).reshape(-1)
        / factor
    )
    copy_or_rebind_param(layer, "weight", _dense_repack(weight[None])[0])
    copy_or_rebind_param(layer, "weight_scale", encoded[0])
    copy_or_rebind_param(layer, "weight_global_scale", global_scale)
    layer.workspace = torch.zeros(
        torch.cuda.get_device_properties(weight.device).multi_processor_count * 4,
        device=weight.device,
        dtype=torch.int32,
    )
    bias = getattr(layer, "bias", None)
    if bias is not None and (bias.dtype != torch.float16 or bias.shape != (n,)):
        raise ValueError("V100 dense NVFP4 bias must match the logical FP16 output")


def dense_marlin_gemm(
    original,
    a,
    c,
    b_q_weight,
    b_scales,
    global_scale,
    b_zeros,
    g_idx,
    perm,
    workspace,
    b_q_type,
    size_m,
    size_n,
    size_k,
    is_k_full=True,
    use_atomic_add=False,
    use_fp32_reduce=False,
    is_zp_float=False,
):
    from sglang.srt.layers.quantization.utils import get_scalar_types
    from sglang.srt.runtime_context import get_buffer

    from .kernels.moe_marlin import moe_wna16_marlin_gemm

    block_fp8 = (
        global_scale is None
        and b_scales.dtype == torch.float16
        and b_q_type.id == get_scalar_types()[1].float8_e4m3fn.id
    )
    if a.dtype != torch.float16 or (global_scale is None and not block_fp8):
        raise ValueError("V100 dense Marlin requires FP16 NVFP4 or block-FP8")
    # The installed NVFP4 extension includes the expert GEMM. A single expert
    # with one route per token computes the same dense multiplication, using
    # its existing logical scale layout and repacked weight format.
    block = 32
    padded = (size_m + block - 1) // block * block

    def make_routes():
        return (
            torch.arange(padded, device=a.device, dtype=torch.int32),
            torch.zeros(padded // block, device=a.device, dtype=torch.int32),
            torch.full((1,), padded, device=a.device, dtype=torch.int32),
            torch.ones((size_m, 1), device=a.device, dtype=torch.float32),
        )

    # Immutable routing metadata is identical for every dense layer with this
    # batch shape. Keep it owned by the runtime, rather than allocating/filling
    # four tensors (including a host-to-device copy) for every GEMM.
    routes = get_buffer(f"v100_dense_routes:{a.device}:{size_m}", make_routes)
    return moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight[None],
        None,
        b_scales[None],
        None if global_scale is None else global_scale.reshape(1),
        b_zeros,
        g_idx,
        perm,
        workspace,
        *routes,
        block,
        1,
        False,
        False,
        b_q_type,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


def _dense_repack(weight, num_bits=4):
    directory = Path(os.environ["SGLANG_V100_MARLIN_DIR"])
    libraries = list(directory.glob("_C*.so"))
    if not libraries:
        raise RuntimeError("Build the SM70 dense Marlin extension before loading NVFP4")
    torch.ops.load_library(str(libraries[0]))
    e, n, packed_k = weight.shape
    k = packed_k * (8 // num_bits)
    perm = torch.empty(0, device=weight.device, dtype=torch.int32)
    result = torch.empty(
        (e, k // 16, n * (num_bits // 2)), device=weight.device, dtype=torch.int32
    )
    for i in range(e):
        # Repack one expert at a time. Transposing the entire expert bank
        # duplicated hundreds of MiB while the whole checkpoint was resident.
        inputs = weight[i].contiguous().view(torch.int32).T.contiguous()
        result[i] = torch.ops._C.gptq_marlin_repack(inputs, perm, k, n, num_bits, False)
    return result


def pad_nvfp4_expert_width(layer):
    """Preserve both gated halves when an 80-wide shard needs a 64-column tile."""
    from sglang.srt.layers.utils import copy_or_rebind_param

    w13, w2 = layer.w13_weight, layer.w2_weight
    if (
        w13.ndim != 3
        or w2.ndim != 3
        or w13.shape[0] != w2.shape[0]
        or tuple(w13.shape[1:]) != (160, 1280)
        or tuple(w2.shape[1:]) != (2560, 40)
        or w13.dtype != torch.uint8
        or w2.dtype != torch.uint8
    ):
        return
    # Marlin requires N divisible by64. Padding each gate/up half80->96
    # also gives FC2 a K divisible by32. Checkpoint bytes/scales stay intact;
    # new columns and FC2 rows represent exact zeros, not requantization.
    for prefix in ("w13", "w2"):
        weight = getattr(layer, prefix + "_weight")
        scales = getattr(layer, prefix + "_weight_scale").view(torch.uint8)
        if prefix == "w13":
            weight = torch.cat(
                [
                    torch.nn.functional.pad(v, (0, 0, 0, 16))
                    for v in weight.chunk(2, dim=1)
                ],
                dim=1,
            )
            scales = torch.cat(
                [
                    torch.nn.functional.pad(v, (0, 0, 0, 16))
                    for v in scales.chunk(2, dim=1)
                ],
                dim=1,
            )
        else:
            weight = torch.nn.functional.pad(weight, (0, 8))
            scales = torch.nn.functional.pad(scales, (0, 1))
        copy_or_rebind_param(layer, prefix + "_weight", weight)
        copy_or_rebind_param(
            layer, prefix + "_weight_scale", scales.view(torch.float8_e4m3fn)
        )


def prepare_nvfp4_moe(original, layer, group_size=None):
    from sglang.srt.layers.utils import copy_or_rebind_param

    if layer.params_dtype != torch.float16:
        raise ValueError("The V100 NVFP4 profile requires FP16 activations")
    if hasattr(layer, "moe_runner_config") and not layer.moe_runner_config.is_gated:
        raise ValueError("The V100 NVFP4 profile requires gated MoE experts")
    if group_size is None:
        group_size = getattr(getattr(layer, "quant_config", None), "group_size", 16)
    if group_size != 16:
        raise ValueError("The V100 NVFP4 profile requires group_size=16")
    if any(getattr(layer, name, None) is not None for name in ("w13_bias", "w2_bias")):
        raise ValueError("The V100 NVFP4 profile does not support expert bias")
    pad_nvfp4_expert_width(layer)
    # Match mainline's Marlin workspace contract: four counters per SM.
    layer.workspace = torch.zeros(
        torch.cuda.get_device_properties(layer.w13_weight.device).multi_processor_count
        * 4,
        device=layer.w13_weight.device,
        dtype=torch.int32,
    )
    for prefix in ("w13", "w2"):
        weight = getattr(layer, prefix + "_weight")
        scales = getattr(layer, prefix + "_weight_scale")
        global_scales = getattr(layer, prefix + "_weight_scale_2")
        if global_scales.ndim == 2:
            if not torch.equal(global_scales[:, 0], global_scales[:, 1]):
                raise ValueError(
                    "Unequal NVFP4 gate/up global scales are unsupported on SM70"
                )
            global_scales = global_scales[:, 0]
        encoded, factor = sm70_nvfp4_marlin_process_scales(
            scales.transpose(1, 2).contiguous(), torch.float16
        )
        global_scales = (
            sm70_nvfp4_marlin_process_global_scale(global_scales, torch.float16)
            / factor
        )
        copy_or_rebind_param(layer, prefix + "_weight", _dense_repack(weight))
        copy_or_rebind_param(layer, prefix + "_weight_scale", encoded)
        copy_or_rebind_param(
            layer, prefix + "_weight_scale_2", global_scales.contiguous()
        )
        # ModelOpt allocates a separate swizzled scale buffer for other MoE
        # backends. Marlin reads only weight_scale; keep the unused attribute
        # as an alias (also following primary-Parameter reloads), not a second
        # allocation. On GLM this releases 2.21 GiB per TP4/PP2 rank.
        swizzled_name = prefix + "_blockscale_swizzled"
        if hasattr(layer, swizzled_name):
            setattr(layer, swizzled_name, getattr(layer, prefix + "_weight_scale"))


def marlin_gemm(original, *args, **kwargs):
    from .kernels.moe_marlin import moe_wna16_marlin_gemm

    return moe_wna16_marlin_gemm(*args, **kwargs)


def moe_runner(original, cls, dispatch_name, runner_name):
    fn = original(cls, dispatch_name, runner_name)
    if dispatch_name != "none" or runner_name != "marlin" or fn is None:
        return fn

    def run(dispatch, quant, config):
        h = dispatch.hidden_states
        topk = dispatch.topk_output
        from sglang.srt.environ import envs

        if (
            envs.SGLANG_OPT_SM70_NVFP4_MOE_GEMV.get()
            and h.is_cuda
            and h.dtype == torch.float16
            and h.ndim == 2
            and 1 <= h.shape[0] <= 4
            and h.shape[1] == 4096
            and tuple(topk.topk_ids.shape) == (h.shape[0], 8)
            and topk.topk_ids.dtype == torch.int32
            and topk.topk_weights.dtype == torch.float32
            and tuple(quant.w2_qweight.shape) in ((288, 16, 8192), (288, 32, 8192))
            and tuple(quant.w13_qweight.shape)
            == (288, 256, quant.w2_qweight.shape[1] * 64)
            and quant.weight_bits == 4
            and quant.w13_scales.dtype == quant.w2_scales.dtype == torch.float8_e4m3fn
            and quant.w13_qzeros is None
            and quant.w2_qzeros is None
            and quant.w13_global_scale is not None
            and quant.w2_global_scale is not None
            and quant.expert_map is None
            and config.num_experts == config.num_local_experts == 288
            and config.activation == "silu"
            and config.is_gated
            and not config.apply_router_weight_on_input
            and not config.no_combine
            and config.swiglu_limit == 10.0
            and config.routed_scaling_factor == 2.5
            and config.gemm1_clamp_limit is None
            and config.gemm1_alpha is None
            and config.gemm1_beta is None
            and quant.w13_bias is None
            and quant.w2_bias is None
            and torch.cuda.get_device_capability(h.device) == (7, 0)
        ):
            from sglang.kernels.ops.moe.sm70_nvfp4 import sm70_nvfp4_moe
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardCombineInput,
            )

            output = sm70_nvfp4_moe(
                h,
                topk.topk_ids,
                topk.topk_weights,
                quant.w13_qweight,
                quant.w2_qweight,
                quant.w13_scales,
                quant.w2_scales,
                quant.w13_global_scale,
                quant.w2_global_scale,
                config.swiglu_limit,
                config.routed_scaling_factor,
            )
            return StandardCombineInput(hidden_states=output)
        if (
            h.dtype == torch.float16
            and quant.weight_bits == 8
            and getattr(quant.w13_scales, "_sm70_fp8_scale", False)
            and getattr(quant.w2_scales, "_sm70_fp8_scale", False)
        ):
            # Explicit SM70 W8A16 Marlin backend, including mapped EP rows.
            return fn(dispatch, quant, config)
        supported = (
            h.dtype == torch.float16
            and h.ndim == 2
            and 1 <= h.shape[0] <= 4
            and h.shape[1] == 2560
            and tuple(topk.topk_ids.shape) == (h.shape[0], 10)
            and tuple(quant.w2_qweight.shape) in ((512, 6, 5120), (512, 10, 5120))
            and tuple(quant.w13_qweight.shape)
            == (512, 160, quant.w2_qweight.shape[1] * 64)
            and quant.w13_scales.dtype == quant.w2_scales.dtype == torch.float8_e4m3fn
            and quant.weight_bits == 4
            and quant.w13_qzeros is None
            and quant.w2_qzeros is None
            and quant.w13_global_scale is not None
            and quant.w2_global_scale is not None
            and quant.expert_map is None
            and config.num_experts == config.num_local_experts == 512
            and config.activation == "silu"
            and config.is_gated
            and not config.apply_router_weight_on_input
            and config.swiglu_limit is None
            and config.gemm1_clamp_limit is None
            and config.gemm1_alpha is None
            and config.gemm1_beta is None
            and config.routed_scaling_factor in (None, 1.0)
        )
        if supported:
            from .kernels.sm70_nvfp4_moe_decode import (
                sm70_nvfp4_moe_decode,
                sm70_nvfp4_moe_decode_enabled,
            )

            if sm70_nvfp4_moe_decode_enabled():
                from sglang.srt.layers.moe.token_dispatcher.standard import (
                    StandardCombineInput,
                )

                output = sm70_nvfp4_moe_decode(
                    h,
                    quant.w13_qweight,
                    quant.w2_qweight,
                    quant.w13_scales,
                    quant.w2_scales,
                    quant.w13_global_scale,
                    quant.w2_global_scale,
                    topk.topk_ids.view(-1),
                    topk.topk_weights.view(-1),
                )
                return StandardCombineInput(hidden_states=output)
        if (
            h.is_cuda
            and h.dtype == torch.float16
            and h.ndim == 2
            and (h.shape[0] > 4 or h.shape[1] != 2560)
            and quant.weight_bits == 4
            and quant.w13_scales.dtype == quant.w2_scales.dtype == torch.float8_e4m3fn
            and torch.cuda.get_device_capability(h.device) == (7, 0)
        ):
            # GLM and prefill use the explicitly selected SM70 Marlin backend.
            # Its GEMM hook is replaced by marlin_gemm; an unavailable native
            # extension raises instead of choosing a stock newer-GPU kernel.
            return fn(dispatch, quant, config)
        reject_fallback(
            "moe.nvfp4_decode",
            "native Qwen MoE requires SGLANG_V100_NVFP4_MOE_DECODE=1, "
            "FP16 [1..4, 2560], 512 NVFP4 gated experts with intermediate "
            "size 96 (padded80) or160, topk=10 and default activation/routing options",
            input=h,
            route_ids=getattr(topk, "topk_ids", None),
            w13=getattr(quant, "w13_qweight", None),
            w2=getattr(quant, "w2_qweight", None),
        )
        return fn(dispatch, quant, config)

    return run


def sm70_nvfp4_marlin_process_scales(
    scales: torch.Tensor, activation_dtype: torch.dtype
) -> tuple[torch.Tensor, float]:
    """Encode logical E4M3 NVFP4 scales for the SM70 Marlin fast path.

    The V100 iterator converts four metadata bytes to FP16/BF16 with integer
    bit operations.  It therefore consumes the special non-negative S0E5M3
    representation used by marlin_v100, not checkpoint-native S1E4M3 bytes.
    ``scales`` must already be in logical ``[E, K/16, N]`` order.

    Returns the encoded metadata and a power-of-two scale factor.  The caller
    must divide the processed global scale by that factor.
    """
    if activation_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"SM70 NVFP4 Marlin supports FP16/BF16 activations, got {activation_dtype}"
        )
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 3:
        raise ValueError(
            "SM70 NVFP4 Marlin expects [E,K/16,N] E4M3 scales, got "
            f"shape={tuple(scales.shape)}, dtype={scales.dtype}"
        )

    logical = scales.to(torch.float16)
    if bool((logical < 0).any()):
        raise ValueError("NVFP4 block scales must be non-negative")

    # FP16 has enough exponent range for ModelOpt's normalized E4M3 scales.
    # BF16 may need a shared rescale so every non-zero S0E5M3 byte keeps its
    # high bit set, which the integer dequantizer relies on.
    scale_factor = 1.0
    if activation_dtype == torch.bfloat16:
        nonzero = logical[logical > 0]
        if nonzero.numel() > 0:
            min_scaled = nonzero.float().min() * (2**7)
            if min_scaled < 2:
                scale_factor = float(torch.ceil(torch.log2(2 / min_scaled)).exp2())
                logical = (logical.float() * scale_factor).to(torch.float16)

    num_experts, num_groups, size_n = logical.shape
    if size_n % 4 != 0:
        raise ValueError(f"SM70 NVFP4 Marlin requires N divisible by 4, got N={size_n}")
    flat = logical.reshape(-1, size_n)
    # The iterator deliberately reverses each pair when expanding metadata.
    flat = flat.view(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, size_n)
    encoded = (flat * (2**7)).view(torch.int16) << 1
    encoded = encoded.view(torch.float8_e4m3fn).reshape(-1, size_n * 2)
    encoded = encoded[:, 1::2].contiguous()
    return encoded.reshape(num_experts, num_groups, size_n), scale_factor


def sm70_nvfp4_marlin_process_global_scale(
    global_scale: torch.Tensor, activation_dtype: torch.dtype
) -> torch.Tensor:
    """Compensate an NVFP4 global scale for SM70's integer dequantizer."""
    if activation_dtype == torch.float16:
        target_exponent = 5
    elif activation_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise ValueError(
            f"SM70 NVFP4 Marlin supports FP16/BF16 activations, got {activation_dtype}"
        )
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return (global_scale.to(torch.float32) * (2.0 ** (exponent_bias - 7))).contiguous()
