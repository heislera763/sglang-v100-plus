"""SM70 FP16 expert GEMM substitution; keep upstream routing and epilogues."""

import torch
import triton.language as tl

from .dispatch import reject_fallback

# Tail of invoke_fused_moe_kernel's public signature. Unknown features fail
# closed for FP16 instead of silently selecting a slower/incorrect epilogue.
_OPTIONS = (
    "compute_type",
    "use_fp8_w8a8",
    "use_int8_w8a8",
    "use_int8_w8a16",
    "use_int4_w4a16",
    "per_channel_quant",
    "block_shape",
    "no_combine",
    "a_use_tma",
    "b_use_tma",
    "c_sorted",
    "filter_expert",
    "fuse_sum_all_reduce",
    "router_topk",
    "fuse_add_to_output",
    "add_output_mask",
    "mask_output",
    "lora_preserve_base",
    "fuse_swiglu",
    "a_output_slices",
)


def invoke_fp16_moe(original, *args, **kwargs):
    a, b = args[:2]
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        return original(*args, **kwargs)
    if len(args) < 15:
        raise ValueError("SM70 FP16 MoE requires the standard positional GEMM inputs")
    options = dict(zip(_OPTIONS, args[15:]))
    options.update(kwargs)
    bias, output, a_scale, b_scale, b_zp = args[2:7]
    unsupported = (
        "use_fp8_w8a8",
        "use_int8_w8a8",
        "use_int8_w8a16",
        "use_int4_w4a16",
        "per_channel_quant",
        "a_use_tma",
        "b_use_tma",
        "c_sorted",
        "fuse_sum_all_reduce",
        "fuse_add_to_output",
        "mask_output",
        "lora_preserve_base",
        "fuse_swiglu",
    )
    if (
        options.get("compute_type") != tl.float16
        or any(v is not None for v in (bias, a_scale, b_scale, b_zp))
        or any(options.get(name, False) for name in unsupported)
        or options.get("a_output_slices", 1) != 1
        or options.get("block_shape") is not None
        or options.get("add_output_mask") is not None
        or set(options) - set(_OPTIONS)
        or len(args) > 15 + len(_OPTIONS)
    ):
        reject_fallback(
            "moe.fp16",
            "unqualified FP16 expert epilogue/quantization/TMA option",
            a=a,
            b=b,
            output=output,
        )
        return original(*args, **kwargs)
    from sglang.kernels.ops.moe.sm70_fp16 import sm70_fp16_moe_gemm

    sm70_fp16_moe_gemm(
        a,
        b,
        output,
        args[9],
        args[10],
        args[11],
        args[7],
        args[14]["BLOCK_SIZE_M"],
        args[13],
        args[12],
    )
