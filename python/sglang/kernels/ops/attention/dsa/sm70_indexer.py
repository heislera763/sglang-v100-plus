"""Software-FP8 DSA primitives for SM70 (no native FP8 instructions).

Quantization and pool4 compression each use one graph-capturable Triton launch.
The runtime adapter still owns request metadata and requires eager execution.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _encode_e4m3fn(x):
    # Round to nearest even in each E4M3 binade, including subnormals. Clamped
    # finite inputs use exponents -6..8; byte 0x7e is the largest value (448).
    bits = x.to(tl.int32, bitcast=True)
    magnitude = tl.abs(x)
    exponent = tl.maximum(((bits >> 23) & 255) - 127, -6)
    step = ((exponent - 3 + 127) << 23).to(tl.float32, bitcast=True)
    mantissa = libdevice.nearbyint(magnitude / step).to(tl.int32)
    encoded = tl.where(
        magnitude < 0.015625, mantissa, ((exponent + 7) << 3) + mantissa - 8
    )
    encoded = tl.where(x != x, 127, encoded)
    return (encoded | ((bits >> 24) & 128)).to(tl.uint8)


@triton.jit
def _quantize_row(x, ROUND_SCALE: tl.constexpr):
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1e-4) / 448.0
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    values = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    return _encode_e4m3fn(values), scale


@triton.jit
def _fp8_quantize_kernel(X, Q, S, STRIDE_ROW: tl.constexpr, ROUND_SCALE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, 128)
    x = tl.load(X + row * STRIDE_ROW + cols).to(tl.float32)
    quantized, scale = _quantize_row(x, ROUND_SCALE)
    tl.store(Q + row * 128 + cols, quantized)
    tl.store(S + row, scale)


def fp8_quantize_sm70(x: torch.Tensor, round_scale: bool = False):
    if x.shape[-1] != 128 or not x.is_cuda:
        raise ValueError("SM70 DSA quantization requires CUDA 128-element blocks")
    # Model queries are contiguous. Preserve the original contract for callers
    # passing strided blocks, without imposing a copy on the serving path.
    if not x.is_contiguous():
        x = x.contiguous()
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    if x.numel():
        _fp8_quantize_kernel[(x.numel() // 128,)](
            x,
            quantized.view(torch.uint8),
            scale,
            128,
            round_scale,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return quantized, scale


@triton.jit
def _hadamard_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    groups = tl.trans(tl.reshape(x, (GROUPS, 2, STRIDE)), 0, 2, 1)
    a, b = tl.split(groups)
    return tl.reshape(tl.trans(tl.join(a + b, a - b), 0, 2, 1), (128,))


@triton.jit
def _kpool_compress_kernel(
    K,
    SCORES,
    APE,
    Q,
    S,
    K_ROW: tl.constexpr,
    K_SLOT: tl.constexpr,
    K_COL: tl.constexpr,
    S_ROW: tl.constexpr,
    S_SLOT: tl.constexpr,
    S_COL: tl.constexpr,
    A_SLOT: tl.constexpr,
    A_COL: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, 128)
    slots = tl.arange(0, 4)
    scores = tl.load(
        SCORES + row * S_ROW + slots[:, None] * S_SLOT + cols[None, :] * S_COL
    ).to(tl.float32)
    scores += tl.load(APE + slots[:, None] * A_SLOT + cols[None, :] * A_COL).to(
        tl.float32
    )
    keys = tl.load(
        K + row * K_ROW + slots[:, None] * K_SLOT + cols[None, :] * K_COL
    ).to(tl.float32)
    probabilities = tl.exp(scores - tl.max(scores, 0)[None, :])
    probabilities = probabilities / tl.sum(probabilities, 0)[None, :]
    x = tl.sum(probabilities * keys, 0).to(tl.bfloat16).to(tl.float32)
    # Same seven-stage normalized Walsh-Hadamard rotation and both BF16
    # rounding boundaries as the native pool kernel and Torch reference.
    x = _hadamard_stage(x, 64, 1)
    x = _hadamard_stage(x, 32, 2)
    x = _hadamard_stage(x, 16, 4)
    x = _hadamard_stage(x, 8, 8)
    x = _hadamard_stage(x, 4, 16)
    x = _hadamard_stage(x, 2, 32)
    x = _hadamard_stage(x, 1, 64)
    x = (x * 128**-0.5).to(tl.bfloat16).to(tl.float32)
    quantized, scale = _quantize_row(x, ROUND_SCALE)
    tl.store(Q + row * 128 + cols, quantized)
    tl.store(S + row, scale)


def kpool_compress_sm70(
    keys: torch.Tensor,
    scores: torch.Tensor,
    ape: torch.Tensor,
    round_scale: bool = False,
):
    if keys.ndim != 3 or keys.shape[-2:] != (4, 128):
        raise ValueError("SM70 K-pooling requires groups of four 128-element keys")
    if scores.shape != keys.shape or ape.shape != (4, 128):
        raise ValueError("K-pool scores and positional gates must match the keys")
    if not keys.is_cuda or scores.device != keys.device or ape.device != keys.device:
        raise ValueError("SM70 K-pooling requires inputs on the same CUDA device")
    quantized = torch.empty(
        (keys.shape[0], 128), device=keys.device, dtype=torch.float8_e4m3fn
    )
    scale = torch.empty((keys.shape[0], 1), device=keys.device, dtype=torch.float32)
    if keys.shape[0]:
        _kpool_compress_kernel[(keys.shape[0],)](
            keys,
            scores,
            ape,
            quantized.view(torch.uint8),
            scale,
            *keys.stride(),
            *scores.stride(),
            *ape.stride(),
            round_scale,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return quantized, scale
