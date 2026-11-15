"""Resident FP16 rank-one adapters with FP32 dot-product accumulation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _coefficients(
    X, A, Ids, Out, K: tl.constexpr, ROUTED: tl.constexpr, BK: tl.constexpr
):
    row = tl.program_id(0)
    expert = tl.load(Ids + row) if ROUTED else 0
    if expert >= 0:
        k = tl.arange(0, BK)
        x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
        a = tl.load(A + expert * K + k, k < K, 0).to(tl.float32)
        value = tl.sum(x * a, 0)
    else:
        value = 0.0
    tl.store(Out + row, value)


@triton.jit
def _add(
    Out, Alpha, B, Ids, Router, N: tl.constexpr, ROUTED: tl.constexpr, BN: tl.constexpr
):
    row = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    expert = tl.load(Ids + row) if ROUTED else 0
    if expert >= 0:
        alpha = tl.load(Alpha + row).to(tl.float32)
        b = tl.load(B + expert * N + n, n < N, 0).to(tl.float32)
        delta = (alpha * b).to(tl.float16).to(tl.float32)
        base = tl.load(Out + row * N + n, n < N, 0).to(tl.float32)
        result = (base + delta).to(tl.float16).to(tl.float32)
        if ROUTED:
            result *= tl.load(Router + row)
        tl.store(Out + row * N + n, result, n < N)


def rank_one_coefficients(x, a, ids=None):
    """Compute FP16 x @ A.T, independently for each already-expanded route."""
    if x.ndim != 2 or a.ndim != 2 or x.shape[1] != a.shape[1]:
        raise ValueError("Rank-one input and factor dimensions disagree")
    if ids is None and a.shape[0] != 1:
        raise ValueError("Dense rank-one coefficients require one factor row")
    if ids is not None and ids.numel() != x.shape[0]:
        raise ValueError("Routed coefficients require one expert ID per row")
    if ids is not None and ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expert IDs must be integers")
    tensors = (x, a) if ids is None else (x, a, ids)
    if any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Rank-one operands must be contiguous and colocated")
    if x.dtype != torch.float16 or a.dtype != torch.float16:
        raise ValueError("Rank-one operands must be FP16")
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (7, 0):
        raise ValueError("Resident rank-one adapters require SM70 CUDA")
    out = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    if x.shape[0]:
        _coefficients[(x.shape[0],)](
            x,
            a,
            ids,
            out,
            x.shape[1],
            ids is not None,
            triton.next_power_of_2(x.shape[1]),
            num_warps=4,
        )
    return out


def rank_one_add(x, output, a, b):
    """Add a dense FP16 rank-one projection to an existing FP16 linear result."""
    x = x.reshape(-1, x.shape[-1]).contiguous()
    y = output.view(x.shape[0], output.shape[-1])
    if b.shape != (1, y.shape[1]) or not y.is_contiguous() or y.dtype != torch.float16:
        raise ValueError("Invalid rank-one output or B factor")
    if b.dtype != torch.float16 or b.device != x.device or not b.is_contiguous():
        raise ValueError("B factor must be contiguous colocated FP16")
    alpha = rank_one_coefficients(x, a)
    if x.shape[0]:
        _add[(x.shape[0], triton.cdiv(y.shape[1], 256))](
            y,
            alpha,
            b,
            None,
            None,
            y.shape[1],
            False,
            256,
            num_warps=4,
        )
    return output


def routed_rank_one_add(x, output, a, b, ids, weights):
    """Add each routed down adapter, then apply its original router weight."""
    if output.ndim != 2 or output.shape[0] != x.shape[0]:
        raise ValueError("Routed output must retain route-major rows")
    if b.shape != (a.shape[0], output.shape[1]) or weights.numel() != ids.numel():
        raise ValueError("Routed factor or router-weight dimensions disagree")
    if (
        output.dtype != torch.float16
        or b.dtype != torch.float16
        or weights.dtype != torch.float32
    ):
        raise ValueError("Routed output must be FP16 with FP32 router weights")
    if any(t.device != x.device or not t.is_contiguous() for t in (output, b, weights)):
        raise ValueError("Routed operands must be contiguous and colocated")
    alpha = rank_one_coefficients(x, a, ids)
    if x.shape[0]:
        _add[(x.shape[0], triton.cdiv(output.shape[1], 256))](
            output,
            alpha,
            b,
            ids,
            weights,
            output.shape[1],
            True,
            256,
            num_warps=4,
        )
    return output
