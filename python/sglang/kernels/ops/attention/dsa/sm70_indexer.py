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


@triton.jit
def _neox_indexer_row(
    X,
    Pos,
    Freq,
    Gamma,
    Beta,
    Gate,
    Loc,
    Out,
    ScaleOut,
    xs0,
    xs1,
    gs0,
    cache_stride,
    freq_stride,
    eps,
    weight_scale,
    KEY: tl.constexpr,
    PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    x = tl.load(X + row * xs0 + head * xs1 + d).to(tl.float32)
    if KEY:
        x -= tl.sum(x, 0) / 128.0
        x *= tl.rsqrt(tl.sum(x * x, 0) / 128.0 + eps)
        x = x * tl.load(Gamma + d).to(tl.float32) + tl.load(Beta + d).to(tl.float32)
    f = Freq + tl.load(Pos + row).to(tl.int64) * freq_stride
    cos = tl.load(f + (d & 31)).to(tl.float32)
    sin = tl.load(f + 32 + (d & 31)).to(tl.float32)
    peer = tl.gather(x, d ^ 32, 0)
    x = tl.where(d < 64, x * cos + tl.where(d < 32, -peer, peer) * sin, x)
    for i in tl.static_range(7):
        other = tl.gather(x, d ^ (1 << i), 0)
        x = tl.where((d & (1 << i)) == 0, x + other, other - x)
    x *= 0.08838834764831845
    if KEY:
        x = x.to(tl.float16).to(tl.float32)
    encoded, scale = _quantize_row(x, False)
    if KEY:
        slot = tl.load(Loc + row).to(tl.int64)
        if slot >= 0:
            page = Out + slot // PAGE * cache_stride
            off = slot % PAGE
            tl.store(page + off * 128 + d, encoded)
            tl.store((page + PAGE * 128).to(tl.pointer_type(tl.float32)) + off, scale)
    else:
        index = row * tl.num_programs(1) + head
        tl.store(Out + index * 128 + d, encoded)
        tl.store(
            ScaleOut + index,
            tl.load(Gate + row * gs0 + head).to(tl.float32) * weight_scale * scale,
        )


def store_neox_index_key(key, positions, freqs, norm, locations, cache, page_size):
    _neox_indexer_row[(key.shape[0], 1)](
        key,
        positions,
        freqs,
        norm.weight,
        norm.bias,
        key,
        locations,
        cache,
        key,
        key.stride(0),
        0,
        0,
        cache.stride(0),
        freqs.stride(0),
        norm.variance_epsilon,
        1.0,
        True,
        page_size,
    )


def quant_neox_index_query(q, weights, positions, freqs, weight_scale):
    rows, heads, dim = q.shape
    assert dim == 128 and q.stride(-1) == weights.stride(-1) == 1
    quant = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
    scales = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    _neox_indexer_row[(rows, heads)](
        q,
        positions,
        freqs,
        q,
        q,
        weights,
        positions,
        quant,
        scales,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        0,
        freqs.stride(0),
        0.0,
        weight_scale,
        False,
        0,
    )
    return quant.view(torch.float8_e4m3fn), scales.unsqueeze(-1)


@triton.jit
def _decode_e4m3fn(encoded):
    bits = encoded.to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    value = tl.where(
        exponent == 0,
        mantissa.to(tl.float32) * 2**-9,
        normal.to(tl.float32, bitcast=True),
    )
    value = tl.where((bits & 127) == 127, float("nan"), value)
    # Multiplication preserves negative zero for the FP8 sign bit.
    return value * tl.where((bits & 128) != 0, -1.0, 1.0)


@triton.jit
def _unpack_shared_index_pages(
    K,
    Table,
    Lens,
    Out,
    Scales,
    width,
    page_stride,
    ROWS: tl.constexpr,
    BR: tl.constexpr,
    PAGE: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.arange(0, BR)
    length = tl.minimum(width, tl.max(tl.load(Lens + row, row < ROWS, 0), 0))
    d = tl.arange(0, 128)
    for tile in range(tl.program_id(0), tl.cdiv(length, BT), tl.num_programs(0)):
        pos = tile * BT + tl.arange(0, BT)
        page = tl.load(Table + pos // PAGE, pos < length, 0).to(tl.int64)
        base = K + page * page_stride
        valid = (pos < length) & (page >= 0)
        key = tl.load(
            base[:, None] + (pos % PAGE)[:, None] * 128 + d[None, :], valid[:, None], 0
        )
        scale = tl.load(
            (base + PAGE * 128).to(tl.pointer_type(tl.float32)) + pos % PAGE, valid, 0
        )
        tl.store(
            Out + pos[:, None].to(tl.int64) * 128 + d[None, :],
            _decode_e4m3fn(key),
            (pos < length)[:, None],
        )
        tl.store(Scales + pos, scale, pos < length)


def shared_paged_scores_sm70(q, cache, weights, lengths, table, width):
    """A single request's verification rows share one FP16 scratch key bank."""
    from sglang.kernels.ops.attention.dsa.sm70_indexer_score import mqa_logits_sm70

    rows, heads, dim = q.shape
    lengths = lengths.reshape(rows).contiguous()
    keys = torch.empty((width, dim), dtype=torch.float16, device=q.device)
    scales = torch.empty(width, dtype=torch.float32, device=q.device)
    if rows and width:
        _unpack_shared_index_pages[(min(256, triton.cdiv(width, 16)),)](
            cache.view(torch.uint8),
            table,
            lengths,
            keys,
            scales,
            width,
            cache.stride(0),
            rows,
            triton.next_power_of_2(rows),
            cache.shape[1],
            16,
        )
    return mqa_logits_sm70(q, keys, scales, weights.reshape(rows, heads), lengths)


@triton.jit
def _selected_scores(
    Q,
    K,
    S,
    W,
    Start,
    End,
    Table,
    Out,
    q_row,
    q_head,
    k_row,
    w_row,
    table_row,
    queries_per_table: tl.constexpr,
    out_row,
    width,
    H: tl.constexpr,
    PAGE: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(1)
    col = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.arange(0, 128)
    begin = 0
    if not PAGE:
        begin = tl.load(Start + row)
    end = tl.minimum(tl.load(End + row), width)
    score = tl.full((BT,), 0.0, tl.float32)
    if tl.program_id(0) * BT < end and (tl.program_id(0) + 1) * BT > begin:
        valid = (col >= begin) & (col < end)
        if PAGE:
            page = tl.load(
                Table + row // queries_per_table * table_row + col // PAGE, valid, 0
            ).to(tl.int64)
            base = K + page * k_row
            key = tl.load(
                base[:, None] + (col % PAGE)[:, None] * 128 + d[None, :],
                valid[:, None],
                0,
            )
            scale = tl.load(
                (base + PAGE * 128).to(tl.pointer_type(tl.float32)) + col % PAGE,
                valid,
                0,
            )
        else:
            key = tl.load(
                K + col[:, None].to(tl.int64) * k_row + d[None, :], valid[:, None], 0
            )
            scale = tl.load(S + col, valid, 0)
        key_fp16 = _decode_e4m3fn(key).to(tl.float16)
        for head in tl.static_range(H // 16):
            h = head * 16 + tl.arange(0, 16)
            query = tl.load(Q + row * q_row + h[:, None] * q_head + d[None, :])
            gates = tl.load(W + row * w_row + h).to(tl.float32)
            dots = tl.dot(_decode_e4m3fn(query).to(tl.float16), tl.trans(key_fp16))
            score += tl.sum(tl.maximum(dots, 0.0) * gates[:, None], 0)
        score = tl.where(valid, score * scale, 0.0)
    tl.store(Out + row.to(tl.int64) * out_row + col, score, col < width)


def index_scores_sm70(q, k, scales, weights, begin, end, width, pages=None):
    q = q.reshape(-1, q.shape[-2], 128)
    rows, heads, _ = q.shape
    assert heads % 16 == 0 and q.dtype == torch.float8_e4m3fn
    assert k.dtype in (torch.uint8, torch.float8_e4m3fn)
    page_size = k.shape[1] if pages is not None else 0
    assert end.numel() == rows and weights.numel() == rows * heads
    weights = weights.reshape(rows, heads)
    if pages is not None and rows:
        assert pages.shape[0] and rows % pages.shape[0] == 0
    output = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    if rows and width:
        _selected_scores[(triton.cdiv(width, 32), rows)](
            q.view(torch.uint8),
            k.view(torch.uint8),
            scales,
            weights,
            begin,
            end,
            pages if pages is not None else end,
            output,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            weights.stride(0),
            pages.stride(0) if pages is not None else 0,
            rows // pages.shape[0] if pages is not None else 1,
            output.stride(0),
            width,
            heads,
            page_size,
            32,
        )
    return output
