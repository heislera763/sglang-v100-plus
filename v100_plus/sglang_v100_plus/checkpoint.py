"""CPU checkpoint decoding; native weights and calibration remain unchanged.

CSF32 format: FlashInfer docs/csf-scale-blocks.md at 35c0f234c4b8.
Decode serialized scale bytes before the ordinary TP sharding/Marlin loader.
This does not reconstruct FP16 expert weights or add work during inference.
"""

from functools import lru_cache

import numpy as np
import torch

_CSF_FIELDS = {
    "bases": torch.uint8,
    "codes": torch.uint8,
    "page_bases": torch.uint32,
    "offsets": torch.uint16,
    "exceptions": torch.uint16,
}


@lru_cache(maxsize=16)
def _csf_geometry(rows, columns):
    pairs = (columns + 1) // 2
    rr, cc = np.indices((rows // 32, pairs))
    groups = (rr // 4) * ((pairs + 1) // 2) + cc // 2
    order = np.argsort((groups * 8 + cc % 2 * 4 + rr % 4).ravel())
    pages = groups.ravel()[order] // 32
    widths = np.minimum(2, columns - (order % pairs) * 2)
    local = np.arange(64)
    positions = (
        ((order // pairs)[:, None] * 32 + local // 2) * columns
        + (order % pairs)[:, None] * 2
        + local % 2
    )
    positions = positions[local % 2 < widths[:, None]]
    return order, pages, widths, positions


def decode_csf32_scales(components):
    """Decode NVFP4 csf_32x2_v1 into exact row-major E4M3 scale bytes."""
    if components.keys() != _CSF_FIELDS.keys():
        raise ValueError("CSF32 requires all five scale components")
    arrays = {}
    for name, dtype in _CSF_FIELDS.items():
        value = components[name]
        if (
            value.device.type != "cpu"
            or value.dtype != dtype
            or value.ndim != 1
            or not value.is_contiguous()
        ):
            raise ValueError(f"Invalid CSF32 {name} dtype, shape or device")
        arrays[name] = value.numpy()
    bases, codes = arrays["bases"], arrays["codes"]
    rows = len(bases)
    if not rows or rows % 32 or not len(codes) or 2 * len(codes) % rows:
        raise ValueError("CSF32 requires complete 32-row blocks and scale columns")
    columns = 2 * len(codes) // rows
    if np.any(bases > 240):
        raise ValueError("CSF32 row palette exceeds the scale byte range")
    order, pages, widths, positions = _csf_geometry(rows, columns)
    page_bases, offsets = arrays["page_bases"], arrays["offsets"]
    exceptions = arrays["exceptions"]
    page_count = int(pages[-1]) + 1
    blocks = len(order)
    if len(page_bases) != page_count or len(offsets) != blocks + page_count:
        raise ValueError("Invalid CSF32 page or prefix extent")
    indexes = np.arange(blocks) + pages
    starts = page_bases[pages].astype(np.int64) // 2 + offsets[indexes]
    ends = page_bases[pages].astype(np.int64) // 2 + offsets[indexes + 1]
    counts = ends - starts
    first = np.r_[True, pages[1:] != pages[:-1]]
    if (
        np.any(page_bases % 2)
        or np.any(offsets[indexes[first]] != 0)
        or starts[0] != 0
        or ends[-1] != len(exceptions)
        or np.any(starts[1:] != ends[:-1])
        or np.any(counts < 0)
        or np.any(counts > 32 * widths)
    ):
        raise ValueError("CSF32 exceptions must cover one consecutive bounded arena")
    record_blocks = np.repeat(np.arange(blocks), counts)
    local = exceptions & 255
    if (
        np.any(local >= 64)
        or np.any(local % 2 >= widths[record_blocks])
        or np.any((record_blocks[1:] == record_blocks[:-1]) & (local[1:] <= local[:-1]))
    ):
        raise ValueError(
            "CSF32 exception positions must be sorted, unique and in bounds"
        )
    fields = np.empty(2 * len(codes), dtype=np.uint8)
    fields[::2], fields[1::2] = codes & 15, codes >> 4
    result = np.empty(rows * columns, dtype=np.uint8)
    result[positions] = bases[positions // columns] + fields
    logical_blocks = order[record_blocks]
    pairs = (columns + 1) // 2
    replacements = (
        (logical_blocks // pairs * 32 + local // 2) * columns
        + logical_blocks % pairs * 2
        + local % 2
    )
    result[replacements] = (exceptions >> 8).astype(np.uint8)
    return torch.from_numpy(result.reshape(rows, columns)).view(torch.float8_e4m3fn)


def _checkpoint_prefix(name):
    return name.replace("model.language_model.", "model.").replace(
        "language_model.model.", "model."
    )


def normalize_block_fp8_weights(weights, prefixes):
    """ModelOpt serializes block scales as [N/128,1,K/128,1] or [N/128,K/128]."""
    prefixes = {_checkpoint_prefix(p) for p in prefixes}
    for name, value in weights:
        prefix, _, field = name.rpartition(".")
        if _checkpoint_prefix(prefix) in prefixes and field in {
            "weight_scale",
            "weight_scale_inv",
        }:
            if value.ndim == 4 and value.shape[1] == value.shape[3] == 1:
                value = value[:, 0, :, 0]
            if value.ndim != 2:
                raise ValueError(f"Invalid ModelOpt block-FP8 scale shape: {name}")
        yield name, value


def expand_mxfp8_fixed_weights(weights, prefix_dtypes):
    """Reconstruct fixed-precision MLA/indexer projections before model loading.

    Ordinary MXFP8 linears remain packed. MLA requires FP16 w_kc/w_vc operands;
    the indexer's weighting projection deliberately executes in FP32. Preserve
    those boundaries without tensorwise requantization or BF16 intermediates.
    """
    prefixes = {_checkpoint_prefix(p): dtype for p, dtype in prefix_dtypes.items()}
    pending, completed = {}, set()
    pending_bytes = 0
    for name, value in weights:
        prefix, _, field = name.rpartition(".")
        prefix = _checkpoint_prefix(prefix)
        if prefix not in prefixes or field not in {
            "weight",
            "weight_scale",
            "weight_scale_inv",
        }:
            yield name, value
            continue
        field = "weight" if field == "weight" else "scale"
        plane = pending.setdefault(prefix, {})
        if prefix in completed or field in plane:
            raise ValueError(f"Duplicate MXFP8 MLA component: {name}")
        plane[field] = (name, value)
        pending_bytes += value.numel() * value.element_size()
        if pending_bytes > 64 << 20:
            raise ValueError("Incomplete MXFP8 MLA stream exceeds staging capacity")
        if len(plane) == 2:
            weight_name, weight = plane["weight"]
            scales = plane["scale"][1]
            if (
                weight.device.type != "cpu"
                or scales.device.type != "cpu"
                or weight.dtype != torch.float8_e4m3fn
                or scales.dtype != torch.uint8
                or weight.ndim != 2
                or weight.shape[1] % 32
                or scales.shape != (weight.shape[0], weight.shape[1] // 32)
                or bool((scales == 255).any())
            ):
                raise ValueError("Invalid serialized MXFP8 MLA operands")
            decoded = torch.ldexp(
                weight.float(), (scales.int() - 127).repeat_interleave(32, dim=1)
            ).to(prefixes[prefix])
            if not bool(torch.isfinite(decoded).all()):
                raise ValueError("MXFP8 fixed-precision operands exceed dtype range")
            pending_bytes -= sum(
                v.numel() * v.element_size() for _, v in plane.values()
            )
            del pending[prefix]
            completed.add(prefix)
            yield weight_name, decoded
    if pending:
        raise ValueError(f"Incomplete MXFP8 MLA components: {next(iter(pending))}")


def expand_csf32_weights(weights, encoded_prefixes, *, max_pending_bytes=256 << 20):
    """Stream complete scale planes; reject missing/duplicate/malformed components.

    Components may cross shards. Bound incomplete payloads instead of retaining
    a model's worth of scale data. Complete planes enter the existing loader
    immediately, before TP slicing, gate/up fusion and scale transformations.
    """
    encoded_prefixes = {_checkpoint_prefix(p) for p in encoded_prefixes}
    pending, completed = {}, set()
    pending_bytes = 0
    for name, value in weights:
        if ".weight_scale.csf32_" not in name:
            yield name, value
            continue
        scale_name, field = name.rsplit(".csf32_", 1)
        prefix = _checkpoint_prefix(scale_name.removesuffix(".weight_scale"))
        while prefix not in encoded_prefixes and "." in prefix:
            prefix = prefix.rsplit(".", 1)[0]
        if prefix not in encoded_prefixes or field not in _CSF_FIELDS:
            raise ValueError(f"Undeclared or unsupported CSF32 scale: {name}")
        plane = pending.setdefault(scale_name, {})
        if scale_name in completed or field in plane:
            raise ValueError(f"Duplicate CSF32 scale component: {name}")
        plane[field] = value
        pending_bytes += value.numel() * value.element_size()
        if pending_bytes > max_pending_bytes:
            raise ValueError("Incomplete CSF32 scale stream exceeds staging capacity")
        if len(plane) == len(_CSF_FIELDS):
            decoded = decode_csf32_scales(plane)
            pending_bytes -= sum(t.numel() * t.element_size() for t in plane.values())
            del pending[scale_name]
            completed.add(scale_name)
            yield scale_name, decoded
    if pending:
        raise ValueError(f"Incomplete CSF32 scale components: {next(iter(pending))}")


def load_glm_csf_weights(original, model, weights, is_nextn=False):
    config = model.quant_config
    layers = getattr(config, "quantized_layers", {})
    encoded = {}
    for prefix, recipe in layers.items():
        encoding = recipe.get("weight_scale_encoding")
        if encoding is None:
            continue
        if encoding != "csf_32x2_v1" or recipe.get("quant_algo", "").upper() not in {
            "NVFP4",
            "W4A16_NVFP4",
        }:
            raise ValueError(
                f"Unsupported V100 scale encoding for {prefix}: {encoding}"
            )
        encoded[prefix] = recipe
    if encoded and is_nextn:
        # The ordinary loader discards target layers for the MTP runner.
        # Filter before decoding, not after expanding every target scale twice.
        layer = (
            0 if model.config.num_hidden_layers == 1 else model.config.num_hidden_layers
        )
        layer_prefix = f"model.layers.{layer}."
        weights = (
            (n, v) for n, v in weights if _checkpoint_prefix(n).startswith(layer_prefix)
        )
    elif encoded and getattr(model, "encoder_only", False):
        weights = ((n, v) for n, v in weights if "visual." in n)
    fixed = {}
    for prefix, recipe in layers.items():
        if recipe.get("quant_algo", "").upper() != "MXFP8":
            continue
        if prefix.endswith(".kv_b_proj"):
            fixed[prefix] = torch.float16
        elif prefix.endswith(".indexer.weights_proj"):
            fixed[prefix] = torch.float32
    if is_nextn:
        # Quantization metadata is already mapped to model.decoder, whereas
        # the weight iterator still carries the checkpoint's layer number.
        layer = (
            0 if model.config.num_hidden_layers == 1 else model.config.num_hidden_layers
        )
        fixed = {
            p.replace("model.decoder.", f"model.layers.{layer}."): dtype
            for p, dtype in fixed.items()
        }
    blocks = [
        prefix
        for prefix, recipe in layers.items()
        if recipe.get("quant_algo", "").upper() in {"FP8_PB_WO", "FP8_BLOCK_SCALES"}
    ]
    weights = normalize_block_fp8_weights(weights, blocks)
    weights = expand_mxfp8_fixed_weights(weights, fixed)
    return original(model, expand_csf32_weights(weights, encoded), is_nextn=is_nextn)
