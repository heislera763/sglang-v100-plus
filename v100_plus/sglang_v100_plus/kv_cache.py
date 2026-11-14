"""Declared SM70 FP16 cache-copy path for dense draft attention."""

import torch

from .dispatch import reject_fallback


def fp16_store_supported(pool, key, value, k_scale, v_scale, dcp_kv_mask):
    if not (
        key.is_cuda
        and value.device == key.device
        and key.dtype == value.dtype == torch.float16
        and key.ndim == value.ndim == 3
        and pool.dtype == torch.float16
        and pool.store_dtype == torch.float16
        and pool.kv_cache_layout == "nhd"
        and not pool.is_quantized_kv_cache
        and not pool.use_hnd
        and key.shape[0] == value.shape[0]
        and key.shape[1] * key.shape[2] == pool.row_dim
        and value.shape[1] * value.shape[2] == pool.v_row_dim
        and key.stride(-1) == value.stride(-1) == 1
        and key.stride(-2) == key.shape[-1]
        and value.stride(-2) == value.shape[-1]
        and torch.cuda.get_device_capability(key.device) == (7, 0)
    ):
        return False
    if dcp_kv_mask is not None:
        # The upstream masked writer is a direct Triton copy on SM70. Its
        # ownership mask is replay-time device data, with no indexed fallback.
        return (
            dcp_kv_mask.device == key.device
            and dcp_kv_mask.dtype == torch.bool
            and dcp_kv_mask.shape == (key.shape[0],)
            and dcp_kv_mask.is_contiguous()
        )
    from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache

    # Quantized draft projections may attach descale parameters even when
    # the cache is FP16. Upstream ignores them when source and cache dtypes
    # match; this remains an exact byte copy with those parameters present.
    # The upstream writer checks this same predicate before choosing its CUDA
    # byte-copy kernel. Resolve it here so a failed JIT build cannot silently
    # switch strict development runs to the indexed Torch fallback.
    if not can_use_store_cache(pool.row_dim * 2, pool.v_row_dim * 2):
        reject_fallback(
            "kv.fp16_store",
            "SM70 CUDA cache-copy kernel is unavailable",
            key=key,
            value=value,
        )
        return False
    return True
