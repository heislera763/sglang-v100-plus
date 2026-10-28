from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType

logger = logging.getLogger(__name__)

# Constants matching device::marlin_moe:: in marlin.cuh
_MAX_THREAD_N = 256


# --- SM70 (V100) marlin_v100 integration ---------------------------------------
# The stock JIT Marlin MoE kernel (csrc/gemm/marlin_moe/marlin_template.h) is an
# empty-body stub for __CUDA_ARCH__ < 800: it launches but writes nothing, so on
# V100 the routed experts silently contribute zero (the unquantized shared
# expert + residual mask the corruption). marlin_v100 (a vLLM-derived SM70 fork
# with real WMMA kernels) replaces it. Built+installed by
# scripts/setup_v100_marlin.sh; auto-detected here on SM70 with no env var.

_IS_SM70: bool = False
try:
    if torch.cuda.is_available():
        _IS_SM70 = torch.cuda.get_device_capability()[0] == 7
except Exception:
    pass

# Cache for the loaded op. States: None = not attempted yet; callable = loaded;
# False = attempted but unavailable (so we only warn once).
_marlin_v100_op = None

_SM70_MARLIN_MOE_ENV_NAMES = (
    "SM70_MARLIN_MOE_CTA_GEOMETRY",
    "SM70_MARLIN_MOE_SPLIT_K",
    "SM70_MARLIN_MOE_METADATA_CACHE",
)
_sm70_marlin_user_tuning = any(os.getenv(name) for name in _SM70_MARLIN_MOE_ENV_NAMES)
_sm70_nvfp4_tuning_stage = None


def _configure_sm70_nvfp4_stage(
    b_scales: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> None:
    """Select measured SM70 NVFP4 geometries before CUDA-graph capture.

    marlin_v100 reads these variables synchronously when its host launcher is
    called.  SGLang captures the resulting kernels in the decode CUDA graph,
    so there is no environment handling on graph replay. The routed prefill
    gate/up and down shapes use measured CTA geometries at TP4 and TP8; other
    shapes clear the override and retain marlin_v100's generic/model selectors.
    """

    global _sm70_nvfp4_tuning_stage
    if not _IS_SM70 or _sm70_marlin_user_tuning:
        return

    stage = None
    values = None
    if (
        b_scales.dtype == torch.float8_e4m3fn
        and moe_block_size == 8
        and top_k == 10
        and size_m == 1
        and size_n == 320
        and size_k == 2560
    ):
        stage = 1
        values = ("32x64x64x4x32x64x16", "1", "vector_words")
    elif (
        b_scales.dtype == torch.float8_e4m3fn
        and moe_block_size == 32
        and top_k == 10
        and 512 <= size_m <= 2048
        and size_n == 320
        and size_k == 2560
    ):
        stage = 3
        values = ("32x64x64x4x32x32x32", "1", "vector_words")
    elif (
        b_scales.dtype == torch.float8_e4m3fn
        and moe_block_size == 64
        and top_k == 1
        and size_m >= 40960
        and size_n == 2560
        and size_k == 160
    ):
        stage = 4
        values = ("64x256x32x4x64x64x32", "1", "vector_words")
    elif (
        b_scales.dtype == torch.float8_e4m3fn
        and moe_block_size == 8
        and top_k == 1
        and size_m == 10
        and size_n == 2560
        and size_k == 160
    ):
        stage = 2
        values = ("64x256x32x4x64x64x32", "1", "lane_vectors")

    # GLM's small decode matrices otherwise use the generic 256x32 N/K
    # tile, leaving too little work in flight. Keep split-K at one: the
    # extension's split path accumulates partials with FP16 atomics.
    elif b_scales.dtype == torch.float8_e4m3fn and (
        (
            size_m == 1
            and top_k == 1
            and (size_n, size_k)
            in ((512, 4096), (3072, 4096), (4096, 1024), (4096, 256))
        )
        or (
            moe_block_size == 8
            and size_m == 1
            and top_k == 8
            and (size_n, size_k) == (512, 4096)
        )
    ):
        stage = 5
        values = ("32x64x64x4x32x64x16", "1", "vector_words")

    elif (
        b_scales.dtype == torch.float8_e4m3fn
        and moe_block_size == 64
        and (
            (
                top_k == 8
                and 128 <= size_m <= 2048
                and (size_n, size_k) in ((512, 4096), (1024, 4096))
            )
            or (
                top_k == 1
                and 1024 <= size_m <= 16384
                and (size_n, size_k) in ((4096, 256), (4096, 512))
            )
        )
    ):
        # A 32-row CTA reloads decoded weights for each half of the aligned
        # expert group. Match all 64 rows while retaining the 32-deep K tile
        # and split-K one: captured prefill outputs stay bit-exact.
        stage = 6
        values = ("64x256x32x4x64x64x32", "1", "vector_words")

    if stage == _sm70_nvfp4_tuning_stage:
        return
    if values is None:
        for name in _SM70_MARLIN_MOE_ENV_NAMES:
            os.environ.pop(name, None)
    else:
        for name, value in zip(_SM70_MARLIN_MOE_ENV_NAMES, values):
            os.environ[name] = value
    _sm70_nvfp4_tuning_stage = stage


def _load_marlin_v100_op():
    """Lazily auto-detect and load the marlin_v100 MoE op on SM70.

    Returns the registered ``torch.ops._moe_C.moe_wna16_marlin_gemm`` callable,
    or ``None`` if it could not be found. Subsequent calls return the cached
    result. No-op (returns None) on non-SM70 devices.
    """
    global _marlin_v100_op
    if _marlin_v100_op is not False and _marlin_v100_op is not None:
        return _marlin_v100_op
    if _marlin_v100_op is False:
        raise RuntimeError("SM70 Marlin extension failed to load")
    _marlin_v100_op = False  # mark attempted

    if not _IS_SM70:
        return None

    directory = os.environ["SGLANG_V100_MARLIN_DIR"]
    candidates = sorted(glob.glob(os.path.join(directory, "_moe_C*.so")))

    for path in candidates:
        try:
            torch.ops.load_library(path)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "SM70 (V100): failed to load marlin_v100 .so at %s: %s", path, e
            )
            continue
        op = getattr(torch.ops._moe_C, "moe_wna16_marlin_gemm", None)
        if op is not None:
            _marlin_v100_op = op
            logger.info("SM70 (V100): using marlin_v100 MoE kernel from %s", path)
            return op

    raise RuntimeError(f"SM70 Marlin extension missing or unusable: {candidates}")


def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
    expert_ids_are_sorted: bool = False,
    zero_inactive_output: bool = False,
    has_inactive_prefix_count: bool = False,
) -> torch.Tensor:
    device = a.device

    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.zeros((size_m * top_k, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    if has_inactive_prefix_count and num_tokens_post_padded.numel() != 2:
        raise ValueError("Marlin inactive-prefix metadata requires two count words")
    route_counts = num_tokens_post_padded
    num_tokens_post_padded = route_counts[:1]

    # SM70 (V100): dispatch to the marlin_v100 kernel when available. Its op
    # signature differs from the JIT module: it takes `a_scales` (None for
    # W4A16) and tuning ints (thread_k/n, blocks_per_sm; -1 = auto-select), and
    # it computes has_act_order/has_bias/has_zp/num_groups/group_size/is_ep
    # internally from the tensor shapes, so those are not forwarded. The raw
    # Optional tensors are passed through unchanged (None => std::nullopt) so
    # the kernel's has_value()-based presence checks fire correctly; do NOT
    # convert None to an empty tensor (the kernel treats a present-but-empty
    # global_scale as an nvfp4-only input and rejects it for GPTQ/AWQ).
    if _IS_SM70:
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.utils import get_scalar_types
        from sglang.srt.runtime_context import get_schedule

        # Batch-one GLM projections at TP8 and TP4. TP4 doubles the sharded
        # output width or reduction depth; the packed-layout kernel is shared.
        dense_gemv = (
            size_m == top_k == 1
            and b_q_weight.shape[0] == 1
            and (size_n, size_k)
            in (
                (512, 4096),
                (3072, 4096),
                (6144, 4096),
                (4096, 1024),
                (4096, 256),
                (4096, 512),
            )
        )
        routed_gemv = (
            b_q_weight.shape[0] == 288
            and moe_block_size == 8
            and (
                (size_m, top_k, size_n, size_k)
                in (
                    (1, 8, 512, 4096),
                    (1, 8, 1024, 4096),
                    (8, 1, 4096, 256),
                    (8, 1, 4096, 512),
                )
            )
            and sorted_token_ids.numel() == 64
            and expert_ids.numel() == 8
        )
        if (
            envs.SGLANG_OPT_SM70_NVFP4_GEMV.get()
            and get_schedule().disable_overlap_schedule
            and (dense_gemv or routed_gemv)
            and a.dtype == torch.float16
            and b_scales.dtype == torch.float8_e4m3fn
            and global_scale_or_none is not None
            and b_q_type.id == get_scalar_types()[1].float4_e2m1f.id
            and all(
                x is None
                for x in (b_bias_or_none, b_zeros_or_none, g_idx_or_none, perm_or_none)
            )
            and not is_ep
            and is_k_full
            and not is_zp_float
        ):
            from sglang.kernels.ops.gemm.sm70_nvfp4_gemv import sm70_nvfp4_gemv
            from sglang.srt.runtime_context import get_buffer

            # Scratch is shared by shapes, not layers; graph replay uses the
            # same stream and each GEMV consumes its partials before reuse.
            routes = expert_ids.numel()
            bn = 16 if dense_gemv and (size_n, size_k) == (4096, 256) else 32
            bk = 128
            partials = get_buffer(
                f"v100_nvfp4_gemv:{a.device}:{routes}:{size_k}:{size_n}",
                lambda: torch.empty(
                    (routes, size_k // bk, size_n), device=a.device, dtype=torch.float32
                ),
            )
            return sm70_nvfp4_gemv(
                a,
                c,
                b_q_weight,
                b_scales,
                global_scale_or_none,
                sorted_token_ids,
                expert_ids,
                topk_weights,
                moe_block_size,
                top_k,
                mul_topk_weights,
                partials,
                bn,
                bk,
                padded_count=num_tokens_post_padded,
                zero_inactive_output=zero_inactive_output,
            )
        if (
            not _sm70_marlin_user_tuning
            and moe_block_size == 8
            and (
                (
                    1 <= size_m <= 4
                    and top_k == 8
                    and (size_n, size_k) in ((512, 4096), (1024, 4096))
                )
                or (
                    size_m in (8, 16, 24, 32)
                    and top_k == 1
                    and (size_n, size_k) in ((4096, 256), (4096, 512))
                )
            )
            and a.dtype == torch.float16
            and b_scales.dtype == torch.float8_e4m3fn
            and global_scale_or_none is not None
            and b_q_type.id == get_scalar_types()[1].float4_e2m1f.id
            and all(
                x is None
                for x in (b_bias_or_none, b_zeros_or_none, g_idx_or_none, perm_or_none)
            )
            and not is_ep
            and is_k_full
            and not is_zp_float
        ):
            from sglang.kernels.ops.gemm.sm70_nvfp4_routed import (
                sm70_nvfp4_routed_gemm,
            )

            # TP4/TP8 small expert groups share M16 and K-major decoded weights;
            # keep the existing activation, expert sum and Half boundaries.
            # use_atomic_add permits split-K, not accumulation into prior C;
            # this implementation owns the full K reduction and overwrites C.
            return sm70_nvfp4_routed_gemm(
                a,
                c,
                b_q_weight,
                b_scales,
                global_scale_or_none,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                top_k,
                mul_topk_weights,
                zero_inactive_output=zero_inactive_output,
            )
        op = _load_marlin_v100_op()
        if op is not None:
            if zero_inactive_output and c_or_none is not None:
                from sglang.kernels.ops.moe.zero_inactive_routes import (
                    zero_inactive_routes,
                )

                zero_inactive_routes(
                    c.reshape(size_m * top_k, size_n),
                    sorted_token_ids,
                    expert_ids,
                    route_counts,
                    moe_block_size,
                    inactive_prefix_count=has_inactive_prefix_count
                    and expert_ids_are_sorted
                    and not is_ep,
                )
            if (
                not _sm70_marlin_user_tuning
                and getattr(b_scales, "_sm70_fp8_scale", False)
                and size_m >= 128
                and (size_n, size_k) in ((1280, 2560), (2560, 640))
                and is_k_full
                and not is_zp_float
                and all(
                    x is None
                    for x in (
                        b_bias_or_none,
                        global_scale_or_none,
                        b_zeros_or_none,
                        g_idx_or_none,
                        perm_or_none,
                    )
                )
            ):
                from sglang.srt.runtime_context import get_buffer

                def block_fp8_op():
                    try:
                        return torch.ops._moe_C.sm70_block_fp8_moe_gemm
                    except AttributeError as exc:
                        raise RuntimeError(
                            "Rebuild Volta Marlin with bash v100_plus/setup.sh "
                            "for block-FP8 expert prefill"
                        ) from exc

                # Only prepare_fp8_moe sets this tag after validating 128x128
                # checkpoint blocks. Per-column scales use the generic API.
                block_op = get_buffer("v100_block_fp8_marlin", block_fp8_op)
                return block_op(
                    a,
                    c,
                    b_q_weight,
                    b_scales,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    topk_weights,
                    moe_block_size,
                    top_k,
                    mul_topk_weights,
                    size_m,
                    size_n,
                    size_k,
                )
            _configure_sm70_nvfp4_stage(
                b_scales,
                moe_block_size,
                top_k,
                size_m,
                size_n,
                size_k,
            )
            op(
                a,
                c,
                b_q_weight,
                b_bias_or_none,
                b_scales,
                None,  # a_scales (W4A16 has no activation quantization)
                global_scale_or_none,
                b_zeros_or_none,
                g_idx_or_none,
                perm_or_none,
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                moe_block_size,
                top_k,
                mul_topk_weights,
                b_q_type.id,
                size_m,
                size_n,
                size_k,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
                -1,  # thread_k  (-1 => C++ model-specific auto-select)
                -1,  # thread_n
                -1,  # blocks_per_sm
            )
            return c
        # fall through to the stock JIT path (empty stub on SM70) with the
        # warning already emitted by _load_marlin_v100_op.

    raise RuntimeError("The V100 Marlin adapter cannot run on another architecture")
