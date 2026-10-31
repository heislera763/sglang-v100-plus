"""Opt-in SM70 gemm support."""

import os

import torch
from sglang_v100_plus.kernels.utils import cache_once, load_jit

# (output features, input features) -> (threads, lanes per row, vector width).
_DENSE_CONFIGS = {
    (4096, 2560): (64, 16, 8),
    (3584, 2560): (64, 32, 8),
    (2560, 1536): (128, 32, 8),
    (320, 2560): (256, 32, 8),
    (2560, 160): (128, 8, 4),
    (512, 2560): (256, 32, 8),
    (640, 2560): (128, 32, 8),
    (24, 2560): (64, 32, 8),
    (1, 2560): (256, 32, 8),
    (10240, 2560): (64, 32, 8),
    (2560, 2560): (128, 32, 8),
    # Qwen TP8: GDN/QSA projections, shared expert and vocabulary shard.
    (2048, 2560): (64, 32, 8),
    (12, 2560): (64, 32, 8),
    (2560, 768): (128, 32, 8),
    (160, 2560): (256, 32, 8),
    (2560, 80): (128, 8, 4),
    (31040, 2560): (64, 32, 8),
}


def supported(x: torch.Tensor, weight: torch.Tensor, bias=None) -> bool:
    return (
        os.environ.get("SGLANG_SM70_DENSE_GEMV", "0") == "1"
        and x.is_cuda
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and weight.device == x.device
        and x.ndim == 2
        and x.shape[0] in (1, 2, 3, 4)
        and weight.ndim == 2
        and x.shape[1] == weight.shape[1]
        and bias is None
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and (
            (
                x.shape[0] == 1
                and (
                    tuple(weight.shape) in _DENSE_CONFIGS
                    or (weight.shape[1] == 2560 and weight.shape[0] >= 32768)
                )
            )
            or shape_supported(x, weight)
        )
        and torch.cuda.get_device_capability(x.device) == (7, 0)
    )


@cache_once
def _dense_module(threads: int, lanes: int, vector: int):
    return load_jit(
        "sm70_dense_gemv",
        threads,
        lanes,
        vector,
        cuda_files=["elementwise/sm70_dense_gemv.cuh"],
        cuda_wrappers=[
            ("gemv", f"sglang::sm70_dense_gemv::gemv<{threads},{lanes},{vector}>")
        ],
    )


def linear_dense(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != 1:
        return linear_small(x, weight)
    config = _DENSE_CONFIGS.get(tuple(weight.shape), (64, 32, 8))
    out = torch.empty((1, weight.shape[0]), dtype=x.dtype, device=x.device)
    _dense_module(*config).gemv(x, weight, out)
    return out


# (rows, output features, input features) -> (threads, lanes per output).
# Shapes where cuBLAS is faster deliberately stay on the existing path.
_SMALL_CONFIGS = {
    (2, 24, 2560): (64, 32),
    (2, 640, 2560): (256, 32),
    (2, 1, 2560): (64, 32),
    (2, 2560, 2560): (128, 32),
    (4, 24, 2560): (64, 32),
    (4, 640, 2560): (128, 32),
    (4, 2560, 2560): (64, 32),
    (2, 4096, 2560): (256, 32),
    (4, 4096, 2560): (64, 32),
    (2, 3584, 2560): (64, 32),
    (4, 3584, 2560): (64, 32),
    (2, 2560, 1536): (64, 32),
    (4, 2560, 1536): (128, 16),
    (2, 320, 2560): (256, 32),
    (4, 320, 2560): (256, 32),
    (2, 2560, 160): (128, 8),
    (4, 2560, 160): (128, 8),
    (2, 512, 2560): (256, 32),
    (4, 512, 2560): (256, 32),
    (2, 62080, 2560): (128, 32),
    (4, 62080, 2560): (128, 32),
}

for _rows in (2, 4):
    _SMALL_CONFIGS.update(
        {
            (_rows, 2048, 2560): (64, 32),
            (_rows, 12, 2560): (64, 32),
            (_rows, 2560, 768): (128, 16),
            (_rows, 160, 2560): (256, 32),
            (_rows, 2560, 80): (128, 8),
            (_rows, 31040, 2560): (128, 32),
        }
    )

# Each row has independent accumulators; reuse the validated four-row geometry.
_SMALL_CONFIGS.update(
    {(3, n, k): config for (rows, n, k), config in _SMALL_CONFIGS.items() if rows == 4}
)
_SMALL_CONFIGS[(3, 1, 2560)] = (64, 32)

# GLM-5.3-Flash's checkpoint-declared unquantized projections, TP8/TP4.
# These use Volta cuBLAS for both decode and verification; they have no
# custom small-GEMM implementation to fall back from.
_GLM_BLAS_SHAPES = {
    (264, 4096),
    (272, 4096),  # FP4-Local KDA B/FG gates, separate from MXFP8 QKV (TP8/TP4)
    (3336, 4096),
    (6416, 4096),  # fused KDA QKV/beta/forget/norm gates
    (2048, 4096),
    (1536, 4096),
    (512, 4096),  # MLA latent projections
    (2048, 1536),
    (4096, 1536),  # MLA queries and replicated indexer query
    (4096, 512),
    (8192, 512),  # MLA keys/values
    (4096, 1024),
    (4096, 2048),
    (4096, 4096),  # KDA/MLA output
    (128, 4096),  # replicated indexer key
    (1024, 4096),
    (4096, 256),  # unquantized NextN shared expert (TP4/TP8)
    (768, 4096),
    (3072, 4096),  # DFlash2 TP8 fused QKV and gate/up projections
}

# MiniMax's unquantized vocabulary, draft projections and candidate heads.
# cuBLAS is the selected FP16 backend; packed target projections use W8A16.
_MINIMAX_BLAS_SHAPES = {
    (25008, 6144),
    (50016, 6144),
    (1280, 6144),
    (1536, 6144),
    (2560, 6144),
    (3072, 6144),
    (6144, 1024),
    (6144, 1536),
    (6144, 2048),
    (6144, 3072),
    (768, 30720),
    (768, 36864),
    (1536, 30720),
    (6144, 36864),
    (256, 6144),
    (6144, 6144),
}


def shape_supported(x, weight):
    return (
        os.environ.get("SGLANG_SM70_MTP_SMALL_GEMM", "1") == "1"
        and (x.shape[0], *weight.shape) in _SMALL_CONFIGS
    )


def blas_supported(x, weight, bias=None):
    """Explicit Volta library paths, separate from small decode kernels.

    Known small-kernel shapes must use that implementation or fail. cuBLAS
    is the selected backend for prefill, Qwen's four-bank embedding projection,
    the two untuned Qwen projections, and GLM's unquantized projections.
    The GLM indexer head-weight projection intentionally stays FP32.
    """
    if not (
        x.is_cuda
        and x.dtype == weight.dtype
        and x.dtype in (torch.float16, torch.float32)
        and x.device == weight.device
        and x.ndim in (2, 3)
        and weight.ndim == 2
        and x.shape[-1] == weight.shape[1]
        and x.is_contiguous()
        and weight.is_contiguous()
        and (bias is None or (bias.device == x.device and bias.dtype == x.dtype))
        and torch.cuda.get_device_capability(x.device) == (7, 0)
    ):
        return False
    if x.dtype == torch.float32:
        return (
            x.ndim == 2
            and tuple(weight.shape) in ((32, 4096), (128, 6144))
            and bias is None
        )
    if x.ndim == 2:
        rows = x.shape[0]
        shape = tuple(weight.shape)
        native_small = (rows, *shape) in _SMALL_CONFIGS or (
            rows == 1
            and (shape in _DENSE_CONFIGS or (shape[1] == 2560 and shape[0] >= 32768))
        )
        if native_small:
            return False
        if (
            shape in _GLM_BLAS_SHAPES or shape in _MINIMAX_BLAS_SHAPES
        ) and bias is None:
            return True
        if rows in (2, 3, 4) and shape in ((1, 2560), (10240, 2560)):
            return True
        if weight.shape[1] == 4096 and weight.shape[0] >= 16384:
            return True
    elif x.shape[1:] == (4, 2560) and weight.shape == (2560, 2560):
        return True
    from sglang_v100_plus.dispatch import in_prefill

    return in_prefill()


@cache_once
def _small_module(rows, threads, lanes):
    return load_jit(
        "sm70_small_gemm",
        rows,
        threads,
        lanes,
        cuda_files=["elementwise/sm70_small_gemm.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::sm70_small_gemm::run<{rows},{threads},{lanes}>")
        ],
    )


def linear_small(x, weight):
    threads, lanes = _SMALL_CONFIGS[(x.shape[0], *weight.shape)]
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _small_module(x.shape[0], threads, lanes).run(x, weight, out)
    return out
