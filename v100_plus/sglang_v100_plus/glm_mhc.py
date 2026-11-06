"""Select the FP16 mHC kernels for the opt-in SM70 integration."""

import torch

from .dispatch import in_prefill, reject_fallback


def mhc_pre_dispatch(
    original,
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_weight=None,
    norm_eps=None,
    n_splits_pre=32,
):
    """Share only eager GLM pre-read work; residual/post remain replicated.

    All token rows are restored before attention/experts. Coefficients travel
    separately in FP32 and retain the contiguous channel order hc_post requires.
    The learned FP32 projection stays replicated to preserve its reduction order.
    No global reduce-scatter, cache ownership change or small-row dispatch change.
    """
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec, get_schedule

    from .paired_prefill import is_active
    from .prefill import quad_partition

    partition = None
    if (
        in_prefill()
        and envs.SGLANG_OPT_SM70_HC_PREFILL_SP.get()
        # Fused full-row PRE avoids two gathers competing with the paired
        # schedule's outstanding TP sums. Other prefill layouts keep sharding.
        and not is_active()
        and residual.ndim == 3
        and residual.dtype == torch.float16
        # Below a large chunk the local gather costs more than replicated PRE.
        and residual.shape[0] >= 1024
        and residual.shape[1:] == (4, 4096)
        and get_schedule().disable_overlap_schedule
        and get_schedule().max_running_requests == 1
        and get_exec().graph.disable_prefill_cuda_graph
    ):
        partition = quad_partition(residual.shape[0])
    if partition is None:
        return original(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
            n_splits_pre=n_splits_pre,
        )

    from sglang.kernels.ops.layernorm.mhc_sm70 import mhc_pre_shard_sm70

    post, comb, layer_input = mhc_pre_shard_sm70(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        rank=partition.rank,
        size=partition.size,
    )
    norm_fused = norm_weight is not None
    if norm_fused:
        # Same AOT FP32-reduction/FP16-output RMSNorm as the unfused caller.
        from sgl_kernel import rmsnorm

        layer_input = rmsnorm(layer_input, norm_weight, norm_eps)
    coefficients = torch.cat(
        (post.reshape(partition.local_rows, 4), comb.reshape(partition.local_rows, 16)),
        dim=1,
    )
    layer_input = partition.gather(layer_input)
    coefficients = partition.gather(coefficients)
    # hc_post's kernel uses dense strides. Returning strided views here would
    # silently reinterpret a combined 20-float row as a 4- or 16-float row.
    post = coefficients[:, :4].contiguous().view(partition.rows, 4, 1)
    comb = coefficients[:, 4:].contiguous().view(partition.rows, 4, 4)
    return post, comb, layer_input, norm_fused


def mhc_pre(original, residual, *args, **kwargs):
    if residual.dtype != torch.float16 or residual.shape[1] != 4:
        reject_fallback(
            "glm.mhc_pre",
            "SM70 mHC requires FP16 residual with four branches",
            residual=residual,
        )
        return original(residual, *args, **kwargs)
    from sglang.kernels.ops.layernorm.mhc_sm70 import mhc_pre_sm70
    from sglang.srt.environ import envs

    fuse_projection = (
        1 <= residual.shape[0] <= 8
        and residual.shape[1:] == (4, 4096)
        and envs.SGLANG_OPT_SM70_MHC_PROJECTION.get()
    )
    fuse_pointwise = (
        residual.shape == (1, 4, 4096) and envs.SGLANG_OPT_SM70_MHC_POINTWISE.get()
    )
    return mhc_pre_sm70(
        residual,
        *args,
        fuse_projection=fuse_projection,
        fuse_pointwise=fuse_pointwise,
        **kwargs,
    )


def mhc_post(original, x, residual, post_layer_mix, comb_res_mix):
    if x.dtype != torch.float16 or residual.shape[1] != 4:
        reject_fallback(
            "glm.mhc_post",
            "SM70 mHC requires FP16 output and four residual branches",
            output=x,
            residual=residual,
        )
        return original(x, residual, post_layer_mix, comb_res_mix)
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h

    return mhc_post_split_h(x, residual, post_layer_mix, comb_res_mix)
