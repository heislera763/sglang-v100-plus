from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.parametrize("tokens", [0, 1024, 1025, 1754, 2048, 2049])
def test_fp16_mhc_shards_preserve_projection_and_post_layout(tokens):
    """Changing GEMM row count or swapping channel orientation alters FP16 output.

    Shards must keep full-row projection arithmetic, own each valid row once,
    define padding, and concatenate to the original PRE/POST operation exactly.
    """
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("CUDA FP16 mHC coverage")
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
    from sglang.kernels.ops.layernorm.mhc_sm70 import (
        mhc_pre_shard_sm70,
        mhc_pre_sm70,
    )

    torch.manual_seed(536)
    storage = torch.full(
        (tokens + 2, 4, 4096), 13.0, device="cuda", dtype=torch.float16
    )
    residual = storage[1 : tokens + 1]
    residual.copy_(torch.randn_like(residual))
    saved = storage.clone()
    fn = torch.randn(24, 16384, device="cuda") * 0.015
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(24, device="cuda") * 0.3
    args = (residual, fn, scale, base, 1e-6, 1e-5, 1e-7, 1.7, 20)
    expected = mhc_pre_sm70(*args)
    shards = [mhc_pre_shard_sm70(*args, rank=rank) for rank in range(4)]
    actual = tuple(
        torch.cat([part[field] for part in shards])[:tokens] for field in range(3)
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    for rank, part in enumerate(shards):
        local_rows = (tokens + 3) // 4
        valid = max(0, min(local_rows, tokens - rank * local_rows))
        for tensor in part:
            assert tensor.is_contiguous()
            assert torch.count_nonzero(tensor[valid:]) == 0
    torch.testing.assert_close(storage, saved, rtol=0, atol=0)
    if tokens:
        update = torch.randn_like(actual[2])
        torch.testing.assert_close(
            mhc_post_split_h(update, residual, *actual[:2]),
            mhc_post_split_h(update, residual, *expected[:2]),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("tokens", [1025, 2048])
def test_fp16_mhc_shard_preparation_graph_reads_current_inputs(tokens):
    """Fused cast/statistics must preserve the mean tree and graph ownership.

    An unaligned contiguous source, an uneven last shard and refreshed weights
    catch vector-load assumptions, stale preparation and padded-row reads.
    """
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("CUDA FP16 mHC coverage")
    from sglang.kernels.ops.layernorm.mhc_sm70 import (
        mhc_pre_shard_sm70,
        mhc_pre_sm70,
    )

    torch.manual_seed(537)
    storage = torch.full(
        (tokens * 16384 + 2,), 13.0, device="cuda", dtype=torch.float16
    )
    residual = storage[1:-1].view(tokens, 4, 4096)
    residual.copy_(torch.randn_like(residual))
    fn = torch.randn(24, 16384, device="cuda") * 0.015
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(24, device="cuda") * 0.3
    args = (residual, fn, scale, base, 1e-6, 1e-5, 1e-7, 1.7, 20)
    rank = 3
    local_rows = (tokens + 3) // 4
    start = rank * local_rows
    valid = tokens - start
    mhc_pre_shard_sm70(*args, rank=rank)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mhc_pre_shard_sm70(*args, rank=rank)
    for refresh in range(3):
        residual.copy_(torch.randn_like(residual) * (1.0, 128.0, 2**-20)[refresh])
        fn.mul_(0.75)
        saved = storage.clone()
        expected = mhc_pre_sm70(*args)
        graph.replay()
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got[:valid], want[start:], rtol=0, atol=0)
            assert torch.count_nonzero(got[valid:]) == 0
        torch.testing.assert_close(storage, saved, rtol=0, atol=0)


def test_fp16_mhc_full_preparation_graph_reads_every_row():
    """Bulk PRE must reduce all rows, preserve the mean tree and own its cast.

    A two-byte source offset and uneven row count catch alignment assumptions;
    graph refresh at large/tiny magnitudes catches stale or truncated means.
    Canaries guard the full-row mean output as well as the FP32 cast.
    """
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("CUDA FP16 mHC coverage")
    from sglang.kernels.ops.layernorm.mhc_sm70 import (
        _prepare_shard_module,
        mhc_pre_sm70,
    )

    torch.manual_seed(538)
    rows = 1025
    source = torch.full((rows * 16384 + 2,), 13.0, device="cuda", dtype=torch.float16)
    residual = source[1:-1].view(rows, 4, 4096)
    residual.copy_(torch.randn_like(residual))
    cast_storage = torch.full((rows * 16384 + 8,), 17.0, device="cuda")
    cast = cast_storage[4:-4].view(rows, 16384)
    mean_storage = torch.full((rows + 2,), 19.0, device="cuda")
    mean = mean_storage[1:-1]
    fn = torch.randn(24, 16384, device="cuda") * 0.015
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(24, device="cuda") * 0.3
    args = (residual, fn, scale, base, 1e-6, 1e-5, 1e-7, 1.7, 20)
    prepare = _prepare_shard_module().run
    prepare(residual, cast, mean, 0)
    mhc_pre_sm70(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare(residual, cast, mean, 0)
        actual = mhc_pre_sm70(*args)
    for magnitude in (1.0, 128.0, 2**-20):
        residual.copy_(torch.randn_like(residual) * magnitude)
        fn.mul_(0.75)
        saved = source.clone()
        expected = mhc._mhc_pre_torch(*args)
        graph.replay()
        torch.testing.assert_close(cast, residual.flatten(1).float(), rtol=0, atol=0)
        torch.testing.assert_close(mean, cast.square().mean(-1), rtol=0, atol=0)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0.001, atol=0.002)
        torch.testing.assert_close(source, saved, rtol=0, atol=0)
        assert torch.all(cast_storage[:4] == 17) and torch.all(cast_storage[-4:] == 17)
        assert mean_storage[0] == mean_storage[-1] == 19


@pytest.mark.parametrize(
    "tokens,iterations,fuse_projection",
    [
        (0, 20, False),
        (1, 1, False),
        (17, 20, False),
        (256, 20, False),
        (2048, 20, False),
        (1, 1, True),
        (1, 20, True),
        (2, 1, True),
        (3, 20, True),
        (4, 20, True),
        (8, 20, True),
    ],
)
def test_fp16_mhc_matches_torch(tokens, iterations, fuse_projection):
    """Guard FP32 gates and input/output channel orientation in FP16 mHC.

    A transposed residual mixing matrix or a BF16/FP16 parameter cast can
    silently produce plausible model output. Compare the whole pre/post pair
    with the independent unfused implementation, including distinct epsilons
    and a non-default post multiplier.
    """
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("CUDA FP16 mHC coverage")
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
    from sglang.kernels.ops.layernorm.mhc_sm70 import mhc_pre_sm70

    torch.manual_seed(533)
    residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.float16)
    fn = torch.randn(24, 16384, device="cuda") * 0.015
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(24, device="cuda") * 0.3
    args = (residual, fn, scale, base, 1e-6, 1e-5, 1e-7, 1.7, iterations)
    unchanged_residual = residual.clone()
    reference = mhc._mhc_pre_torch(*args)
    actual = mhc_pre_sm70(*args, fuse_projection=fuse_projection)
    assert torch.equal(residual, unchanged_residual)
    for got, expected in zip(actual, reference):
        assert got.dtype == expected.dtype
        torch.testing.assert_close(got, expected, atol=0.002, rtol=0.001)
    post, comb, layer = actual
    torch.testing.assert_close(
        comb.sum(-2), torch.ones_like(comb.sum(-2)), atol=1e-5, rtol=0
    )
    x = torch.randn_like(layer)
    expected = mhc._mhc_post_torch(x, residual, reference[0], reference[1])
    got = mhc_post_split_h(x, residual, post, comb)
    torch.testing.assert_close(got, expected, atol=0.002, rtol=0.001)


class TestFP16MHCWorkspace(CustomTestCase):
    def test_prefill_avoids_two_full_fp32_workspaces(self):
        """A 2048-row chunk must not hold both cast and square copies live.

        The redundant square temporary caused a real 128 MiB CUDA allocation
        failure on GLM PP2 MTP. Live residual storage must remain unchanged.
        """
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA FP16 mHC coverage")
        from sglang.kernels.ops.layernorm.mhc_sm70 import mhc_pre_sm70

        torch.manual_seed(535)
        residual = torch.randn(2048, 4, 4096, device="cuda", dtype=torch.float16)
        original = residual.clone()
        fn = torch.randn(24, 16384, device="cuda") * 0.015
        scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
        base = torch.randn(24, device="cuda") * 0.3
        args = (residual, fn, scale, base, 1e-6, 1e-5, 1e-7, 1.7, 20)
        warm = mhc_pre_sm70(*args)
        torch.cuda.synchronize()
        del warm
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        outputs = mhc_pre_sm70(*args)
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - baseline
        fp32_input_bytes = residual.numel() * 4
        self.assertLess(peak_extra, 2 * fp32_input_bytes)
        self.assertTrue(torch.equal(residual, original))
        reference = mhc._mhc_pre_torch(*args)
        for got, expected in zip(outputs, reference):
            torch.testing.assert_close(got, expected, atol=0.002, rtol=0.001)


@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_fp16_mhc_projection_graph_refresh(tokens):
    """Decode graphs must read current residuals and FP32 learned parameters."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("CUDA FP16 mHC coverage")
    from sglang.kernels.ops.layernorm.mhc_sm70 import mhc_pre_sm70

    torch.manual_seed(534)
    residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.float16)
    fn = torch.randn(24, 16384, device="cuda") * 0.015
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(24, device="cuda") * 0.3
    args = (residual, fn, scale, base, 1e-6, 1e-5, 1e-7, 1.7, 20)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mhc_pre_sm70(*args, fuse_projection=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mhc_pre_sm70(*args, fuse_projection=True)
    for magnitude in (0.0, 0.25, 2.0):
        residual.copy_(torch.randn_like(residual) * magnitude)
        fn.copy_(torch.randn_like(fn) * 0.015)
        base.copy_(torch.randn_like(base) * 0.3)
        graph.replay()
        expected = mhc._mhc_pre_torch(*args)
        for got, ref in zip(actual, expected):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, ref, atol=0.002, rtol=0.001)
        # Projection partials and norm reductions are row-owned. A row offset
        # omitted in either kernel mixes different draft positions silently.
        singles = [
            mhc_pre_sm70(residual[row : row + 1], *args[1:], fuse_projection=True)
            for row in range(tokens)
        ]
        for output, rows in zip(actual, zip(*singles)):
            assert torch.equal(output, torch.cat(rows))


@pytest.mark.parametrize("iterations,rms_eps", [(1, 1e-6), (20, 1e-6), (20, 1e-5)])
def test_fp16_mhc_pointwise_exact_graph_refresh(iterations, rms_eps):
    """Pointwise fusion must retain default reductions and exact outputs.

    Changing cast/square precision, mean reduction order, rsqrt semantics or
    graph input refresh can otherwise hide behind the approximate mHC tests.
    """
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("CUDA FP16 mHC coverage")
    from sglang.kernels.ops.layernorm.mhc_sm70 import mhc_pre_sm70

    torch.manual_seed(536)
    residual = torch.randn(1, 4, 4096, device="cuda", dtype=torch.float16)
    fn = torch.randn(24, 16384, device="cuda") * 0.015
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(24, device="cuda") * 0.3
    args = (residual, fn, scale, base, rms_eps, 1e-5, 1e-7, 1.7, iterations)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mhc_pre_sm70(*args, fuse_pointwise=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mhc_pre_sm70(*args, fuse_pointwise=True)
    for magnitude in (0.0, 2**-15, 0.25, 1.0, 32.0):
        for _ in range(8):
            residual.copy_(torch.randn_like(residual) * magnitude)
            fn.copy_(torch.randn_like(fn) * 0.015)
            base.copy_(torch.randn_like(base) * 0.3)
            eager = mhc_pre_sm70(*args, fuse_pointwise=True)
            reference = mhc_pre_sm70(*args)
            graph.replay()
            for outputs in (eager, captured):
                for actual, expected in zip(outputs, reference):
                    assert torch.equal(actual, expected)


@pytest.fixture
def stated_tp_group():
    """Provide a TP-group placeholder for kernels with mocked symmetric memory."""
    from sglang.srt.runtime_context import get_parallel

    with get_parallel().override(tp_group=None):
        yield


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [0, 1, 6, 8, 17, 32, 64])
@pytest.mark.parametrize("use_norm", [False, True])
def test_mhc_fused_post_pre_matches_unfused(
    monkeypatch, hidden_size, num_tokens, use_norm, stated_tp_group
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang mHC kernels")

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    # Disable symmetric-memory allocation for this single-process kernel test.
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hc_mult = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    residual = (
        torch.randn(
            num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    post_prev = torch.rand(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_prev = (
        torch.rand(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
        * 0.25
    )
    fn = (
        torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device=device, dtype=torch.float32)
    norm_weight = (
        torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
        if use_norm
        else None
    )
    norm_eps = 1e-6 if use_norm else None

    rms_eps = 1e-6
    hc_eps = 1e-6
    sinkhorn_repeat = 2

    residual_ref = post_ref = comb_ref = layer_ref = None
    if num_tokens > 0:
        residual_ref = mhc_post(x, residual, post_prev, comb_prev)
        post_ref, comb_ref, layer_ref = mhc_pre(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    residual_out, post_out, comb_out, layer_out = mhc_fused_post_pre(
        x,
        residual,
        post_prev,
        comb_prev,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        2.0,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    if hidden_size == 4096 and num_tokens in (0, 1, 6, 17):
        _check_glm_boundary(
            x,
            residual,
            post_prev,
            comb_prev,
            fn,
            hc_scale,
            hc_base,
            use_norm=use_norm,
        )

    torch.cuda.synchronize()
    if num_tokens == 0:
        assert residual_out.shape == residual.shape
        assert post_out.shape == (0, hc_mult, 1)
        assert comb_out.shape == (0, hc_mult, hc_mult)
        assert layer_out.shape == (0, hidden_size)
        assert residual_out.dtype == torch.bfloat16
        assert post_out.dtype == torch.float32
        assert comb_out.dtype == torch.float32
        assert layer_out.dtype == torch.bfloat16
        return

    assert residual_ref is not None
    assert post_ref is not None
    assert comb_ref is not None
    assert layer_ref is not None
    assert residual_out.shape == residual_ref.shape
    assert post_out.shape == post_ref.shape
    assert comb_out.shape == comb_ref.shape
    assert layer_out.shape == layer_ref.shape

    torch.testing.assert_close(residual_out, residual_ref, atol=0, rtol=0)
    torch.testing.assert_close(post_out, post_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb_out, comb_ref, atol=1e-3, rtol=1e-3)
    layer_atol = 2e-2 if use_norm else 2e-3
    layer_rtol = 2e-2 if use_norm else 2e-3
    torch.testing.assert_close(layer_out, layer_ref, atol=layer_atol, rtol=layer_rtol)


def _check_glm_boundary(x, residual, post, comb, fn, scale, base, *, use_norm):
    from sglang.srt.environ import envs
    from sglang.srt.layers.layer_boundary import MHCState
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.glm5_next import Glm5NextDecoderLayer

    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        mhc=True,
        hc_mult=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    layer.hc_ffn_fn = torch.nn.Parameter(fn)
    layer.hc_ffn_scale = torch.nn.Parameter(scale)
    layer.hc_ffn_base = torch.nn.Parameter(base)
    norm = RMSNorm(x.shape[-1], eps=1e-6).to(x) if use_norm else None
    states = [
        MHCState(
            hc_mult=4,
            hc_attn_pre=layer.hc_attn_pre,
            hc_ffn_pre=layer.hc_ffn_pre,
            hc_post=layer.hc_post,
            hc_ffn_post_pre=callback,
            h_res=comb.flatten(1),
            h_post=post.flatten(1),
        )
        for callback in (None, layer.hc_ffn_post_pre)
    ]
    # Literal, not derived from the cutoff constant: deriving it makes this a
    # mirror that stays green when the cutoff moves. None is the empty batch,
    # which update_and_read_ffn_input short-circuits before reaching the callback.
    fused_expected = {1: True, 6: True, 17: False}[x.shape[0]] if x.shape[0] else None
    with envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(True):
        if x.shape[0] > 0:
            declined = (
                layer.hc_ffn_post_pre(
                    hidden_states=x,
                    residual=residual.flatten(1),
                    h_res=comb.flatten(1),
                    h_post=post.flatten(1),
                    out_norm_weight=None,
                    out_norm_eps=None,
                )
                is None
            )
            assert declined is not fused_expected, (
                f"num_tokens={x.shape[0]} fused={not declined}, "
                f"expected fused={fused_expected}"
            )
        outputs = [
            s.update_and_read_ffn_input(x, residual.flatten(1), norm) for s in states
        ]
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
    torch.testing.assert_close(states[0].h_res, states[1].h_res, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(states[0].h_post, states[1].h_post, atol=1e-3, rtol=1e-3)
    # The next combine must consume the FFN mixing matrices, not attention's.
    torch.testing.assert_close(
        states[0].apply_post(x, outputs[0][1]),
        states[1].apply_post(x, outputs[1][1]),
        atol=2e-3,
        rtol=2e-2,
    )


def _hopper_mhc_layer(norm, w, scale, base):
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    layer = SimpleNamespace(
        input_layernorm=norm,
        post_attention_layernorm=norm,
        config=SimpleNamespace(model_type="deepseek_v41"),
        hc_pre_from_prev_sublayer=True,
        hc_attn_fn=w,
        hc_ffn_fn=-w,
        hc_attn_scale=scale,
        hc_ffn_scale=scale,
        hc_attn_base=base,
        hc_ffn_base=base,
        hc_cfg=model_mhc.HcConfig(4, 20, 1e-6, 1e-6, 5120, True, False),
        hc_mult=4,
        hc_sinkhorn_iters=20,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
    )
    layer._init_hyper_connections = lambda: (
        DeepseekV4DecoderLayer._init_hyper_connections(layer)
    )
    return layer


@pytest.mark.parametrize("num_tokens", [32, 33, 64, 128, 257, 4096, 4097, 8192])
def test_hopper_compensated_mhc(num_tokens):
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.utils import is_sm90_supported

    if not is_sm90_supported():
        pytest.skip("Hopper compensated mHC dispatch")
    torch.manual_seed(192 + num_tokens)
    x = torch.randn(num_tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 20480, device="cuda") * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    norm = SimpleNamespace(weight=torch.ones(5120, device="cuda", dtype=torch.bfloat16))
    layer = _hopper_mhc_layer(norm, w, scale, base)
    with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        assert layer._hc_attn_bf16_parts is not None
        assert layer._hc_attn_tf32_parts is None
        actual = model_mhc.mix_stats(layer.attn_hc, x)
    xd = x.flatten(1).double()
    z = (xd @ w.double().T) * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-6)
    expected_pre = torch.sigmoid(z[:, :4] * scale[0] + base[:4]) + 1e-6
    expected_post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    comb = (z[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
    comb = (comb - comb.amax(-1, keepdim=True)).exp()
    comb = comb / comb.sum(-1, keepdim=True) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for result, expected in zip(actual, (expected_pre, expected_post, comb)):
        torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)
    with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False):
        fallback = model_mhc.mix_stats(layer.attn_hc, x)
        for result, expected in zip(fallback, (expected_pre, expected_post, comb)):
            torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        assert layer._hc_attn_bf16_parts is None


@pytest.mark.parametrize("num_tokens", [1, 4, 64, 4096])
def test_hopper_combine_norm(num_tokens):
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.utils import is_sm90_supported

    if not is_sm90_supported():
        pytest.skip("Hopper combine/norm dispatch")
    torch.manual_seed(39 + num_tokens)
    x = torch.randn(num_tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    pre = torch.rand(num_tokens, 4, device="cuda")
    norm = RMSNorm(5120, eps=1e-6).cuda().bfloat16()
    hc = model_mhc.HcSubLayer(
        model_mhc.HcConfig(4, 20, 1e-6, 1e-6, 5120, True, False),
        None,
        None,
        None,
        norm,
    )
    actual = model_mhc.combine(hc, model_mhc.HcState(x, pre))
    expected = norm(mhc.mhc_combine(x.flatten(1), pre, 4, x.dtype))
    torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=1e-5)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
def test_hopper_mhc_stats_stream_graph(num_tokens):
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.utils import is_sm90_supported

    if not is_sm90_supported():
        pytest.skip("Hopper mHC overlap")
    torch.manual_seed(934 + num_tokens)
    x = torch.randn(num_tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 20480, device="cuda") * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    pre = torch.rand(num_tokens, 4, device="cuda")
    norm = RMSNorm(5120, eps=1e-6).cuda().bfloat16()
    layer = _hopper_mhc_layer(norm, w, scale, base)
    layer.hc_stats_stream = torch.cuda.Stream()
    DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
    assert model_mhc.use_stats_stream(
        layer.hc_cfg, SimpleNamespace(forward_mode=ForwardMode.DECODE), x
    )
    stream = layer.hc_stats_stream

    def run(stats_stream):
        # The layer forks the side stream right before each combine; the combine
        # itself no longer takes a stream.
        model_mhc.fork_stats_stream(stats_stream)
        combined = model_mhc.combine(layer.attn_hc, model_mhc.HcState(x, pre))
        coefficients = model_mhc.mix_stats(layer.attn_hc, x, stats_stream)
        if stats_stream is not None:
            torch.cuda.current_stream().wait_stream(stats_stream)
        # Read on the consumer stream after the join, as hc_post does.
        return combined.clone(), *(v.clone() for v in coefficients)

    run(None)
    run(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(stream)
    for _ in range(3):
        x.normal_()
        pre.uniform_()
        expected = run(None)
        graph.replay()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_mhc_refresh_without_deepgemm_prenorm():
    from unittest.mock import patch

    from sglang.srt.environ import envs
    from sglang.srt.layers.deep_gemm_wrapper import configurer
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.runtime_context import override_platform

    norm = SimpleNamespace(weight=torch.ones(5120, device="cuda", dtype=torch.bfloat16))
    w = torch.randn(24, 20480, device="cuda")
    scale = torch.ones(3, device="cuda")
    base = torch.zeros(24, device="cuda")
    layer = _hopper_mhc_layer(norm, w, scale, base)
    layer._can_fuse_attn_mhc = layer._can_fuse_ffn_mhc = True
    with (
        override_platform(is_sm100=True, is_sm90=False),
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
        patch.object(configurer, "ENABLE_JIT_DEEPGEMM", True),
        patch.dict("sys.modules", {"deep_gemm": SimpleNamespace()}),
    ):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
    assert layer.attn_hc.fn is w
    assert layer.ffn_hc.fn is layer.hc_ffn_fn
    assert layer.attn_hc.tf32_parts is None
    assert layer.attn_hc.bf16_parts is None
    assert "_can_fuse_attn_mhc" not in layer.__dict__
    assert "_can_fuse_ffn_mhc" not in layer.__dict__


@pytest.mark.parametrize(
    "replicated,add_shared,has_shared,expected",
    [
        (False, False, True, 16.0),
        (True, False, True, 2.0),
        (True, True, True, 9.0),
        (True, True, False, 2.0),
    ],
)
def test_moe_post_reduces_before_adding_replicated_shared(
    monkeypatch, replicated, add_shared, has_shared, expected
):
    from sglang.srt.layers import moe
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.models import deepseek_v4_mhc as model_mhc

    routed = torch.ones(2, 3)
    shared = torch.full((2, 3), 7.0) if has_shared else None
    pieces = model_mhc.MoEOutput(routed, shared, None, 1.0, replicated)
    partial = routed if replicated or shared is None else routed + shared
    monkeypatch.setattr(model_mhc.MoEOutput, "get_merged", lambda self: partial)
    monkeypatch.setattr(model_mhc, "is_deferred_finalize", lambda value: False)
    monkeypatch.setattr(
        moe_utils, "should_add_replicated_moe_output", lambda: add_shared
    )
    reduced_inputs = []

    def reduce(value):
        reduced_inputs.append(value.clone())
        return value * 2

    monkeypatch.setattr(moe, "post_experts_all_reduce", reduce)
    monkeypatch.setattr(model_mhc, "_compute_triplet", lambda *args: None)
    monkeypatch.setattr(
        model_mhc, "_post_fusion", lambda hc, value, residual, triplet, *args: value
    )
    actual = model_mhc.run_moe_post(
        None, pieces, None, stats_stream=None, next=None, world_size=2
    )
    assert len(reduced_inputs) == 1
    torch.testing.assert_close(reduced_inputs[0], partial)
    torch.testing.assert_close(actual, torch.full((2, 3), expected))


_EPS = 1e-6
_SINKHORN_ITERS = 20
_POST_MULT = 2.0


@pytest.fixture
def single_process(monkeypatch, stated_tp_group):
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    yield


def _mhc_pre_reference(residual, fn, hc_scale, hc_base, *, eps_outside_sqrt):
    hc_mult = residual.shape[1]
    x = residual.flatten(1).float()
    mixes = x @ fn.t()
    mean_sq = x.pow(2).mean(-1, keepdim=True)
    if eps_outside_sqrt:
        rms = 1.0 / (mean_sq.sqrt() + _EPS)
    else:
        rms = torch.rsqrt(mean_sq + _EPS)
    mixes = mixes * rms
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + _EPS
    post = _POST_MULT * torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = (mixes[:, 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]).view(
        -1, hc_mult, hc_mult
    )
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + _EPS
    comb = comb / (comb.sum(-2, keepdim=True) + _EPS)
    for _ in range(_SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + _EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + _EPS)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(1)
    return post, comb, layer_input


def _eps_outside_sqrt_inputs(num_tokens, hc_mult=4, hidden=4096):
    torch.manual_seed(0)
    mix = hc_mult * (2 + hc_mult)
    # A small residual puts mean_sq near eps, where the two eps placements differ.
    residual = (
        torch.randn(num_tokens, hc_mult, hidden, device="cuda") * 1e-3
    ).bfloat16()
    fn = torch.randn(mix, hc_mult * hidden, device="cuda") * 0.01
    hc_scale = torch.rand(3, device="cuda")
    hc_base = torch.randn(mix, device="cuda") * 0.1
    return residual, fn, hc_scale, hc_base


def _run_mhc_pre_eps_outside_sqrt(residual, fn, hc_scale, hc_base, **norm):
    return mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=_EPS,
        hc_pre_eps=_EPS,
        hc_sinkhorn_eps=_EPS,
        hc_post_mult_value=_POST_MULT,
        sinkhorn_repeat=_SINKHORN_ITERS,
        eps_outside_sqrt=True,
        **norm,
    )


def _assert_post_uses_eps_outside_sqrt(post, inputs):
    ref_post, _, _ = _mhc_pre_reference(*inputs, eps_outside_sqrt=True)
    dsv4_post, _, _ = _mhc_pre_reference(*inputs, eps_outside_sqrt=False)
    # Tolerance covers the kernel's split-K fp32 GEMM order.
    torch.testing.assert_close(post.squeeze(-1), ref_post, rtol=2e-3, atol=2e-3)
    assert (post.squeeze(-1) - dsv4_post).abs().max() > 1e-2


@pytest.mark.parametrize("num_tokens", [7, 300, 3000])
def test_mhc_pre_eps_outside_sqrt(num_tokens, single_process):
    inputs = _eps_outside_sqrt_inputs(num_tokens)
    post, comb, layer_input = _run_mhc_pre_eps_outside_sqrt(*inputs)
    _, ref_comb, ref_input = _mhc_pre_reference(*inputs, eps_outside_sqrt=True)
    _assert_post_uses_eps_outside_sqrt(post, inputs)
    torch.testing.assert_close(comb, ref_comb, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(layer_input.float(), ref_input, rtol=2e-2, atol=1e-4)


@pytest.mark.parametrize("num_tokens", [7, 300])
def test_mhc_pre_eps_outside_sqrt_fused_norm(num_tokens, single_process):
    inputs = _eps_outside_sqrt_inputs(num_tokens)
    hidden = inputs[0].shape[-1]
    norm_weight = (1 + 0.1 * torch.randn(hidden, device="cuda")).bfloat16()
    post, _, layer_input = _run_mhc_pre_eps_outside_sqrt(
        *inputs, norm_weight=norm_weight, norm_eps=_EPS
    )
    _assert_post_uses_eps_outside_sqrt(post, inputs)
    _, _, ref_input = _mhc_pre_reference(*inputs, eps_outside_sqrt=True)
    ref = torch.nn.functional.rms_norm(ref_input, (hidden,), norm_weight.float(), _EPS)
    # The reference norms the fp32 layer input; the kernel scales a bf16 stash.
    torch.testing.assert_close(layer_input.float(), ref, rtol=2e-2, atol=2e-2)


def _hc_head_reference(x, fn, scale, base, norm_eps, hc_eps, eps_outside_sqrt):
    flat = x.flatten(1).float()
    mean_sq = flat.square().mean(-1, keepdim=True)
    if eps_outside_sqrt:
        rsqrt = 1.0 / (mean_sq.sqrt() + norm_eps)
    else:
        rsqrt = torch.rsqrt(mean_sq + norm_eps)
    pre = torch.sigmoid(torch.nn.functional.linear(flat, fn) * rsqrt * scale + base)
    return torch.sum((pre + hc_eps).unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


@pytest.mark.parametrize("eps_outside_sqrt", [False, True])
def test_fused_hc_head(eps_outside_sqrt):
    from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head

    torch.manual_seed(0)
    num_tokens, hc, hidden = 33, 4, 4096
    x = torch.randn(num_tokens, hc, hidden, device="cuda").bfloat16()
    fn = torch.randn(hc, hc * hidden, device="cuda") * 0.1
    scale = torch.ones(1, device="cuda")
    base = torch.randn(hc, device="cuda") * 0.1
    # A large eps puts a quarter or more of the outputs of the two formulas
    # outside the tolerance of each other.
    norm_eps = 0.5
    out = fused_hc_head(
        x,
        fn,
        scale,
        base,
        norm_eps=norm_eps,
        hc_eps=1e-6,
        eps_outside_sqrt=eps_outside_sqrt,
    )
    ref = _hc_head_reference(x, fn, scale, base, norm_eps, 1e-6, eps_outside_sqrt)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
