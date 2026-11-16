"""FP32 adapter updates, masked routes and changing-input graph replay."""

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Resident SM70 rank-one adapters",
)


@pytest.fixture(autouse=True)
def runtime_context():
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    publish(
        ServerArgs(
            model_path="dummy",
            moe_runner_backend="marlin",
            disable_overlap_schedule=True,
            max_running_requests=1,
        ),
        role="test",
    )
    yield
    torch.cuda.synchronize()
    reset_context()


def _reference(x, base, a, b, ids=None, weights=None):
    x, base, a, b = (t.cpu().double() for t in (x, base, a, b))
    result = base.half().clone()
    for row in range(x.shape[0]):
        e = 0 if ids is None else int(ids.flatten()[row])
        if e < 0:
            continue
        alpha = (x[row] * a[e]).sum().float()
        delta = alpha * b[e].float()
        result[row] = base[row].half().float() + delta
        if weights is not None:
            result[row] = result[row].float() * float(weights.flatten()[row])
    return result


@pytest.mark.parametrize("factor_dtype", [torch.float16, torch.float32])
def test_dense_factors_and_output_canaries(factor_dtype):
    from sglang.kernels.ops.lora.sm70_rank_one import rank_one_add

    torch.manual_seed(531)
    for m, k in ((0, 160), (1, 1536), (4, 1536), (33, 160)):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        a = torch.randn(1, k, device="cuda", dtype=factor_dtype) * 0.1
        b = torch.randn(1, 2560, device="cuda", dtype=factor_dtype) * 0.1
        storage = torch.full((m * 2560 + 32,), 19.0, device="cuda", dtype=torch.float16)
        y = storage[16:-16].view(m, 2560)
        y.normal_(0, 0.1)
        expected = _reference(x, y, a, b)
        rank_one_add(x, y, a, b)
        torch.testing.assert_close(y.cpu(), expected, rtol=0.001, atol=0.0002)
        assert (storage[:16] == 19).all() and (storage[-16:] == 19).all()

    # Rounding either the coefficient or delta to FP16 erases this update.
    x = torch.ones(1, 2, device="cuda", dtype=torch.float16)
    a = torch.tensor([[1.0, 2**-11]], device="cuda", dtype=factor_dtype)
    if factor_dtype == torch.float32:
        a = torch.tensor([[1.0 + 2**-12, 2**-12]], device="cuda")
    b = torch.ones(1, 1, device="cuda", dtype=factor_dtype)
    y = -torch.ones(1, 1, device="cuda", dtype=torch.float16)
    expected = (y.float() + (x.float() @ a.float().t()) @ b.float()).half()
    rank_one_add(x, y, a, b)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


@pytest.mark.parametrize("factor_dtype", [torch.float16, torch.float32])
def test_routed_changed_inputs_and_graph_replay(factor_dtype):
    from sglang.kernels.ops.lora.sm70_rank_one import routed_rank_one_add

    torch.manual_seed(531)
    m, k, n, e = 40, 640, 2560, 4
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    a = torch.randn(e, k, device="cuda", dtype=factor_dtype) * 0.1
    b = torch.randn(e, n, device="cuda", dtype=factor_dtype) * 0.1
    ids = torch.arange(m, device="cuda", dtype=torch.int32).remainder(e)
    ids[::3] = -1
    weights = torch.rand(m, device="cuda")
    base = torch.randn(m, n, device="cuda", dtype=torch.float16) * 0.1
    base[ids < 0] = 0
    storage = torch.full((m * n + 32,), 19.0, device="cuda", dtype=torch.float16)
    y = storage[16:-16].view(m, n)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            y.copy_(base)
            routed_rank_one_add(x, y, a, b, ids, weights)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y.copy_(base)
        routed_rank_one_add(x, y, a, b, ids, weights)
    for mode in (0, 1, 2):
        x.normal_(0, 0.1)
        a.normal_(0, 0.1)
        b.normal_(0, 0.1)
        weights.uniform_()
        ids.fill_(-1 if mode == 2 else mode)
        if mode == 1:
            ids[::2] = -1
        base.normal_(0, 0.1)
        base[ids < 0] = 0
        expected = _reference(x, base, a, b, ids.cpu(), weights.cpu())
        graph.replay()
        torch.testing.assert_close(y.cpu(), expected, rtol=0.001, atol=0.0002)
        assert (storage[:16] == 19).all() and (storage[-16:] == 19).all()


@pytest.mark.parametrize("factor_dtype", [torch.float16, torch.float32])
def test_packed_fp8_decode_and_marlin_with_expert_adapters(factor_dtype):
    from functools import partial
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.layers.moe.fused_moe_triton import fused_marlin_moe as marlin
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang_v100_plus.fp8 import fp8_marlin_scalar_type, prepare_fp8_moe
    from sglang_v100_plus.kernels.moe_marlin import moe_wna16_marlin_gemm
    from sglang_v100_plus.kernels.sm70_fp8_moe_decode import fp8_moe_decode

    torch.manual_seed(531)
    e, h, k = 3, 2560, 640
    w1 = (torch.randn(e, 2 * k, h, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    w2 = (torch.randn(e, h, k, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    layer = torch.nn.Module()
    layer.params_dtype = torch.float16
    for prefix, weight in (("w13", w1), ("w2", w2)):
        layer.register_parameter(
            prefix + "_weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        scales = torch.full(
            (e, weight.shape[1] // 128, weight.shape[2] // 128), 0.25, device="cuda"
        )
        layer.register_parameter(
            prefix + "_weight_scale_inv",
            torch.nn.Parameter(scales, requires_grad=False),
        )
    method = SimpleNamespace(
        use_mxfp8=False,
        is_fp4_expert=False,
        weight_block_size=[128, 128],
        quant_config=SimpleNamespace(is_checkpoint_fp8_serialized=True),
    )
    with patch(
        "sglang.srt.layers.moe.get_moe_runner_backend",
        return_value=MoeRunnerBackend.MARLIN,
    ):
        prepare_fp8_moe(None, method, layer)
    x = torch.randn(4, h, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.tensor(
        [[0, 1, -1], [2, 0, -1], [1, -1, 2], [2, 2, 0]],
        device="cuda",
        dtype=torch.int32,
    )
    weights = torch.rand(ids.shape, device="cuda") / 3
    a = torch.randn(e, k, device="cuda", dtype=factor_dtype) * 0.1
    b = torch.randn(e, h, device="cuda", dtype=factor_dtype) * 0.1
    expected = torch.zeros(4, h, dtype=torch.float32)
    w1_ref, w2_ref = w1.cpu().double() * 0.25, w2.cpu().double() * 0.25
    for row in range(4):
        for route in range(3):
            expert = int(ids[row, route])
            if expert < 0:
                continue
            projected = (x[row].cpu().double() @ w1_ref[expert].t()).half().float()
            activation = (
                torch.nn.functional.silu(projected[:k]) * projected[k:]
            ).half()
            base = (activation.double() @ w2_ref[expert].t()).half()[None, :]
            adapted = _reference(
                activation[None, :],
                base,
                a[expert : expert + 1],
                b[expert : expert + 1],
            )
            expected[row] += (
                (adapted[0].float() * float(weights[row, route])).half().float()
            )
    expected = expected.half()
    fast = fp8_moe_decode(
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        ids,
        weights,
        down_lora=(a, b),
    )
    torch.testing.assert_close(fast.cpu(), expected, rtol=0.001, atol=0.0002)

    with (
        patch.object(marlin, "moe_wna16_marlin_gemm", moe_wna16_marlin_gemm),
        patch.object(
            marlin,
            "get_scalar_type",
            partial(fp8_marlin_scalar_type, marlin.get_scalar_type),
        ),
    ):
        full = marlin.fused_marlin_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            torch.zeros(4, e, device="cuda"),
            weights,
            ids,
            down_lora_a=a,
            down_lora_b=b,
        )
    torch.testing.assert_close(full.cpu(), expected, rtol=0.001, atol=0.0002)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
