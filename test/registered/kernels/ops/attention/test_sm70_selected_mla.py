"""Hy4's index order changes FP16 attention results even for the same key set."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Volta Tensor Core accumulation order",
)
def test_selected_key_order_and_sinks():
    from sglang.kernels.ops.attention.dsa.sm70_sparse_mla import (
        _prefill,
        sparse_mla_sm70,
    )

    device = "cuda"
    rng = torch.Generator(device=device).manual_seed(531)
    rows, heads, width, dim = 129, 4, 2048, 576
    kv = torch.randn(4223, dim, device=device, generator=rng, dtype=torch.float16)
    q = torch.randn(rows, heads, dim, device=device, generator=rng, dtype=torch.float16)
    sink = torch.randn(heads, device=device, generator=rng)
    ids = torch.randint(
        len(kv), (rows, width), device=device, generator=rng, dtype=torch.int32
    )
    lengths = ids.new_full((rows,), width)
    ids[0] = -1
    ids[1, ::3] = len(kv)
    lengths[2] = 17
    args = q, kv, ids, lengths, sink, 0.0625
    expected = _prefill(*args, q.new_empty(rows, heads, 512))
    output = sparse_mla_sm70(*args)
    assert torch.equal(output, expected)
    for row in (0, 1, 2, rows - 1):
        selected = ids[row, : lengths[row].item()]
        keys = kv[selected[(selected >= 0) & (selected < len(kv))].long()].double()
        scores = q[row].double() @ keys.T * 0.0625
        probabilities = torch.cat((scores, sink[:, None].double()), 1).softmax(1)
        reference = probabilities[:, :-1] @ keys[:, :512]
        torch.testing.assert_close(
            output[row].double(), reference, rtol=0.002, atol=0.001
        )
