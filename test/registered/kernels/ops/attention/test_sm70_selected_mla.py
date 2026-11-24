"""Volta selected-key attention, including head sinks and separate physical banks."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Volta Tensor Core accumulation order",
)
@pytest.mark.parametrize("heads", [4, 8])
def test_selected_key_order_and_sinks(heads):
    from sglang.kernels.ops.attention.dsa.sm70_sparse_mla import (
        _prefill,
        sparse_mla_sm70,
    )

    device = "cuda"
    rng = torch.Generator(device=device).manual_seed(531)
    rows, width, dim = 129, 2048, 576
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Volta Tensor Core accumulation order",
)
@pytest.mark.parametrize("rows, heads", [(1, 8), (65, 16)])
def test_hca_two_banks_share_softmax(rows, heads):
    from sglang.kernels.ops.attention.dsa.sm70_hca import hca_fp16

    rng = torch.Generator(device="cuda").manual_seed(531)

    def samples(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.float16, generator=rng)

    q = samples(rows, heads, 512)
    sink = samples(heads).float()
    banks = [samples(113, 512), samples(269, 512)]
    indices = [
        torch.randint(
            len(bank), (rows, width), device="cuda", dtype=torch.int32, generator=rng
        )
        for bank, width in zip(banks, (67, 257))
    ]
    lengths = [idx.new_full((rows,), idx.shape[1] - 1) for idx in indices]
    for bank, idx in zip(banks, indices):
        idx[:, ::5] = -1
        idx[:, 1::13] = len(bank)
    if rows > 1:
        lengths[0][0] = lengths[1][0] = 0
        lengths[1][1] = 0
    scale = 512**-0.5
    actual = hca_fp16(
        q,
        banks[0],
        indices[0],
        lengths[0],
        banks[1],
        indices[1],
        lengths[1],
        sink,
        scale,
    )
    for row in {0, min(1, rows - 1), rows - 1}:
        parts = []
        for bank, idx, ends in zip(banks, indices, lengths):
            slots = idx[row, : int(ends[row])]
            parts.append(bank[slots[(slots >= 0) & (slots < len(bank))].long()])
        keys = torch.cat(parts).double()
        scores = q[row].double() @ keys.T * scale
        probs = torch.cat((scores, sink[:, None].double()), dim=1).softmax(1)
        torch.testing.assert_close(
            actual[row].double(), probs[:, :-1] @ keys, rtol=0.002, atol=0.002
        )
