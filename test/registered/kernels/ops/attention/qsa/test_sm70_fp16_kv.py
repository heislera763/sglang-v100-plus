"""FP16 QSA caches must retain element addressing through native SM70 kernels.

The old wrapper viewed Half caches as bytes and failed startup verification.
Independent indexed attention and live graph updates guard silent byte-stride,
request-row, causal-mask and stale-buffer errors in the cache-format port.
"""

from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=70, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (7, 0),
    reason="native Volta QSA cache-layout coverage",
)


def test_relative_prefill_index_scores_keep_fp16_operands():
    """The newer relative-column entry point must avoid unsupported BF16 MMA."""
    from sglang.srt.layers.attention.qsa import mqa
    from sglang_v100_plus.qsa import scoring_dtype

    torch.manual_seed(723)
    q = torch.randn(9, 4, 128, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(194, 1, 128, device="cuda", dtype=torch.float16) * 0.1
    starts = torch.tensor([0] * 4 + [97] * 5, device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [1, 16, 32, 63, 64, 65, 80, 96, 97], device="cuda", dtype=torch.int32
    )
    original = mqa._scoring_dtype
    with patch.object(
        mqa, "_scoring_dtype", lambda q, k: scoring_dtype(original, q, k)
    ):
        actual = mqa.qsa_mqa_prefill_relative(q, k, starts, starts + lengths, 97)
    columns = starts[:, None] + torch.arange(97, device="cuda")
    selected = k[columns.clamp(max=k.shape[0] - 1), 0].double()
    expected = torch.einsum("rhd,rnd->rnh", q.double(), selected).relu().sum(-1)
    expected /= 128**0.5
    valid = torch.arange(97, device="cuda")[None, :] < lengths[:, None]
    torch.testing.assert_close(
        actual[valid].double(), expected[valid], rtol=1e-4, atol=2e-5
    )


def _inputs(rows, heads, pages=262273, length=2161, topk=2051):
    torch.manual_seed(701 + rows + heads)
    q = torch.randn(rows, heads, 264, device="cuda", dtype=torch.float16)[..., :256]
    k = torch.randn(pages, 1, 256, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    # Three disjoint request maps mix low and high physical cache addresses.
    table = torch.stack(
        [torch.randperm(pages, device="cuda")[:length] for _ in range(3)]
    ).int()
    indices = torch.stack(
        [torch.randperm(length, device="cuda")[:topk] for _ in range(rows)]
    ).int()
    indices[:, ::7] = -1
    indices[:, -1] = length  # A future/out-of-range logical position.
    return q, k, v, table, indices


def _reference(q, k, v, table, requests, indices, lengths):
    expected = torch.zeros(q.shape, dtype=torch.float64)
    for row, (request, length) in enumerate(zip(requests.tolist(), lengths.tolist())):
        selected = indices[row, : min(indices.shape[1], max(0, length))].long()
        selected = selected[(selected >= 0) & (selected < length)]
        slots = table[request, selected].long()
        slots = slots[slots >= 0]
        if slots.numel():
            key = k[slots, 0].double().cpu()
            value = v[slots, 0].double().cpu()
            scores = q[row].double().cpu() @ key.T * 256**-0.5
            expected[row] = scores.softmax(-1) @ value
    return expected


def _check(actual, expected):
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double().cpu(), expected, rtol=0.005, atol=0.002)


@pytest.mark.parametrize("heads", [3, 6])
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_fp16_decode_and_verification_preserve_large_cache_addresses(heads, rows):
    from sglang_v100_plus.kernels.qsa_cuda import sm70_cuda_qsa_decode

    q, k, v, table, indices = _inputs(rows, heads)
    requests = torch.arange(rows, device="cuda", dtype=torch.int32) % 3
    lengths = torch.full((rows,), table.shape[1], device="cuda", dtype=torch.int32)
    actual = sm70_cuda_qsa_decode(q, k, v, table, requests, indices, lengths, 256**-0.5)
    _check(actual, _reference(q, k, v, table, requests, indices, lengths))


@pytest.mark.parametrize("heads", [3, 6])
def test_fp16_decode_graph_refreshes_masks_requests_lengths_and_values(heads):
    from sglang_v100_plus.kernels.qsa_cuda import sm70_cuda_qsa_decode

    q, k, v, table, indices = _inputs(4, heads, pages=4099)
    requests = torch.tensor([2, 0, 1, 2], device="cuda", dtype=torch.int32)
    lengths = torch.full((4,), table.shape[1], device="cuda", dtype=torch.int32)
    for _ in range(3):
        sm70_cuda_qsa_decode(q, k, v, table, requests, indices, lengths, 256**-0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sm70_cuda_qsa_decode(
            q, k, v, table, requests, indices, lengths, 256**-0.5
        )
    for update in range(3):
        q.normal_()
        k.normal_()
        v.normal_()
        requests.copy_(
            torch.tensor([update, 2, 0, 1], device="cuda", dtype=torch.int32)
        )
        lengths.copy_(torch.tensor([0, 1, 65, 2161], device="cuda", dtype=torch.int32))
        indices.copy_(
            torch.randint(0, 2161, indices.shape, device="cuda", dtype=torch.int32)
        )
        indices[0].fill_(-1)
        indices[1].fill_(-1)
        indices[1, 0] = 0
        indices[2, ::3] = -1
        if update == 0:
            indices[2].fill_(-1)  # Positive length with no selected keys.
        graph.replay()
        _check(actual, _reference(q, k, v, table, requests, indices, lengths))
        assert torch.count_nonzero(actual[0]) == 0
        if update == 0:
            assert torch.count_nonzero(actual[2]) == 0


@pytest.mark.parametrize("heads", [3, 6])
@pytest.mark.parametrize("tensor_core", [False, True])
def test_fp16_prefill_preserves_selection_causality_and_empty_rows(heads, tensor_core):
    from sglang_v100_plus.kernels.qsa_cuda import sm70_cuda_qsa_prefill
    from sglang_v100_plus.kernels.qsa_prefill import qsa_masked_prefill

    rows = 129 if tensor_core else 17
    q, k, v, table, indices = _inputs(rows, heads, pages=4099)
    requests = torch.tensor([2], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([2161], device="cuda", dtype=torch.int32)
    indices[-1].fill_(-1)
    if tensor_core:

        def run():
            return qsa_masked_prefill(
                q, k, v, table, requests, indices, 2161, 256**-0.5
            )
    else:

        def run():
            return sm70_cuda_qsa_prefill(
                q, k, v, table, requests, indices, lengths, 256**-0.5
            )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for update in range(2):
        q.normal_()
        k.normal_()
        v.normal_()
        indices.copy_(
            torch.stack(
                [torch.randperm(2161, device="cuda")[:2051] for _ in range(rows)]
            ).int()
        )
        indices[:, ::7] = -1
        indices[-1].fill_(-1)
        graph.replay()
        visible = torch.arange(2161 - rows + 1, 2162, device="cuda", dtype=torch.int32)
        _check(
            actual, _reference(q, k, v, table, requests.expand(rows), indices, visible)
        )
        assert torch.count_nonzero(actual[-1]) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
