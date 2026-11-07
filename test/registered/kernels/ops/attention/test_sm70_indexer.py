"""Software E4M3 boundaries and pool4 rotation parity without FP8 instructions."""

import pytest
import torch

from sglang.kernels.ops.attention.dsa.sm70_indexer import (
    fp8_quantize_sm70,
    kpool_compress_sm70,
    mqa_logits_sm70,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA software FP8 kernels",
)


def reference_quantize(x, round_scale):
    x = x.float()
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def _score_chunks(query, keys, scales, weights, lengths, chunk):
    return torch.cat(
        [
            mqa_logits_sm70(
                query[i : i + chunk],
                keys,
                scales,
                weights[i : i + chunk],
                lengths[i : i + chunk],
            )
            for i in range(0, query.shape[0], chunk)
        ]
    )


def test_scoring_query_batches_preserve_causality_and_head_reduction(subtests):
    """All chunk sizes share one fixture and independent arithmetic oracle."""
    torch.manual_seed(739)
    query, _ = fp8_quantize_sm70(torch.randn(151, 32, 128, device="cuda"))
    keys, scales = fp8_quantize_sm70(torch.randn(521, 128, device="cuda"))
    weights = torch.randn(151, 32, device="cuda") * 0.01
    lengths = torch.arange(151, device="cuda", dtype=torch.int32) // 4 + 485
    lengths[0] = 0
    reference_dots = torch.einsum(
        "thd,kd->thk", query.cpu().float(), keys.cpu().float()
    )
    expected = (reference_dots.relu() * weights.cpu()[:, :, None]).sum(1)
    expected *= scales.cpu().reshape(1, -1)
    future = torch.arange(keys.shape[0])[None, :] >= lengths.cpu()[:, None]
    expected.masked_fill_(future, float("-inf"))
    baseline = _score_chunks(query, keys, scales, weights, lengths, 32).cpu()
    for chunk in (4, 8, 16, 32, 64, 128):
        with subtests.test(chunk=chunk):
            actual = _score_chunks(query, keys, scales, weights, lengths, chunk).cpu()
            assert torch.equal(torch.isneginf(actual), future)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
            assert torch.equal(actual, baseline)


@pytest.mark.parametrize("batched", [False, True])
def test_scoring_private_buffer_reuse_is_exact_and_preserves_inputs(batched):
    torch.manual_seed(741)
    rows, heads, history = (4, 32, 521) if batched else (32, 32, 521)
    query, _ = fp8_quantize_sm70(torch.randn(rows, heads, 128, device="cuda"))
    keys = torch.randn(
        *((rows, history, 128) if batched else (history, 128)),
        device="cuda",
        dtype=torch.float16,
    )
    scales = torch.rand(*((rows, history) if batched else (history, 1)), device="cuda")
    weights = torch.randn(rows, heads, device="cuda")
    lengths = torch.arange(rows, device="cuda", dtype=torch.int32) + history - rows
    lengths[0] = 0
    inputs = (query, keys, scales, weights, lengths)
    saved = [x.clone() for x in inputs]
    if batched:
        dots = torch.bmm(query.half(), keys.transpose(1, 2), out_dtype=torch.float32)
    else:
        dots = torch.mm(
            query.half().reshape(-1, 128), keys.half().T, out_dtype=torch.float32
        ).reshape(rows, heads, -1)
    expected = (dots.relu() * weights.float().unsqueeze(-1)).sum(1)
    expected *= scales if batched else scales.reshape(1, -1)
    expected.masked_fill_(
        torch.arange(history, device="cuda")[None, :] >= lengths[:, None],
        float("-inf"),
    )
    actual = mqa_logits_sm70(*inputs)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    for original, snapshot in zip(inputs, saved):
        dtype = torch.uint8 if original.dtype == torch.float8_e4m3fn else original.dtype
        assert torch.equal(original.view(dtype), snapshot.view(dtype))


def test_long_history_scoring_batches_preserve_exact_logits(subtests):
    # Smaller history-dependent query batches must preserve the same per-row
    # GEMM and head reduction at the384Ki-token pool4 history size.
    torch.manual_seed(745)
    rows, heads, history = 32, 32, 98304
    query = torch.randn(rows, heads, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(history, 128, device="cuda", dtype=torch.float16)
    scales = torch.rand(history, 1, device="cuda")
    weights = torch.randn(rows, heads, device="cuda")
    lengths = torch.arange(rows, device="cuda", dtype=torch.int32) + history - rows
    expected = mqa_logits_sm70(query, keys, scales, weights, lengths)
    for chunk in (4, 8, 16):
        with subtests.test(chunk=chunk):
            actual = _score_chunks(query, keys, scales, weights, lengths, chunk)
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_scoring_workspace_does_not_duplicate_full_head_scores():
    # Each redundant full-size intermediate adds64MiB for this shape.
    # Warm the same GEMM first so a lazy library workspace is not the contract.
    rows, heads, history = 32, 32, 16384
    query = torch.ones(rows, heads, 128, device="cuda", dtype=torch.float16)
    keys = torch.ones(history, 128, device="cuda", dtype=torch.float16)
    scales = torch.ones(history, 1, device="cuda")
    weights = torch.ones(rows, heads, device="cuda")
    lengths = torch.full((rows,), history, device="cuda", dtype=torch.int32)
    mqa_logits_sm70(query, keys, scales, weights, lengths)
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = mqa_logits_sm70(query, keys, scales, weights, lengths)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - live
    score_bytes = rows * heads * history * 4
    assert peak < 2 * score_bytes, peak
    assert torch.equal(result, torch.full_like(result, 128 * heads))


@pytest.mark.parametrize("round_scale", [False, True])
def test_quantize_rounding_boundaries_and_strides(round_scale):
    # Independent Torch conversion covers every finite FP8 midpoint, ties to
    # even, subnormals, signed zero, and the 448 endpoint. Fixed maxima make
    # the unrounded scale exactly one; near-midpoint values guard off-by-one
    # encodings that random numerical tolerance tests would miss.
    values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mids = (values[:-1] + values[1:]) / 2
    delta = torch.tensor([-1e-5, 0, 1e-5])
    boundary = (mids[:, None] + delta).flatten()
    boundary = torch.cat((boundary, -boundary, torch.tensor([0.0, -0.0])))
    rows = torch.zeros((boundary.numel(), 256), dtype=torch.float32)
    rows[:, 0] = boundary
    rows[:, 2] = 448
    x = rows.cuda()[:, ::2]
    expected, scale = reference_quantize(rows[:, ::2], round_scale)
    actual, actual_scale = fp8_quantize_sm70(x, round_scale)
    torch.testing.assert_close(actual_scale.cpu(), scale, rtol=0, atol=0)
    assert torch.equal(actual.view(torch.uint8).cpu(), expected.view(torch.uint8))

    # Real query layout and data-dependent (usually non-power-of-two) scales.
    query = torch.randn(3, 32, 128, generator=torch.Generator().manual_seed(736)).half()
    query[0, 0] = 0
    query = query.cuda()
    # Use the prior CUDA Torch arithmetic for data-dependent scales: CPU and
    # GPU division can differ at an exact FP8 midpoint even before this change.
    expected, scale = reference_quantize(query, round_scale)
    actual, actual_scale = fp8_quantize_sm70(query, round_scale)
    torch.testing.assert_close(actual_scale, scale, rtol=1e-6, atol=1e-8)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("round_scale", [False, True])
@pytest.mark.parametrize("rows", [1, 65])
def test_pool4_matches_dense_walsh_reference(rows, round_scale):
    # A dense +/-1 Walsh matrix checks the butterfly orientation independently.
    # Noncontiguous score/key/APE inputs defend runtime slice semantics.
    generator = torch.Generator().manual_seed(735)
    key_storage = torch.randn(rows, 4, 256, generator=generator).bfloat16()
    score_storage = torch.randn(rows, 4, 256, generator=generator)
    ape_storage = torch.randn(4, 256, generator=generator)
    keys, scores, ape = (
        key_storage[:, :, ::2],
        score_storage[:, :, ::2],
        ape_storage[:, ::2],
    )
    walsh = torch.tensor(
        [[(-1) ** ((r & c).bit_count() % 2) for c in range(128)] for r in range(128)],
        dtype=torch.float32,
    )
    pooled = ((scores + ape).softmax(1) * keys.float()).sum(1).bfloat16().float()
    rotated = ((pooled @ walsh) * 128**-0.5).bfloat16().float()
    expected, scale = reference_quantize(rotated, round_scale)
    actual, actual_scale = kpool_compress_sm70(
        key_storage.cuda()[:, :, ::2],
        score_storage.cuda()[:, :, ::2],
        ape_storage.cuda()[:, ::2],
        round_scale,
    )
    torch.testing.assert_close(actual_scale.cpu(), scale, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=0, atol=0)


def test_quantization_and_pool_graph_replay_use_new_inputs():
    # A captured primitive must read updated input storage, not freeze results.
    q = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(1, 4, 128, device="cuda").bfloat16()
    scores = torch.randn(1, 4, 128, device="cuda")
    ape = torch.randn(4, 128, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fp8_quantize_sm70(q)
            kpool_compress_sm70(keys, scores, ape)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q8, qs = fp8_quantize_sm70(q)
        k8, ks = kpool_compress_sm70(keys, scores, ape)
    for _ in range(2):
        q.copy_(torch.randn_like(q))
        keys.copy_(torch.randn_like(keys))
        scores.copy_(torch.randn_like(scores))
        graph.replay()
        expected_q, expected_qs = fp8_quantize_sm70(q)
        expected_k, expected_ks = kpool_compress_sm70(keys, scores, ape)
        torch.testing.assert_close(qs, expected_qs, rtol=0, atol=0)
        torch.testing.assert_close(ks, expected_ks, rtol=0, atol=0)
        assert torch.equal(q8.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(k8.view(torch.uint8), expected_k.view(torch.uint8))


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize(
    "token_page_size,index_page_size", [(64, 64), (64, 16), (256, 64)]
)
def test_device_pool4_replays_lengths_requests_and_page_boundaries(
    use_graph, token_page_size, index_page_size
):
    from sglang.kernels.ops.attention.dsa.sm70_pool4_decode import pool4_decode_sm70

    torch.manual_seed(737)
    scale_offset = index_page_size * 128
    cache = torch.zeros(40, index_page_size * 132, device="cuda", dtype=torch.uint8)
    finite_codes = (
        torch.cat((torch.arange(127), torch.arange(128, 255))).to(torch.uint8).cuda()
    )
    cache[:, :scale_offset] = finite_codes.repeat((40 * scale_offset + 253) // 254)[
        : 40 * scale_offset
    ].reshape(40, scale_offset)
    cache[:, scale_offset:].view(torch.float32).fill_(0.01)
    tail_k = torch.randn(3, 4, 128, device="cuda").bfloat16()
    tail_g = torch.randn_like(tail_k)
    table = torch.zeros(3, 512, device="cuda", dtype=torch.int32)
    for req, start in ((1, 1), (2, 17)):
        pages = torch.randperm(512 // token_page_size, device="cuda") + start
        table[req] = (
            pages[:, None] * token_page_size
            + torch.arange(token_page_size, device="cuda")
        ).flatten()
    reference_cache, reference_k, reference_g = (
        cache.clone(),
        tail_k.clone(),
        tail_g.clone(),
    )
    keys = torch.zeros(3, 128, device="cuda", dtype=torch.float16)
    gates = torch.zeros_like(keys)
    ape = torch.randn(4, 128, device="cuda")
    reqs = torch.tensor([1, 2, -1], device="cuda", dtype=torch.int64)
    lengths = torch.zeros(3, device="cuda", dtype=torch.int32)

    def run():
        return pool4_decode_sm70(
            keys,
            gates,
            ape,
            tail_k,
            tail_g,
            cache,
            table,
            reqs,
            lengths,
            token_page_size=token_page_size,
        )

    for _ in range(3):
        run()
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_keys, actual_scales, actual_pools = run()

    for step in range(12):
        order = [1, 2, -1] if step % 2 == 0 else [2, 1, -1]
        seqs = [
            step + 1 if req == 1 else step + 255 if req == 2 else 0 for req in order
        ]
        keys.normal_()
        gates.normal_()
        reqs.copy_(torch.tensor(order, device="cuda"))
        lengths.copy_(torch.tensor(seqs, device="cuda", dtype=torch.int32))
        if use_graph:
            graph.replay()
        else:
            actual_keys, actual_scales, actual_pools = run()
        for row, (req, length) in enumerate(zip(order, seqs)):
            if req < 0:
                assert actual_pools[row] == 0
                continue
            prefix = length - 1
            old_tail = prefix % 4
            pos = torch.arange(prefix - old_tail, prefix, device="cuda") % 4
            if length % 4 == 0:
                assembled_k = torch.cat(
                    (reference_k[req, pos], keys[row : row + 1].bfloat16())
                )
                assembled_g = torch.cat(
                    (reference_g[req, pos].float(), gates[row : row + 1].float())
                )
                q, scale = kpool_compress_sm70(
                    assembled_k[None], assembled_g[None], ape
                )
                pool_id = prefix // 4
                page = (
                    int(table[req, pool_id // index_page_size * (index_page_size * 4)])
                    // token_page_size
                )
                offset = pool_id % index_page_size
                reference_cache[page, offset * 128 : (offset + 1) * 128] = q.view(
                    torch.uint8
                )[0]
                reference_cache[
                    page, scale_offset + offset * 4 : scale_offset + (offset + 1) * 4
                ] = scale.view(torch.uint8).flatten()
            else:
                reference_k[req, prefix % 4] = keys[row].bfloat16()
                reference_g[req, prefix % 4] = gates[row].bfloat16()
            count = length // 4
            ids = torch.arange(count, device="cuda")
            pages = (
                table[req, ids // index_page_size * (index_page_size * 4)].long()
                // token_page_size
            )
            columns = (
                ids[:, None] % index_page_size * 128
                + torch.arange(128, device="cuda")[None, :]
            )
            expected_keys = (
                reference_cache[pages[:, None], columns]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .half()
            )
            expected_scales = reference_cache[:, scale_offset:].view(torch.float32)[
                pages, ids % index_page_size
            ]
            assert actual_pools[row] == count
            torch.testing.assert_close(
                actual_keys[row, :count], expected_keys, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_scales[row, :count], expected_scales, rtol=0, atol=0
            )
        assert torch.equal(cache, reference_cache)
        assert torch.equal(tail_k, reference_k)
        assert torch.equal(tail_g, reference_g)


@pytest.mark.parametrize("num_draft_tokens", [2, 6, 8])
@pytest.mark.parametrize("round_scale", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("use_effective", [False, True])
@pytest.mark.parametrize(
    "token_page_size,index_page_size", [(64, 64), (64, 16), (256, 64)]
)
def test_pool4_spec_chain_rejection_and_page_boundary(
    num_draft_tokens,
    round_scale,
    use_graph,
    use_effective,
    token_page_size,
    index_page_size,
):
    """Verify -> partial commit -> verify must replace rejected future pools.

    Shuffled request pages and rings defend logical/physical mapping. Two
    requests cross different pool/page boundaries, while a padded row must
    leave its storage untouched. Graph replay reads fresh plans and candidates.
    """
    from sglang.kernels.ops.attention.dsa.sm70_pool4_decode import pool4_spec_sm70

    torch.manual_seed(741)
    n, batch = num_draft_tokens, 3
    scale_offset = index_page_size * 128
    cache = torch.zeros(40, index_page_size * 132, device="cuda", dtype=torch.uint8)
    tail_k = torch.randn(3, 4 + n, 128, device="cuda").bfloat16()
    tail_g = torch.randn_like(tail_k)
    table = torch.zeros(3, 512, device="cuda", dtype=torch.int32)
    for req, first in ((1, 1), (2, 17)):
        pages = torch.randperm(512 // token_page_size, device="cuda") + first
        table[req] = (
            pages[:, None] * token_page_size
            + torch.arange(token_page_size, device="cuda")
        ).flatten()
    reference_cache, reference_k, reference_g = (
        cache.clone(),
        tail_k.clone(),
        tail_g.clone(),
    )
    keys = torch.zeros(batch * n, 128, device="cuda", dtype=torch.bfloat16)
    gates = torch.zeros_like(keys)
    ape = torch.randn(4, 128, device="cuda")
    reqs = torch.tensor([1, 2, -1], device="cuda", dtype=torch.int64)
    starts = torch.zeros(batch, device="cuda", dtype=torch.int32)
    tail_starts = torch.zeros_like(starts)
    locations = torch.zeros(batch, (n + 3) // 4, device="cuda", dtype=torch.int64)
    out_locations = torch.zeros(batch * n, device="cuda", dtype=torch.int64)
    query_reqs = reqs.repeat_interleave(n)
    lengths = torch.zeros(batch * n, device="cuda", dtype=torch.int32)
    effective = torch.full((batch,), n, device="cuda", dtype=torch.int32)

    def run():
        return pool4_spec_sm70(
            keys,
            gates,
            ape,
            tail_k,
            tail_g,
            cache,
            table,
            reqs,
            starts,
            tail_starts,
            locations,
            out_locations,
            query_reqs,
            lengths,
            effective if use_effective else None,
            round_scale,
            token_page_size=token_page_size,
        )

    actual_keys, actual_scales, actual_pools = run()
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_keys, actual_scales, actual_pools = run()
    out_locations[:-n] = 1
    committed = {1: 254, 2: 1}
    for step in range(4):
        order = [1, 2, -1] if step % 2 == 0 else [2, 1, -1]
        counts = [n if step % 2 == 0 else 1, n - 1 if step % 2 else n, 0]
        if not use_effective:
            counts = [n, n, 0]
        keys.normal_()
        gates.normal_()
        reqs.copy_(torch.tensor(order, device="cuda"))
        query_reqs.copy_(reqs.repeat_interleave(n))
        prefixes = [committed.get(req, 0) for req in order]
        starts.copy_(torch.tensor(prefixes, device="cuda", dtype=torch.int32))
        tail_starts.copy_(starts - starts % 4)
        effective.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        for row, (req, prefix) in enumerate(zip(order, prefixes)):
            lengths[row * n : (row + 1) * n] = (
                torch.arange(prefix + 1, prefix + n + 1, device="cuda")
                if req >= 0
                else 0
            )
            if req < 0:
                continue
            for p in range((n + 3) // 4):
                pool_id = prefix // 4 + p
                locations[row, p] = (
                    table[
                        req, pool_id // index_page_size * (index_page_size * 4)
                    ].long()
                    // token_page_size
                    * index_page_size
                    + pool_id % index_page_size
                )
            for i in range(n):
                pos = (prefix + i) % (4 + n)
                reference_k[req, pos] = keys[row * n + i]
                reference_g[req, pos] = gates[row * n + i]
            for pool_id in range(prefix // 4, (prefix + counts[row]) // 4):
                positions = (pool_id * 4 + torch.arange(4, device="cuda")) % (4 + n)
                q, scale = kpool_compress_sm70(
                    reference_k[req, positions][None],
                    reference_g[req, positions][None],
                    ape,
                    round_scale,
                )
                page = (
                    int(table[req, pool_id // index_page_size * (index_page_size * 4)])
                    // token_page_size
                )
                offset = pool_id % index_page_size
                reference_cache[page, offset * 128 : (offset + 1) * 128] = q.view(
                    torch.uint8
                )[0]
                reference_cache[
                    page, scale_offset + offset * 4 : scale_offset + (offset + 1) * 4
                ] = scale.view(torch.uint8).flatten()
        if use_graph:
            graph.replay()
        else:
            actual_keys, actual_scales, actual_pools = run()
        assert torch.equal(cache, reference_cache)
        assert torch.equal(tail_k, reference_k)
        assert torch.equal(tail_g, reference_g)
        for row, req in enumerate(order):
            for i in range(n):
                qrow = row * n + i
                count = (prefixes[row] + i + 1) // 4 if req >= 0 else 0
                assert actual_pools[qrow] == count
                ids = torch.arange(count, device="cuda")
                pages = (
                    table[req, ids // index_page_size * (index_page_size * 4)].long()
                    // token_page_size
                    if req >= 0
                    else ids
                )
                columns = (
                    ids[:, None] % index_page_size * 128
                    + torch.arange(128, device="cuda")[None, :]
                )
                expected_k = (
                    reference_cache[pages[:, None], columns]
                    .contiguous()
                    .view(torch.float8_e4m3fn)
                    .half()
                )
                expected_s = reference_cache[:, scale_offset:].view(torch.float32)[
                    pages, ids % index_page_size
                ]
                assert torch.equal(actual_keys[qrow, :count], expected_k)
                assert torch.equal(actual_scales[qrow, :count], expected_s)
            if req >= 0:
                # Target verification staged the full window; rejecting all
                # proposed drafts keeps only the bonus-token prefix.
                committed[req] += 1 if step % 2 == 0 else counts[row]


@pytest.mark.parametrize("token_page_size,index_page_size", [(64, 16), (256, 64)])
def test_prefill_cache_uses_compact_pooled_token_locations(
    token_page_size, index_page_size
):
    from sglang_v100_plus.glm_dsa import (
        pooled_locations,
        read_pooled_cache,
        write_pooled_cache,
    )

    torch.manual_seed(742)
    pages = torch.randperm(8, device="cuda") + 1
    table = (
        pages[:, None] * token_page_size + torch.arange(token_page_size, device="cuda")
    ).flatten()
    ids = torch.arange(table.numel() // 4, device="cuda")
    # In the compact upstream layout a pool's physical slot is the first of
    # its four token slots divided by four. This reference does not group IDs
    # using the adapter's page-start formula.
    expected_locations = table[ids * 4].long() // 4
    locations = pooled_locations(
        table, ids, token_page_size=token_page_size, index_page_size=index_page_size
    )
    assert torch.equal(locations, expected_locations)
    codes = torch.randint(0, 127, (ids.numel(), 128), device="cuda", dtype=torch.uint8)
    keys = codes.view(torch.float8_e4m3fn)
    scales = torch.rand(ids.numel(), device="cuda", dtype=torch.float32)
    cache = torch.zeros(10, index_page_size * 132, device="cuda", dtype=torch.uint8)
    expected = cache.clone()
    for i, loc in enumerate(expected_locations.tolist()):
        page, offset = divmod(loc, index_page_size)
        expected[page, offset * 128 : (offset + 1) * 128] = codes[i]
        expected[
            page,
            index_page_size * 128 + offset * 4 : index_page_size * 128
            + (offset + 1) * 4,
        ] = scales[i : i + 1].view(torch.uint8)
    write_pooled_cache(cache, locations, keys, scales)
    assert torch.equal(cache, expected)
    actual_keys, actual_scales = read_pooled_cache(cache, locations.flip(0))
    assert torch.equal(actual_keys.view(torch.uint8), codes.flip(0))
    assert torch.equal(actual_scales, scales.flip(0))


@pytest.mark.parametrize("rows,capacity", [(1, 67), (3, 521), (4, 65601), (8, 521)])
def test_decode_scoring_live_bounds_graph_refresh_and_fp64_reference(rows, capacity):
    """Tile skipping must preserve ragged causal rows, graph refresh and FP32 scores."""
    from sglang.kernels.ops.attention.dsa.sm70_indexer_decode import (
        mqa_logits_decode_sm70,
    )

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core index scoring")
    torch.manual_seed(749)
    query, query_scale = fp8_quantize_sm70(
        torch.randn(rows, 32, 128, device="cuda") * 0.2
    )
    if rows == 3:
        # Also exercise non-E4M3-representable FP16 query mantissas.
        query = (torch.randn(rows, 32, 128, device="cuda") * 60).half()
    encoded, scales = fp8_quantize_sm70(
        torch.randn(rows, capacity, 128, device="cuda") * 0.2
    )
    keys = encoded.half()
    scales = scales.squeeze(-1)
    weights = torch.randn(rows, 32, device="cuda") * 0.1 * query_scale.squeeze(-1)
    lengths = torch.full((rows,), min(capacity, 2051), device="cuda", dtype=torch.int32)
    inputs = (query, keys, scales, weights, lengths)
    mqa_logits_decode_sm70(*inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = mqa_logits_decode_sm70(*inputs)
    ptr = output.data_ptr()
    # Cross both sides of the tile boundary and the final reserved column.
    for pattern in (
        [0, 1, 63, 64],
        [65, capacity - 1, capacity, capacity + 7],
        [-1, 0, 1, 65],
    ):
        lengths.copy_(
            torch.tensor((pattern * 2)[:rows], device="cuda", dtype=torch.int32)
        )
        keys.mul_(0.5)
        weights.mul_(-0.75)
        query.copy_(query.half().neg().to(query.dtype))
        saved = [x.clone() for x in inputs]
        graph.replay()
        eager = mqa_logits_decode_sm70(*inputs)
        assert output.data_ptr() == ptr
        assert torch.equal(output.view(torch.int32), eager.view(torch.int32))
        q64, k64 = query.cpu().double(), keys.cpu().double()
        expected = (
            torch.bmm(q64, k64.transpose(1, 2)).relu()
            * weights.cpu().double()[:, :, None]
        ).sum(1)
        expected *= scales.cpu().double()
        future = torch.arange(capacity)[None, :] >= lengths.cpu()[:, None]
        expected.masked_fill_(future, float("-inf"))
        assert torch.equal(torch.isneginf(output.cpu()), future)
        torch.testing.assert_close(
            output.cpu().double(), expected, rtol=2e-5, atol=2e-6
        )
        for value, snapshot in zip(inputs, saved):
            dtype = torch.uint8 if value.dtype == torch.float8_e4m3fn else value.dtype
            assert torch.equal(value.view(dtype), snapshot.view(dtype))


@pytest.mark.parametrize("capacity", [67, 262400])
def test_pool4_live_gather_poisoned_future_and_shrinking_graph(capacity):
    """Reserved-context scratch is not initialized outside live pool counts."""
    from sglang.kernels.ops.attention.dsa.sm70_indexer_decode import (
        mqa_logits_decode_sm70,
    )
    from sglang.kernels.ops.attention.dsa.sm70_pool4_decode import pool4_decode_sm70

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core index scoring")
    torch.manual_seed(752)
    rows, page = 3, 64
    pages = (capacity + page - 1) // page
    cache = torch.zeros(3 * pages, page * 132, device="cuda", dtype=torch.uint8)
    cache[:, : page * 128] = torch.arange(page * 128, device="cuda").byte() % 125
    cache[:, page * 128 :].view(torch.float32).fill_(0.125)
    table = torch.empty(3, capacity * 4, device="cuda", dtype=torch.int32)
    for req in range(3):
        table[req] = req * pages * 256 + torch.arange(capacity * 4, device="cuda")
    reqs = torch.tensor([2, 0, -1], device="cuda", dtype=torch.int64)
    lengths = torch.full((rows,), 7, device="cuda", dtype=torch.int32)
    keys = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128, device="cuda")
    tails = torch.zeros(3, 4, 128, device="cuda", dtype=torch.bfloat16)
    tail_gates = torch.zeros_like(tails)
    query = torch.randn(rows, 32, 128, device="cuda", dtype=torch.float16) * 0.1
    weights = torch.randn(rows, 32, device="cuda") * 0.01

    def run():
        gathered, scales, counts = pool4_decode_sm70(
            keys,
            gates,
            ape,
            tails,
            tail_gates,
            cache,
            table,
            reqs,
            lengths,
            token_page_size=256,
        )
        logits = mqa_logits_decode_sm70(query, gathered, scales, weights, counts)
        return gathered, scales, counts, logits

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gathered, scales, counts, logits = run()
    for live in ([capacity, capacity - 1, 0], [1, 17, 0], [0, 0, 0], [65, capacity, 0]):
        live = [min(capacity, v) for v in live]
        lengths.copy_(torch.tensor([4 * v + 1 for v in live], device="cuda"))
        gathered.fill_(float("nan"))
        scales.fill_(float("nan"))
        graph.replay()
        assert counts.tolist() == live
        for row, count in enumerate(live):
            assert torch.isneginf(logits[row, count:]).all()
            assert torch.isfinite(logits[row, :count]).all()
            # Poison remains outside live history: neither gather nor score
            # needs to touch those reserved rows, even after a shrinking replay.
            assert torch.isnan(gathered[row, count:]).all()
            assert torch.isnan(scales[row, count:]).all()
            if count:
                sample = torch.tensor(sorted({0, count // 2, count - 1}), device="cuda")
                physical = table[reqs[row], sample * 4].long()
                encoded = cache[physical // 256, : page * 128].view(-1, page, 128)
                expected_k = (
                    encoded[torch.arange(sample.numel(), device="cuda"), sample % page]
                    .contiguous()
                    .view(torch.float8_e4m3fn)
                    .half()
                )
                torch.testing.assert_close(
                    gathered[row, sample], expected_k, rtol=0, atol=0
                )
                expected = (
                    (query[row].double() @ expected_k.double().T).relu()
                    * weights[row, :, None].double()
                ).sum(0) * 0.125
                torch.testing.assert_close(
                    logits[row, sample].double(), expected, rtol=2e-5, atol=2e-6
                )


@pytest.mark.parametrize(
    "bad_input", ["key_alignment", "query_alignment", "length_dtype"]
)
def test_decode_scoring_rejects_unsafe_inputs(bad_input):
    """Contiguous storage-offset views can violate generated vector-load alignment."""
    from sglang.kernels.ops.attention.dsa.sm70_indexer_decode import (
        mqa_logits_decode_sm70,
    )

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core index scoring")
    query = torch.zeros(1, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.zeros(1, 67, 128, device="cuda", dtype=torch.float16)
    scales = torch.ones(1, 67, device="cuda")
    weights = torch.ones(1, 32, device="cuda")
    lengths = torch.ones(1, device="cuda", dtype=torch.int32)
    if bad_input == "key_alignment":
        keys = torch.zeros(keys.numel() + 1, device="cuda", dtype=keys.dtype)[
            1:
        ].view_as(keys)
    elif bad_input == "query_alignment":
        query = torch.zeros(query.numel() + 1, device="cuda", dtype=query.dtype)[
            1:
        ].view_as(query)
    else:
        lengths = lengths.long()
    with pytest.raises(ValueError):
        mqa_logits_decode_sm70(query, keys, scales, weights, lengths)


@pytest.mark.parametrize(
    "rows,capacity", [(1, 1), (5, 65), (32, 513), (128, 4099), (5, 0), (0, 64)]
)
def test_shared_prefill_scoring_masks_tails_and_matches_fp64(rows, capacity):
    """Shared-bank addressing must preserve rows, causal tails and FP32 math."""
    from sglang.kernels.ops.attention.dsa.sm70_indexer_prefill import (
        mqa_logits_prefill_sm70,
    )

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta shared-history index scoring")
    torch.manual_seed(753)
    query = torch.randn(rows, 32, 128, device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    keys = (
        torch.randn(capacity, 128, device="cuda", dtype=torch.float16)
        .to(torch.float8_e4m3fn)
        .half()
    )
    scales = torch.rand(capacity, device="cuda") * 0.01
    weights = torch.randn(rows, 32, device="cuda")
    lengths = torch.linspace(-1, capacity + 7, rows, device="cuda").int()
    values = (query, keys, scales, weights, lengths)
    saved = [x.clone() for x in values]
    actual = mqa_logits_prefill_sm70(*values)
    assert actual.shape == (rows, capacity)
    if not rows or not capacity:
        return
    expected = (
        torch.einsum("rhd,kd->rhk", query.cpu().double(), keys.cpu().double()).relu()
        * weights.cpu().double()[:, :, None]
    ).sum(1) * scales.cpu().double()[None, :]
    future = torch.arange(capacity)[None, :] >= lengths.cpu()[:, None]
    expected.masked_fill_(future, -torch.inf)
    assert torch.equal(torch.isneginf(actual.cpu()), future)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-5, atol=2e-6)
    for value, snapshot in zip(values, saved):
        dtype = torch.uint8 if value.dtype == torch.float8_e4m3fn else value.dtype
        assert torch.equal(value.view(dtype), snapshot.view(dtype))


def test_shared_prefill_scoring_changed_graph_and_output_canaries():
    """Replay must refresh shared keys/weights/causality without writing guards."""
    from sglang.kernels.ops.attention.dsa.sm70_indexer_prefill import (
        _index_score_prefill_kernel,
        mqa_logits_prefill_sm70,
    )

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta shared-history index scoring")
    rows, capacity = 5, 513
    query = torch.randn(rows, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(capacity, 128, device="cuda", dtype=torch.float16)
    scales = torch.rand(capacity, device="cuda") * 0.01
    weights = torch.randn(rows, 32, device="cuda")
    lengths = torch.full((rows,), capacity, device="cuda", dtype=torch.int32)
    guard = torch.full((rows * capacity + 32,), 1234567.0, device="cuda")
    output = guard[16:-16].view(rows, capacity)
    kernel = _index_score_prefill_kernel()
    kernel(query, keys, scales, weights, lengths, output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(query, keys, scales, weights, lengths, output)
    for pattern in (
        [0, 0, 0, 0, 0],
        [capacity, 0, 65, 1, capacity],
        [1, capacity, 0, 65, 1],
        [63, 64, 65, 0, capacity],
        [capacity] * rows,
    ):
        query.mul_(-0.9)
        keys.mul_(0.95)
        scales.mul_(0.8)
        weights.mul_(-0.75)
        lengths.copy_(torch.tensor(pattern, device="cuda", dtype=torch.int32))
        graph.replay()
        eager = mqa_logits_prefill_sm70(query, keys, scales, weights, lengths)
        assert torch.equal(output.view(torch.int32), eager.view(torch.int32))
        # Adjacent queries can have opposite causal limits; an odd last query
        # must not read its absent partner or inherit another row's reduction.
        expected = (
            torch.einsum("rhd,kd->rhk", query.double(), keys.double()).relu()
            * weights.double()[:, :, None]
        ).sum(1) * scales.double()[None, :]
        expected.masked_fill_(
            torch.arange(capacity, device="cuda")[None, :] >= lengths[:, None],
            -torch.inf,
        )
        torch.testing.assert_close(output.double(), expected, rtol=2e-5, atol=2e-6)
        assert torch.all(guard[:16] == 1234567).item()
        assert torch.all(guard[-16:] == 1234567).item()


@pytest.mark.parametrize("bad", ["query_alignment", "key_alignment", "length_dtype"])
def test_shared_prefill_scoring_rejects_unsafe_views(bad):
    """Vector-copy addresses and causal integer dtypes are kernel preconditions."""
    from sglang.kernels.ops.attention.dsa.sm70_indexer_prefill import (
        mqa_logits_prefill_sm70,
    )

    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta shared-history index scoring")
    query = torch.zeros(3, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.zeros(65, 128, device="cuda", dtype=torch.float16)
    scales = torch.ones(65, device="cuda")
    weights = torch.ones(3, 32, device="cuda")
    lengths = torch.ones(3, device="cuda", dtype=torch.int32)
    if bad == "query_alignment":
        query = torch.zeros(query.numel() + 1, device="cuda", dtype=query.dtype)[
            1:
        ].view_as(query)
    elif bad == "key_alignment":
        keys = torch.zeros(keys.numel() + 1, device="cuda", dtype=keys.dtype)[
            1:
        ].view_as(keys)
    else:
        lengths = lengths.long()
    with pytest.raises(ValueError):
        mqa_logits_prefill_sm70(query, keys, scales, weights, lengths)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
