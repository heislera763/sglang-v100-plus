"""Canonical FA adapters preserve padded rows and sequence-sized snapshots."""

import math
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="canonical SM70 FA adapter coverage",
)


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("live,total", [(7, 8), (34, 36)])
def test_sparse_adapter_restores_prefill_padding(heads, live, total):
    """A unit batch axis hid real rows from core's zero-tail restoration.

    Padded prefill then returned fewer rows than the mHC residual contained.
    Real rows must be byte-identical to an unpadded canonical attention call.
    """
    from sglang_v100_plus.glm_dsa import SM70SparseAttnBackend

    from sglang.kernels.ops.attention.v100 import sparse_attention_v100
    from sglang.srt.layers.radix_attention import _attention_on_real_rows
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    torch.manual_seed(531)
    q = torch.randn(total, heads, 512, device="cuda", dtype=torch.float16) * 0.1
    kv = torch.randn(128, 512, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.randint(0, 128, (total, 128), device="cuda", dtype=torch.int32)
    rope = torch.empty(total, heads, 0, device="cuda", dtype=torch.float16)
    layer = SimpleNamespace(
        qk_head_dim=512, v_head_dim=512, tp_q_head_num=heads, is_cross_attention=False
    )
    locations = torch.arange(total, device="cuda")
    positions = locations.clone()
    batch = SimpleNamespace(
        out_cache_loc=locations,
        positions=positions,
        _attn_output=None,
        mha_return_lse=False,
    )

    class Backend:
        attn_dcp_size = 1

        def forward(self, q, k, v, layer, batch, save, **kwargs):
            return SM70SparseAttnBackend._forward_v100_sparse(
                self, q, kwargs["q_rope"], kv, kwargs["topk_indices"], 512**-0.5, 512
            )

    backend = Backend()

    def run():
        with forward_context(ForwardContext(attn_backend=backend)):
            return _attention_on_real_rows(
                layer, live, q, None, None, batch, False, q_rope=rope, topk_indices=ids
            )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for replay in range(2):
        if replay:
            q.mul_(0.625)
            kv.mul_(0.75)
            ids.add_(17).remainder_(128)
        graph.replay()
        expected = sparse_attention_v100(q[:live], kv, ids[:live], 512**-0.5)
        torch.testing.assert_close(out[:live], expected, rtol=0, atol=0)
        assert out.shape == (total, heads, 512)
        assert torch.count_nonzero(out[live:]) == 0
        assert batch.out_cache_loc is locations and batch.positions is positions
        assert batch._attn_output is None


@pytest.mark.parametrize("heads", [8, 16])
def test_kda_adapter_ignores_unused_snapshot_index_capacity(heads):
    """An eager-batch-capacity table must not reject a one-sequence graph."""
    from sglang_v100_plus.attention import V100KDAKernel

    from sglang.kernels.ops.attention.v100 import recurrent_attention_v100

    torch.manual_seed(531)
    shape = (4, heads, 128)
    q, k, v, gate = [
        torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(4)
    ]
    beta = torch.randn(shape[:2], device="cuda", dtype=torch.float16)
    rate = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads * 128, device="cuda")
    initial = (
        torch.randn(2, heads, 128, 128, device="cuda", dtype=torch.float16) * 0.025
    )
    state = initial.clone()
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    snapshots = torch.full(
        (2, 4, heads, 128, 128), 123.0, device="cuda", dtype=torch.float16
    )
    table = torch.tensor([1, 123456, 123456, 123456], device="cuda", dtype=torch.int32)
    kernel = V100KDAKernel()

    def run():
        return kernel.target_verify(
            rate,
            bias,
            q,
            k,
            v,
            gate,
            beta,
            ssm_states=state,
            cache_indices=slots,
            query_start_loc=offsets,
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=table,
            cache_steps=4,
            retrieve_parent_token=None,
            lower_bound=-5.0,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for replay in range(2):
        if replay:
            q.mul_(0.5)
            k.add_(0.1)
            gate.mul_(0.75)
            beta.mul_(0.8)
            table[0] = 0
            slots[0] = 1
        snapshots.fill_(123.0)
        graph.replay()
        actual_out, actual_snapshots = out.clone(), snapshots.clone()
        snapshots.fill_(123.0)
        expected = recurrent_attention_v100(
            q,
            k,
            v,
            gate,
            beta,
            state,
            slots,
            offsets,
            A_log=rate,
            dt_bias=bias,
            snapshots=snapshots,
            snapshot_slots=table[:1],
            write_state=False,
            raw_beta=True,
        )
        torch.testing.assert_close(actual_out.squeeze(0), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_snapshots, snapshots, rtol=0, atol=0)
        torch.testing.assert_close(state, initial, rtol=0, atol=0)
        unused_slot = 1 - int(table[0])
        assert torch.all(snapshots[unused_slot] == 123.0)


@pytest.mark.parametrize("heads", [8, 16])
def test_kda_prefill_prefix_checkpoint_preserves_live_recurrence(heads):
    """A prefix checkpoint must observe FP32 carry without rounding live state.

    Reusing verification snapshots here would round at every prefill token.
    Changed boundaries and inputs on graph replay must overwrite the checkpoint,
    while untracked rows and unrelated pool slots remain untouched.
    """
    from sglang_v100_plus.attention import V100KDAKernel

    from sglang.kernels.ops.attention.v100 import recurrent_attention_v100

    torch.manual_seed(531)
    lengths = [257, 17]
    total = sum(lengths)
    q, k, v = [
        torch.randn(total, heads, 128, device="cuda", dtype=torch.float16) * 0.1
        for _ in range(3)
    ]
    gate = torch.full_like(q, -0.1, dtype=torch.float32)
    beta = torch.full((total, heads), 0.3, device="cuda", dtype=torch.float16)
    initial = (
        torch.randn(4, heads, 128, 128, device="cuda", dtype=torch.float16) * 0.025
    )
    slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
    state = initial.clone()
    track = torch.full((2, heads, 128, 128), 123.0, device="cuda", dtype=torch.float32)
    kernel = V100KDAKernel()

    def run():
        return kernel.extend(
            q,
            k,
            v,
            gate,
            beta,
            ssm_states=state,
            cache_indices=slots,
            query_start_loc=offsets,
            return_intermediate_states=True,
            track_state=track,
            track_chunk_idx=chunks,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, history = run()
    assert history is None
    for replay in range(2):
        if replay:
            q.mul_(0.75)
            k.mul_(0.625)
            v.add_(0.015)
            chunks[0] = 3
        state.copy_(initial)
        track.fill_(123.0)
        graph.replay()
        actual_out, actual_state, actual_track = (
            out.clone(),
            state.clone(),
            track.clone(),
        )
        expected_state = initial.clone()
        expected_out = kernel.extend(
            q,
            k,
            v,
            gate,
            beta,
            ssm_states=expected_state,
            cache_indices=slots,
            query_start_loc=offsets,
        )
        torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        steps = int(chunks[0]) * 64
        prefix_state = initial.clone()
        recurrent_attention_v100(
            q[:steps],
            k[:steps],
            v[:steps],
            gate[:steps],
            beta[:steps],
            prefix_state,
            slots[:1],
            offsets.new_tensor([0, steps]),
        )
        torch.testing.assert_close(
            actual_track[0].half(), prefix_state[2], rtol=0, atol=0
        )
        # Independent FP64 recurrence at the selected boundary.
        carry = initial[2].double()
        for t in range(steps):
            qt, kt = q[t].double(), k[t].double()
            qt *= torch.rsqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
            kt *= torch.rsqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            carry *= gate[t].double().exp().unsqueeze(-2)
            residual = (v[t].double() - (carry * kt.unsqueeze(-2)).sum(-1)) * beta[
                t
            ].double().unsqueeze(-1)
            carry += residual.unsqueeze(-1) * kt.unsqueeze(-2)
        torch.testing.assert_close(
            actual_track[0].double(), carry, rtol=0.001, atol=0.0002
        )
        assert torch.all(actual_track[1] == 123.0)
        torch.testing.assert_close(actual_state[1], initial[1], rtol=0, atol=0)
        torch.testing.assert_close(actual_state[3], initial[3], rtol=0, atol=0)


class TestWindowAttention(CustomTestCase):
    def test_real_triton_entry_preserves_encoder_sliding_window(self):
        """ENCODER_ONLY removes causality, but retains a query-dependent left mask."""
        from types import MethodType

        from sglang_v100_plus.dflash import forward_attention
        from transformers import Qwen3Config

        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.dflash import _get_dflash_layer_attention_params

        torch.manual_seed(531)
        q = torch.randn(8, 4, 128, device="cuda", dtype=torch.float16) * 0.5
        k = torch.randn(8, 1, 128, device="cuda", dtype=torch.float16) * 0.5
        v = torch.randn_like(k)
        kc, vc = [
            torch.randn(2311, 1, 128, device="cuda", dtype=torch.float16)
            for _ in range(2)
        ]
        ids = torch.randperm(2311, device="cuda", dtype=torch.int32)[:2048]
        ptr = torch.tensor([0, 2048], device="cuda", dtype=torch.int32)
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.use_dense_fp8_chunked_prefill = False
        backend.use_mla = backend.enable_deterministic = False
        backend.use_verify_shared_kv = backend.use_verify_splitkv = False
        backend.use_verify_gfx115 = False
        backend.dcp_size = 1
        backend.page_size = 256
        backend._v100_dflash = True
        backend._v100_dflash_workspaces = {}
        backend.dllm_attention = None
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: kc,
            get_value_buffer=lambda _: vc,
            layer_transfer_counter=None,
        )
        from sglang.srt.mem_cache.kv.router import KVLayerRouter

        backend.token_to_kv_pool.layer_router = KVLayerRouter.from_layer_maps(
            backend.token_to_kv_pool, [("full", backend.token_to_kv_pool, {0: 0})]
        )
        backend.forward_metadata = SimpleNamespace(
            window_kv_indptr=ptr,
            window_kv_indices=ids,
            window_kv_offsets=torch.zeros(1, device="cuda", dtype=torch.int32),
            qo_indptr=ptr.new_tensor([0, 8]),
            custom_mask=None,
            mask_indptr=None,
            max_extend_len=8,
        )
        backend._forward_extend_kernel = MethodType(
            lambda self, layer, *args, **kwargs: forward_attention(
                None, self, layer, *args, **kwargs
            ),
            backend,
        )
        config = Qwen3Config(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            sliding_window=2048,
            use_sliding_window=True,
            is_causal=False,
        )
        left_window, attn_type = _get_dflash_layer_attention_params(config, 0)
        self.assertEqual(left_window, 2047)
        layer = SimpleNamespace(
            qk_head_dim=128,
            v_head_dim=128,
            tp_q_head_num=4,
            layer_id=0,
            logit_capping_method="tanh",
            logit_cap=0,
            is_cross_attention=False,
            attn_type=attn_type,
            sliding_window_size=left_window,
            k_scale=None,
            v_scale=None,
            xai_temperature_len=-1,
            scaling=128**-0.5,
        )
        batch = SimpleNamespace(
            _attn_output=None,
            mha_one_shot=False,
            forward_mode=ForwardMode.TARGET_VERIFY,
            extend_seq_lens_cpu=torch.tensor([8]),
        )
        actual = backend.forward_extend(
            q.reshape(8, 512), k, v, layer, batch, save_kv_cache=False
        )
        keys = torch.cat((kc[ids.long()], k)).cpu().double().view(-1, 128)
        values = torch.cat((vc[ids.long()], v)).cpu().double().view(-1, 128)
        scores = q.cpu().double().reshape(32, 128) @ keys.T * 128**-0.5
        lower = 2048 + torch.arange(8).repeat_interleave(4) - left_window
        scores.masked_fill_(
            torch.arange(len(keys))[None, :] < lower[:, None], -torch.inf
        )
        expected = scores.softmax(-1) @ values
        torch.testing.assert_close(
            actual.cpu().double().view(32, 128), expected, rtol=0.001, atol=0.0002
        )

    def test_changed_window_graph_matches_fp64_and_triton(self):
        """Native FA must preserve per-query windows, page IDs and FP16 KV.

        Poisoned tail/split scratch catches padding reads; changed graph counts
        and permuted/duplicate IDs catch frozen metadata or accidental causality.
        Q/K/V and cache views deliberately have different physical row strides.
        """
        from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
        from sglang.kernels.ops.attention.v100 import window_attention_v100

        torch.manual_seed(531)
        for id_dtype in (torch.int32, torch.int64):
            bank = torch.randn(8, 6, 128, device="cuda", dtype=torch.float16)
            q, k, v = bank[:, :4], bank[:, 4:5], bank[:, 5:]
            cache = torch.randn(2311, 1, 256, device="cuda", dtype=torch.float16)
            kc, vc = cache[:, :, :128], cache[:, :, 128:]
            ids = torch.arange(2065, device="cuda", dtype=id_dtype)
            ptr = torch.tensor([0, 2048], device="cuda", dtype=torch.int32)
            qo = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
            guard = torch.full((32 * 128 + 32,), 37, device="cuda", dtype=torch.float16)
            out = guard[16:-16].view(8, 4, 128)
            workspace = {}
            scale = 128**-0.5

            def run():
                return window_attention_v100(
                    q,
                    k,
                    v,
                    kc,
                    vc,
                    ids,
                    ptr,
                    scale,
                    left_window=2048,
                    workspace=workspace,
                    out=out,
                )

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for i, prefix in enumerate(
                (0, 1, 7, 16, 31, 32, 63, 64, 65, 2039, 2040, 2048)
            ):
                with self.subTest(id_dtype=id_dtype, prefix=prefix):
                    # Calibrated against the original FP16-probability path.
                    # Large random Q/K also exceed this FP64 bound in Triton;
                    # amplify V under uniform scores to test range separately.
                    bank.normal_(std=0.5)
                    if i == 0:
                        v.mul_(3)
                        q.zero_()
                    cache.normal_()
                    ids.copy_((torch.arange(2065, device="cuda") * 97 + i * 19) % 2311)
                    ids[::11] = 17
                    start = 0 if i % 2 else 17
                    ptr.copy_(ptr.new_tensor([start, start + prefix]))
                    # No poison may leak from an inactive key tile or split.
                    for scratch in workspace["window"]:
                        if scratch.is_floating_point():
                            scratch.fill_(float("nan"))
                    graph.replay()
                    saved = out.clone()
                    selected = ids[start : start + prefix].long()
                    keys = torch.cat((kc[selected], k)).cpu().double().view(-1, 128)
                    values = torch.cat((vc[selected], v)).cpu().double().view(-1, 128)
                    scores = (q.cpu().double().reshape(32, 128) @ keys.T) * scale
                    query_positions = prefix + torch.arange(8).repeat_interleave(4)
                    visible = (
                        torch.arange(len(keys))[None, :]
                        >= query_positions[:, None] - 2048
                    )
                    expected = (
                        scores.masked_fill(~visible, -torch.inf).softmax(-1) @ values
                    )
                    torch.testing.assert_close(
                        saved.cpu().double().reshape(32, 128),
                        expected,
                        rtol=0.001,
                        atol=0.0002,
                    )
                    reference = torch.empty_like(out)
                    extend_attention_fwd(
                        q,
                        k,
                        v,
                        reference,
                        kc,
                        vc,
                        qo,
                        ptr,
                        ids,
                        None,
                        False,
                        None,
                        8,
                        1.0,
                        1.0,
                        scale,
                        sliding_window_size=2048,
                    )
                    torch.testing.assert_close(
                        saved, reference, rtol=0.001, atol=0.0002
                    )
                    torch.testing.assert_close(saved, run(), rtol=0, atol=0)
                    self.assertTrue(bool(torch.all(guard[:16] == 37)))
                    self.assertTrue(bool(torch.all(guard[-16:] == 37)))


class TestSparseVerificationRows(CustomTestCase):
    def test_wider_graph_keeps_split_workspace_and_query_support(self):
        """Five/eight-row verification must retain bounded split scheduling.

        A shape cutoff previously sent these rows through paired prefill and
        ignored the caller split hint/workspace. Changing graph support must
        reuse the same bounded buffers and preserve each query's selected IDs.
        """
        from math import log

        from sglang.kernels.ops.attention.v100 import sparse_attention_v100

        torch.manual_seed(531)
        for rows in (5, 6, 8):
            with self.subTest(rows=rows):
                bank = torch.randn(rows, 33, 512, device="cuda", dtype=torch.float16)
                q = bank[:, :32]
                kv = torch.randn(257, 512, device="cuda", dtype=torch.float16) * 0.25
                ids = torch.zeros(rows, 129, device="cuda", dtype=torch.int32)
                lengths = torch.full((rows,), 129, device="cuda", dtype=torch.int32)
                workspace = {}

                def run():
                    return sparse_attention_v100(
                        q,
                        kv,
                        ids,
                        512**-0.5,
                        lengths,
                        workspace=workspace,
                        kv_splits=7,
                        return_lse=True,
                    )

                run()
                self.assertIn("sparse", workspace)
                storage = workspace["sparse"]
                self.assertLessEqual(storage[0].numel(), rows * 7 * 32)
                self.assertLessEqual(storage[1].numel(), rows * 7 * 32 * 512)
                self.assertEqual(storage[0].dtype, torch.float32)
                self.assertEqual(storage[1].dtype, torch.float16)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual, actual_lse = run()
                for replay in range(2):
                    q.normal_(std=0.25)
                    kv.normal_(std=0.25)
                    ids.random_(-2, 260)
                    ids[:, ::7] = 17
                    lengths.copy_(
                        torch.tensor(
                            [
                                (0, 1, 65, 129, 132, -1, 32, 64)[(i + replay) % 8]
                                for i in range(rows)
                            ],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    graph.replay()
                    self.assertIs(workspace["sparse"][0], storage[0])
                    self.assertIs(workspace["sparse"][1], storage[1])
                    for row in range(rows):
                        selected = ids[
                            row, : max(0, min(129, int(lengths[row])))
                        ].long()
                        selected = selected[(selected >= 0) & (selected < len(kv))]
                        if not len(selected):
                            self.assertEqual(torch.count_nonzero(actual[row]).item(), 0)
                            self.assertTrue(bool(torch.isneginf(actual_lse[row]).all()))
                            continue
                        values = kv[selected].double()
                        scores = q[row].double() @ values.T * 512**-0.5
                        expected = scores.softmax(-1) @ values
                        torch.testing.assert_close(
                            actual[row].double(), expected, rtol=0.001, atol=0.0002
                        )
                        torch.testing.assert_close(
                            actual_lse[row].double(),
                            scores.logsumexp(-1) / log(2),
                            rtol=1e-5,
                            atol=1e-5,
                        )


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("pages,seq,topk", [(257, 33, 67), (262273, 129, 2051)])
def test_sm70_tensorcore_preserves_support_ragged_tail_and_graph_replay(
    heads, pages, seq, topk
):
    """Gather/softmax MMA tiles must preserve independent sparse query support."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core sparse prefill")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(740)
    # Head-strided input, odd query count and non-power-of-two pool4 width.
    packed = torch.randn(seq, heads, 576, device="cuda", dtype=torch.float16)
    q = packed[:, :, :512]
    kv = torch.randn(pages, 1, 512, device="cuda", dtype=torch.float16)
    indices = torch.full((seq, 1, topk), -1, device="cuda", dtype=torch.int32)
    scale = 256**-0.5
    for _ in range(3):
        sparse_attention_v100(q, kv, indices, scale).unsqueeze(0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sparse_attention_v100(q, kv, indices, scale).unsqueeze(0)
    for update in range(3):
        packed.normal_()
        kv.normal_()
        indices.fill_(-1)
        for row in range(seq):
            if (row + update) % 5 == 0:
                continue
            valid = torch.randperm(pages, device="cuda")[: min(topk, row * 3 + 1)]
            columns = torch.randperm(topk, device="cuda")[: valid.numel()]
            indices[row, 0, columns] = valid.int()
        # A populated first query with an entirely empty second query guards
        # the all-masked initial tile: a finite sentinel would leak KV values.
        indices[0].fill_(-1)
        indices[0, 0, :3] = torch.tensor([5, 17, 31], device="cuda")
        indices[1].fill_(-1)
        if topk == 2051:
            indices.fill_(-1)
            for row, count in enumerate((1, 65, 128, 2048, 2051)):
                indices[row + 2, 0, :count] = torch.arange(
                    pages - count, pages, device="cuda", dtype=torch.int32
                )
            indices[4, 0, 16:32] = -1
        before = indices.clone()
        graph.replay()
        eager = sparse_attention_v100(q, kv, indices, scale).unsqueeze(0)
        expected = torch.zeros_like(q, dtype=torch.float64)
        for row in range(seq):
            ids = indices[row, 0]
            selected = kv[ids[ids >= 0].long(), 0].double()
            if selected.numel():
                expected[row] = (q[row].double() @ selected.T * scale).softmax(
                    -1
                ) @ selected
        for result in (actual, eager):
            assert torch.isfinite(result).all()
            torch.testing.assert_close(
                result[0].double(), expected, rtol=0.01, atol=0.001
            )
            assert torch.count_nonzero(result[0, 1]) == 0
        assert torch.equal(indices, before)
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("topk", [67, 2051])
def test_sm70_tensorcore_graph_refreshes_explicit_length_bounds(heads, topk):
    """Shared-tile reuse must retain live bounds, masks and duplicate support."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core sparse prefill")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(742)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.float16)
    kv = torch.randn(257, 1, 512, device="cuda", dtype=torch.float16)
    indices = (
        torch.arange(topk, dtype=torch.int32, device="cuda").remainder(257).repeat(3, 1)
    )
    indices[:, ::11] = -1
    indices[:, ::17] = 257  # Out-of-pool padding.
    indices[:, 33:41] = 256  # Duplicate keys keep their original multiplicity.
    indices[:, 64:128] = -1  # Empty tile before later populated tiles.
    indices[:, 0] = 7
    lengths = torch.tensor([0, 1, topk], dtype=torch.int32, device="cuda")
    scale = 512**-0.5

    def run():
        return sparse_attention_v100(q, kv, indices, scale, lengths).unsqueeze(0)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for bounds in ((0, 1, topk), (topk + 33, -1, 65), (1, topk, 0)):
        lengths.copy_(torch.tensor(bounds, dtype=torch.int32, device="cuda"))
        q.normal_()
        kv.normal_()
        indices.copy_(indices.roll(1, dims=1))
        graph.replay()
        eager = run()
        expected = torch.zeros_like(q, dtype=torch.float64)
        for row, count in enumerate(bounds):
            ids = indices[row, : min(topk, max(0, count))]
            selected = kv[ids[(ids >= 0) & (ids < kv.shape[0])].long(), 0].double()
            if selected.numel():
                expected[row] = (q[row].double() @ selected.T * scale).softmax(
                    -1
                ) @ selected
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual[0].double(), expected, rtol=0.01, atol=0.001)
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))


@pytest.mark.parametrize("topk", [67, 2051, 4096])
def test_sparse_length_scan_handles_ragged_width_and_empty_rows(topk):
    """The first backscan block may cross column zero, never its row boundary."""
    from sglang.kernels.ops.attention.metadata import q8kv8_topk_length_from_indices

    indices = torch.full((4, topk), -1, dtype=torch.int32, device="cuda")
    indices[1, 0] = 7
    indices[2, topk - 1] = 11
    indices[3, min(65, topk - 1)] = 19
    expected = torch.tensor(
        [1, 1, topk, min(65, topk - 1) + 1], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(q8kv8_topk_length_from_indices(indices), expected)


@pytest.mark.parametrize(
    "rows,heads,topk,splits,use_lengths,owned_workspace",
    [
        (1, 8, 2051, None, False, True),
        (1, 16, 2051, None, False, True),
        (4, 16, 2051, None, False, False),
        (3, 16, 67, 3, True, True),
        (2, 8, 67, 1, True, False),
        (4, 16, 2051, 3, True, True),
    ],
)
def test_sm70_tensorcore_decode_reference_graph_and_workspace(
    rows, heads, topk, splits, use_lengths, owned_workspace
):
    """Split partials must retain sparse support, stride and live graph inputs."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core sparse decode")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(743)
    pages = 262273 if topk == 2051 else 257
    packed = torch.randn(rows, heads, 576, device="cuda", dtype=torch.float16)
    if rows == 4 and not use_lengths:
        # BMM-derived queries can be head-major after a transpose, not merely
        # a head-strided view of a token-major tensor.
        packed = packed.transpose(0, 1).contiguous().transpose(0, 1)
    q = packed[:, :, :512]
    kv = torch.randn(pages, 1, 512, device="cuda", dtype=torch.float16)
    ids = torch.full((rows, 1, topk), -1, device="cuda", dtype=torch.int32)
    lengths = (
        torch.zeros(rows, device="cuda", dtype=torch.int32) if use_lengths else None
    )
    workspace = {} if owned_workspace else None
    scale = 256**-0.5

    def run():
        return sparse_attention_v100(
            q, kv, ids, scale, lengths, workspace, splits
        ).unsqueeze(0)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    old_buffers = [
        (lse.data_ptr(), acc.data_ptr()) for lse, acc in (workspace or {}).values()
    ]
    for update, count in enumerate((0, 1, 65, 2048, 2051, 0)):
        packed.normal_()
        kv.normal_()
        ids.fill_(-1)
        for row in range(rows):
            valid = min(topk, max(0, count - row * 16))
            if valid:
                selected = torch.randint(
                    pages, (valid,), device="cuda", dtype=torch.int32
                )
                columns = torch.randperm(topk, device="cuda")[:valid]
                ids[row, 0, columns] = selected
                ids[row, 0, -1] = pages - 1
                # Invalid upper slot plus duplicate selection and an interior hole.
                ids[row, 0, :3] = torch.tensor(
                    [pages, pages - 1, pages - 1], device="cuda"
                )
                ids[row, 0, 16:32] = -1
        if use_lengths:
            lengths.copy_(
                torch.tensor(
                    [
                        -1 if count == 0 else min(topk + 7, count - row * 16)
                        for row in range(rows)
                    ],
                    device="cuda",
                    dtype=torch.int32,
                )
            )
        before = ids.clone()
        graph.replay()
        eager = run()
        expected = torch.zeros_like(q, dtype=torch.float64)
        for row in range(rows):
            limit = topk if lengths is None else min(topk, max(0, int(lengths[row])))
            selected_ids = ids[row, 0, :limit]
            selected_ids = selected_ids[(selected_ids >= 0) & (selected_ids < pages)]
            selected = kv[selected_ids.long(), 0].double()
            if selected.numel():
                expected[row] = (q[row].double() @ selected.T * scale).softmax(
                    -1
                ) @ selected
        for output in (actual, eager):
            assert output.dtype == torch.float16 and output.shape == (
                1,
                rows,
                heads,
                512,
            )
            assert torch.isfinite(output).all()
            torch.testing.assert_close(
                output[0].double(), expected, rtol=0.01, atol=0.001
            )
            if count == 0:
                assert torch.count_nonzero(output) == 0
        assert torch.equal(ids, before)
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))
        assert old_buffers == [
            (lse.data_ptr(), acc.data_ptr()) for lse, acc in (workspace or {}).values()
        ]


@pytest.mark.parametrize("misaligned", ["q", "kv"])
def test_sm70_tensorcore_decode_rejects_unaligned_storage_views(misaligned):
    """A contiguous view can still violate the vector-load pointer alignment."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta Tensor Core sparse decode")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.float16)
    kv = torch.zeros(65, 512, device="cuda", dtype=torch.float16)
    ids = torch.zeros(1, 1, 67, device="cuda", dtype=torch.int32)
    if misaligned == "q":
        q = torch.empty(q.numel() + 1, device="cuda", dtype=q.dtype)[1:].view_as(q)
        q.zero_()
    else:
        kv = torch.empty(kv.numel() + 1, device="cuda", dtype=kv.dtype)[1:].view_as(kv)
    if misaligned == "kv":
        with pytest.raises(RuntimeError, match="alignment"):
            sparse_attention_v100(q, kv, ids, 0.0625)
    else:
        assert torch.count_nonzero(sparse_attention_v100(q, kv, ids, 0.0625)) == 0


@pytest.mark.parametrize("heads", [8, 16])
def test_sm70_sparse_prefill_realigns_contiguous_offset_views(heads):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 sparse FA")
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(743)
    q = torch.randn(3 * heads * 512 + 1, device="cuda", dtype=torch.float16)[1:]
    q = q.view(3, heads, 512)
    kv = torch.randn(67 * 512 + 1, device="cuda", dtype=torch.float16)[1:]
    kv = kv.view(67, 512)
    assert q.is_contiguous() and kv.is_contiguous()
    assert q.data_ptr() % 16 and kv.data_ptr() % 16
    ids = torch.tensor(
        [[1, 1, -1], [-1, -1, -1], [5, 31, 66]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([3, 3, 3], device="cuda", dtype=torch.int32)
    before = ids.clone()
    # The shared API realigns Q; the cache must already meet its storage contract.
    with pytest.raises(RuntimeError, match="alignment"):
        sparse_attention_v100(q, kv, ids, 0.0625, lengths)
    actual = sparse_attention_v100(q, kv.clone(), ids, 0.0625, lengths)
    expected = torch.zeros_like(q, dtype=torch.float64)
    for row in (0, 2):
        selected = kv[ids[row][ids[row] >= 0].long()].double()
        expected[row] = (q[row].double() @ selected.T * 0.0625).softmax(-1) @ selected
    torch.testing.assert_close(actual.double(), expected, rtol=0.01, atol=0.001)
    assert torch.equal(ids, before)


@pytest.mark.parametrize("invalid", ["dtype", "lengths", "heads", "alignment"])
def test_sm70_sparse_fa_binding_rejects_unsafe_operands(invalid):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 sparse FA binding")
    from sglang_fa_sm70 import ensure_loaded

    ensure_loaded()
    q = torch.zeros(2, 8, 512, device="cuda", dtype=torch.float16)
    kv = torch.zeros(67, 512, device="cuda", dtype=torch.float16)
    ids = torch.zeros(2, 3, device="cuda", dtype=torch.int32)
    lengths = torch.full((2,), 3, device="cuda", dtype=torch.int32)
    if invalid == "dtype":
        q = q.float()
    elif invalid == "lengths":
        lengths = lengths[:1]
    elif invalid == "heads":
        q = q[:, :4].contiguous()
    else:
        q = torch.empty(q.numel() + 1, device="cuda", dtype=q.dtype)[1:].view_as(q)
    with pytest.raises(RuntimeError, match="SM70 sparse MLA"):
        torch.ops.sglang_fa_sm70.prefill(q, kv, ids, lengths, 0.0625)


@pytest.mark.parametrize("rows", [1, 4, 5, 129])
@pytest.mark.parametrize("dcp", [2, 4, 8])
def test_sm70_dcp_normalization_and_graph_replay(rows, dcp):
    """Latent stripes must reconstruct sparse attention, including empty owners.

    Pooled-index selection stays global and replicated; normalizers must be
    base-2, with -inf for empty rows, before the upstream FP32 DCP correction.
    """
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta FA partial normalization")
    from sglang.kernels.ops.attention.dcp_kernels import correct_attn_out
    from sglang.kernels.ops.attention.dsa.transform_index import (
        transform_index_page_table_dcp,
    )
    from sglang.kernels.ops.attention.v100 import sparse_attention_v100

    torch.manual_seed(745)
    heads, count, width = 8 * dcp, 4099, 2051
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.float16)
    kv = torch.randn(count, 512, device="cuda", dtype=torch.float16)
    stripes = [kv[rank::dcp].contiguous() for rank in range(dcp)]
    ids = torch.zeros(rows, width, device="cuda", dtype=torch.int32)
    workspaces = [{} for _ in range(dcp)]
    scale = 256**-0.5

    def run():
        outputs, normalizers = [], []
        for rank in range(dcp):
            local, lengths = transform_index_page_table_dcp(ids, dcp, rank)
            out, lse = sparse_attention_v100(
                q,
                stripes[rank],
                local,
                scale,
                lengths=lengths,
                workspace=workspaces[rank],
                return_lse=True,
            )
            outputs.append(out)
            normalizers.append(lse)
        lses = torch.stack(normalizers)
        corrected = [
            correct_attn_out(
                outputs[rank],
                lses,
                rank,
                None,
                new_output=torch.empty(
                    heads, rows, 512, device="cuda", dtype=torch.float32
                ),
                is_lse_base_on_e=False,
            )[0]
            for rank in range(dcp)
        ]
        return torch.stack(corrected).sum(0).transpose(0, 1).half(), lses

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, lses = run()
    for replay in range(3):
        q.normal_()
        kv.normal_()
        ids.random_(-1, count + 2)
        if rows > 1 or replay == 1:
            ids[0].fill_(-1)  # Every owner is empty.
        if rows > 1:
            ids[1] = dcp * (torch.arange(width, device="cuda") % (count // dcp))
            ids[-1, -3:] = torch.tensor([count - 1, -1, 7], device="cuda")
        for rank in range(dcp):
            stripes[rank].copy_(kv[rank::dcp])
        graph.replay()
        cq, ck, ci = q.double().cpu(), kv.double().cpu(), ids.cpu()
        expected = torch.zeros(rows, heads, 512, dtype=torch.float64)
        expected_lse = torch.full((dcp, rows, heads), -torch.inf, dtype=torch.float64)
        for row in range(rows):
            selected = ci[row][(ci[row] >= 0) & (ci[row] < count)].long()
            if selected.numel():
                scores = cq[row] @ ck[selected].T * scale
                expected[row] = scores.softmax(-1) @ ck[selected]
            for rank in range(dcp):
                local = selected[selected % dcp == rank]
                if local.numel():
                    expected_lse[rank, row] = torch.logsumexp(
                        cq[row] @ ck[local].T * scale, -1
                    ) / math.log(2)
        torch.testing.assert_close(
            actual.double().cpu(), expected, rtol=0.001, atol=0.0002
        )
        torch.testing.assert_close(
            lses.double().cpu(), expected_lse, rtol=0.0001, atol=0.0001
        )
        if rows > 1 or replay == 1:
            assert torch.count_nonzero(actual[0]) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
