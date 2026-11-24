"""Small-batch projections, HC gates and draft caches on SM70."""

import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[3] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.dispatch import V100FallbackError
    from sglang_v100_plus.kernels import gemm, sm70_hc_mix
    from sglang_v100_plus.runtime import apply_unquant, store


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0),
    "Requires a V100",
)
class TestV100SmallBatches(CustomTestCase):
    def test_spec_sample_graph_preserves_native_tokens_rng_and_retained_results(self):
        from contextlib import nullcontext
        from inspect import unwrap

        from sglang_v100_plus.sample_graph import SampleGraph

        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.runtime_context import get_context, get_parallel
        from sglang.srt.speculative.eagle_utils import eagle_sample

        original = unwrap(eagle_sample)
        group = SimpleNamespace(world_size=1)
        for width in (2, 3, 4):
            info = SimpleNamespace(
                temperatures=torch.ones(1, 1, device="cuda"),
                top_ps=torch.ones(1, device="cuda"),
                top_ks=torch.full((1,), -1, device="cuda", dtype=torch.int32),
                min_ps=torch.zeros(1, device="cuda"),
                is_all_greedy=False,
                need_top_k_sampling=False,
                need_top_p_sampling=False,
                sampling_seed=None,
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                logit_bias=None,
            )
            batch = SimpleNamespace(
                device="cuda",
                forward_mode=ForwardMode.TARGET_VERIFY,
                seq_lens=torch.tensor([32], device="cuda"),
                sampling_info=info,
            )
            plan = SimpleNamespace(
                tree_topk=1,
                draft_token_num=width,
                max_tree_depth=width,
                draft_token=torch.arange(width, device="cuda"),
                retrieve_index=torch.arange(width, device="cuda")[None, :],
                retrieve_next_token=torch.tensor(
                    [list(range(1, width)) + [-1]], device="cuda"
                ),
                retrieve_next_sibling=torch.full((1, width), -1, device="cuda"),
                draft_probs=torch.softmax(
                    torch.randn(1, width - 1, 4096, device="cuda"), dim=-1
                ),
            )
            logits = SimpleNamespace(
                next_token_logits=torch.randn(width, 4096, device="cuda")
            )
            with (
                self.subTest(width=width),
                torch.inference_mode(),
                get_context().override_server_args(
                    model_path="dummy",
                    speculative_algorithm="EAGLE",
                    speculative_use_rejection_sampling=True,
                ),
                get_parallel().override(tp_group=group),
                # This single-GPU test needs no distributed capture context;
                # native multi-rank runs separately exercise NCCL broadcasts.
                patch(
                    "sglang.srt.distributed.graph_capture", return_value=nullcontext()
                ),
            ):
                before = torch.cuda.get_rng_state()
                graph = SampleGraph(original, plan, batch, logits, group)
                self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
                retained = saved = None
                for iteration in range(6):
                    logits.next_token_logits.normal_()
                    plan.draft_probs.copy_(
                        torch.softmax(torch.randn_like(plan.draft_probs), dim=-1)
                    )
                    plan.draft_token.add_(17).remainder_(4096)
                    info.temperatures.fill_(0.7 if iteration % 2 else 1.0)
                    batch.seq_lens.add_(1)
                    proposal = plan.draft_probs.clone()
                    before = torch.cuda.get_rng_state()
                    reference = tuple(v.clone() for v in original(plan, batch, logits))
                    expected_rng = torch.cuda.get_rng_state()
                    torch.cuda.set_rng_state(before)
                    actual = graph.run(plan, batch, logits)
                    for value, expected in zip(actual, reference):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    self.assertTrue(
                        torch.equal(expected_rng, torch.cuda.get_rng_state())
                    )
                    self.assertTrue(torch.equal(proposal, plan.draft_probs))
                    if retained is not None:
                        for value, expected in zip(retained, saved):
                            torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    retained, saved = actual, tuple(v.clone() for v in actual)

    def test_pp_commit_graph_refreshes_request_acceptance_and_source_state(self):
        """Capture must refresh inputs and keep chain/state-pool owners distinct."""
        from sglang_v100_plus.commit_graph import commit_relayed_states

        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.runtime_context import get_context, get_parallel

        def pool(width):
            return SimpleNamespace(
                destination=torch.zeros(4, 8, device="cuda"),
                source=torch.empty(width, 8, device="cuda"),
            )

        def original(
            worker, batch, accept_lens, accept_index, draft_tokens, prepared=None
        ):
            state = worker.model_runner.req_to_token_pool.mamba_pool
            steps = accept_index.gather(1, (accept_lens - 1).reshape(-1, 1)).flatten()
            state.destination.index_copy_(
                0,
                batch.req_pool_indices,
                state.source.reshape(draft_tokens, 8).index_select(0, steps),
            )

        for quant, ep in (("fp8", 4), ("modelopt_fp4", 1)):
            for width in (2, 3, 4):
                request_pool = SimpleNamespace(mamba_pool=pool(width))
                worker = SimpleNamespace(
                    model_runner=SimpleNamespace(
                        req_to_token_pool=request_pool,
                        attn_backend=object(),
                        model_config=SimpleNamespace(
                            hf_config=SimpleNamespace(
                                architectures=["Qwen4ExpForConditionalGeneration"]
                            )
                        ),
                    )
                )
                graphs, expected = {}, torch.zeros(4, 8)
                old_source = old_destination = old_output = None
                with (
                    self.subTest(quant=quant, width=width),
                    torch.inference_mode(),
                    get_context().override_server_args(
                        model_path="dummy",
                        quantization=quant,
                        tp_size=4,
                        ep_size=ep,
                        pp_size=2,
                        max_running_requests=1,
                        disable_overlap_schedule=True,
                        speculative_algorithm="EAGLE",
                        speculative_num_steps=width - 1,
                        speculative_eagle_topk=1,
                        speculative_num_draft_tokens=width,
                        speculative_use_rejection_sampling=True,
                    ),
                    get_parallel().override(pp_rank=0),
                    envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.override(True),
                    envs.SGLANG_ENABLE_PP_SPEC.override(True),
                    patch(
                        "sglang_v100_plus.commit_graph.get_buffer", return_value=graphs
                    ),
                ):
                    for iteration in range(16):
                        if iteration == 8:
                            old_state = request_pool.mamba_pool
                            old_output = old_state.destination.clone()
                            old_source = weakref.ref(old_state.source)
                            old_destination = weakref.ref(old_state.destination)
                            request_pool.mamba_pool = pool(width)
                            del old_state
                            expected.zero_()
                        state = request_pool.mamba_pool
                        slot, count = iteration % 4, iteration % width + 1
                        batch = SimpleNamespace(
                            req_pool_indices=torch.tensor([slot], device="cuda"),
                            forward_mode=ForwardMode.DECODE,
                            mamba_track_indices=None,
                        )
                        accept = torch.tensor([count], device="cuda")
                        indices = torch.tensor(
                            [list(reversed(range(width)))], device="cuda"
                        )
                        state.source.copy_(
                            torch.arange(width * 8, device="cuda").reshape(width, 8)
                            + iteration * 100
                        )
                        commit_relayed_states(
                            original, worker, batch, accept, indices, width
                        )
                        expected[slot] = (
                            torch.arange(8) + (width - count) * 8 + iteration * 100
                        )
                        self.assertTrue(
                            torch.equal(
                                state.destination.cpu().view(torch.uint8),
                                expected.view(torch.uint8),
                            )
                        )
                        if old_source is not None:
                            self.assertIsNotNone(old_source())
                            self.assertIsNotNone(old_destination())
                            self.assertTrue(torch.equal(old_destination(), old_output))
                    self.assertEqual(len(graphs), 2)
                    self.assertTrue(
                        all(value.graph is not None for value in graphs.values())
                    )

    def test_pp_output_pack_preserves_bits_and_cross_stream_readiness(self):
        from collections import deque

        from sglang_v100_plus.pipeline import (
            receive_output,
            send_output_dict,
            unpack_output,
        )

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        owner = SimpleNamespace(
            _v100_pp_local_outputs=deque(),
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        tensors = {
            "next_token_ids": torch.arange(3, device="cuda", dtype=torch.int32),
            "seq_lens": torch.tensor([8193], device="cuda", dtype=torch.int64),
            "probabilities": torch.tensor([-0.0, 0.125, 0.875], device="cuda"),
            "hidden_states": torch.arange(12, device="cuda", dtype=torch.float16)
            .reshape(3, 4)
            .T,
            "empty_parents": torch.empty(1, 0, device="cuda", dtype=torch.int64),
            "mask": torch.tensor([True, False, True], device="cuda"),
            "scalar": torch.tensor(7, device="cuda", dtype=torch.int64),
            "spec_next_draft_probs": torch.ones(1, 2, 3, device="cuda"),
            "__msg_type__": "output",
            "nested_logprobs": {
                "values": [torch.tensor([-0.0, -0.125], device="cuda")],
                "indices": (torch.tensor([7, 13], device="cuda"), None),
                "cpu": torch.empty(2, dtype=torch.int64, pin_memory=True),
            },
        }
        producer, sender = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            torch.cuda._sleep(200000)
            tensors["hidden_states"].add_(16)
            tensors["nested_logprobs"]["values"][0].add_(0.5)
            tensors["nested_logprobs"]["cpu"].copy_(
                tensors["nested_logprobs"]["indices"][0], non_blocking=True
            )
            ready = torch.cuda.Event()
            ready.record()
        sent = []

        def wire(_, payload, **kwargs):
            sent.append((payload, kwargs["ready_event"]))
            return []

        with torch.cuda.stream(sender):
            send_output_dict(wire, owner, tensors, msg_type="output", ready_event=ready)
        payload, packed_ready = sent[0]
        self.assertIsNot(packed_ready, ready)
        torch.cuda.current_stream().wait_event(packed_ready)
        actual = unpack_output(payload)
        self.assertNotIn("spec_next_draft_probs", actual)
        self.assertIn("spec_next_draft_probs", tensors)
        self.assertEqual(payload["__v100_pp_output_payload__"].numel() % 8, 0)

        def compare(actual_value, expected):
            if isinstance(expected, torch.Tensor):
                self.assertEqual(actual_value.shape, expected.shape)
                self.assertEqual(actual_value.dtype, expected.dtype)
                self.assertEqual(actual_value.device.type, expected.device.type)
                self.assertTrue(
                    torch.equal(
                        actual_value.contiguous().reshape(-1).view(torch.uint8),
                        expected.contiguous().reshape(-1).view(torch.uint8),
                    )
                )
            elif isinstance(expected, dict):
                for key in expected:
                    compare(actual_value[key], expected[key])
            elif isinstance(expected, (list, tuple)):
                self.assertIsInstance(actual_value, type(expected))
                for item, reference in zip(actual_value, expected):
                    compare(item, reference)
            else:
                self.assertEqual(actual_value, expected)

        for key, expected in tensors.items():
            if key != "spec_next_draft_probs":
                compare(actual[key], expected)

        # Prefill has no proposal q, but nested GPU logprobs still must travel
        # as tensor payloads. Restoring CPU leaves must wait for receive DMA.
        prefill = {"nested_logprobs": tensors["nested_logprobs"]}
        with torch.cuda.stream(sender):
            send_output_dict(wire, owner, prefill, msg_type="output", ready_event=ready)
        packed, sent_ready = sent[-1]
        receiver = SimpleNamespace(
            _v100_pp_local_outputs=deque(), pp_group=SimpleNamespace(is_last_rank=False)
        )
        copied = packed.copy()
        with torch.cuda.stream(sender):
            sender.wait_event(sent_ready)
            torch.cuda._sleep(200000)
            copied["__v100_pp_output_payload__"] = packed[
                "__v100_pp_output_payload__"
            ].clone()
            received_ready = torch.cuda.Event()
            received_ready.record()
        restored, event = receive_output(lambda _: (copied, received_ready), receiver)
        self.assertIs(event, received_ready)
        compare(restored, prefill)
        if torch.cuda.device_count() > 1:
            copied["__v100_pp_output_payload__"] = copied[
                "__v100_pp_output_payload__"
            ].to("cuda:1")
            restored = unpack_output(copied)
            self.assertEqual(restored["nested_logprobs"]["values"][0].device.index, 1)
            torch.testing.assert_close(
                restored["nested_logprobs"]["values"][0].cpu(),
                prefill["nested_logprobs"]["values"][0].cpu(),
                rtol=0,
                atol=0,
            )

    def test_wide_topk_preserves_ties_mass_and_graph_inputs(self):
        """The faster renorm must retain cutoff ties and read each graph replay."""
        from sgl_kernel.sampling import _top_k_renorm_probs_internal
        from sglang_v100_plus.runtime import top_k_renorm_probs

        from sglang.srt.sampling.sampling_params import TOP_K_ALL

        torch.manual_seed(531)
        for vocab in (131072, 154880, 248320, 262144):
            for rows in (1, 2, 3, 4):
                probs = torch.softmax(
                    torch.randn(rows, vocab, device="cuda") * 4, dim=-1
                )
                top_ks = torch.full((rows,), 20, device="cuda", dtype=torch.int32)
                top_k_renorm_probs(probs, top_ks)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = top_k_renorm_probs(probs, top_ks)
                for tied in (False, True, False):
                    with self.subTest(vocab=vocab, rows=rows, tied=tied):
                        if tied:
                            probs.fill_(1 / probs.shape[-1])
                        else:
                            probs.copy_(
                                torch.softmax(torch.randn_like(probs) * 4, dim=-1)
                            )
                        top_ks.copy_(
                            torch.tensor(
                                [1, 20, 50, TOP_K_ALL][:rows],
                                device="cuda",
                                dtype=torch.int32,
                            ).roll(1 if tied else 0)
                        )
                        graph.replay()
                        # An independent FP64 mass calculation and the old
                        # AOT implementation both constrain the new output.
                        indices = top_ks.long().clamp(max=vocab)[:, None] - 1
                        pivot = (
                            probs.double()
                            .sort(dim=-1, descending=True)
                            .values.gather(1, indices)
                        )
                        kept = torch.where(probs >= pivot, probs.double(), 0.0)
                        reference = (kept / kept.sum(-1, keepdim=True)).float()
                        legacy = _top_k_renorm_probs_internal(probs, top_ks, 0)
                        self.assertTrue(torch.equal(actual > 0, reference > 0))
                        self.assertTrue(torch.equal(actual > 0, legacy > 0))
                        torch.testing.assert_close(
                            actual, reference, rtol=3e-6, atol=1e-8
                        )
                        torch.testing.assert_close(actual, legacy, rtol=3e-6, atol=1e-8)

    def test_dflash_output_convolution_keeps_range_until_residual_norm(self):
        from sglang_v100_plus.dflash import SM70DFlashGroupedConv

        conv = SM70DFlashGroupedConv(16, 8, 2, 16).cuda().half()
        conv.base_kernel.data.fill_(2.0)
        hidden = torch.full((16, 16), 2048.0, device="cuda", dtype=torch.float16)
        coefficients = torch.full((16, 2, 1), 63.0, device="cuda", dtype=torch.float16)
        conv.finish(hidden, coefficients)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = conv.finish(hidden, coefficients)
        for value in (2048.0, 1024.0):
            hidden.fill_(value)
            graph.replay()
            expected = torch.full(
                (16, 16), 2 * 65 * value, device="cuda", dtype=torch.float32
            )
            expected[::8] = 65 * value
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(output).all())

    def test_dflash_topk_graph_reads_fresh_logits(self):
        from sglang_v100_plus.dflash import candidate_topk

        def forbidden_backend(*args):
            self.fail("Volta candidate selection must not delegate to FlashInfer")

        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype), torch.inference_mode():
                scores = torch.full((8, 19360), -4096.0, device="cuda", dtype=dtype)
                rows = torch.arange(8, device="cuda")[:, None]
                positions = torch.arange(16, device="cuda")[None, :] * 1024
                values = torch.arange(16, device="cuda", dtype=dtype)[None, :].expand(
                    8, -1
                )
                scores[rows, positions.expand(8, -1)] = values
                candidate_topk(forbidden_backend, scores, 16)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual_values, actual_ids = candidate_topk(
                        forbidden_backend, scores, 16
                    )
                for shift in (5, 91):
                    scores.fill_(-4096)
                    expected_ids = (positions + rows * 37 + shift).expand(8, -1)
                    scores[rows, expected_ids] = values
                    graph.replay()
                    torch.testing.assert_close(
                        actual_values, values.flip(-1), rtol=0, atol=0
                    )
                    self.assertTrue(torch.equal(actual_ids, expected_ids.flip(-1)))

    def test_fp16_draft_cache_preserves_strided_rows_and_graph_replay(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        with torch.inference_mode():
            pool = MHATokenToKVPool(
                size=64,
                page_size=64,
                dtype=torch.float16,
                head_num=1,
                head_dim=128,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
                enable_alt_stream=False,
            )
            pool.k_buffer[0].zero_()
            pool.v_buffer[0].zero_()
            qkv = torch.randn(8, 768, device="cuda", dtype=torch.float16)
            key = qkv[:, 512:640].view(8, 1, 128)
            value = qkv[:, 640:].view(8, 1, 128)
            self.assertFalse(key.is_contiguous())
            locations = torch.arange(8, device="cuda", dtype=torch.int64)
            layer = SimpleNamespace(layer_id=0)
            k_scale = torch.tensor(2.0, device="cuda")
            v_scale = torch.tensor(3.0, device="cuda")

            def write():
                return store(
                    MHATokenToKVPool.set_kv_buffer,
                    pool,
                    layer,
                    locations,
                    key,
                    value,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

            write()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                write()
            expected_k, expected_v = pool.k_buffer[0].clone(), pool.v_buffer[0].clone()
            for start in (0, 17, 41):
                qkv.normal_()
                key[..., 0] = 0.0
                key[..., 1] = -0.0
                value[..., 0] = 2**-24
                locations.copy_(torch.arange(start, start + 8, device="cuda"))
                graph.replay()
                # Slot 0 is reserved graph padding and the primary CUDA writer
                # skips it, including during capture/warmup.
                live = locations != 0
                expected_k[locations[live]] = key[live]
                expected_v[locations[live]] = value[live]
                self.assertTrue(
                    torch.equal(
                        pool.k_buffer[0].view(torch.int16), expected_k.view(torch.int16)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        pool.v_buffer[0].view(torch.int16), expected_v.view(torch.int16)
                    )
                )

    def test_volta_blas_projections_preserve_operand_dtype(self):
        def forbidden_fallback(*args):
            self.fail("Declared Volta projections must not delegate dispatch")

        with (
            envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True),
            torch.inference_mode(),
        ):
            for rows in (1, 6, 8):
                for n, dtype in (
                    (3336, torch.float16),
                    (192, torch.float16),  # ERNIE fused index projection
                    (3337, torch.float16),  # cuBLAS handles arbitrary FP16 widths
                    (32, torch.float32),
                    (768, torch.float16),
                    (3072, torch.float16),
                ):
                    x = torch.randn(rows, 4096, device="cuda", dtype=dtype)
                    weight = torch.randn(n, 4096, device="cuda", dtype=dtype) * 0.02
                    actual = apply_unquant(
                        forbidden_fallback, None, SimpleNamespace(weight=weight), x
                    )
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(
                        actual, torch.nn.functional.linear(x, weight), rtol=0, atol=0
                    )
            x = torch.zeros(6, 4096, device="cuda", dtype=torch.float32)
            weight = torch.zeros(3337, 4096, device="cuda", dtype=torch.float32)
            with self.assertRaisesRegex(V100FallbackError, "gemm.unquantized_linear"):
                apply_unquant(
                    forbidden_fallback, None, SimpleNamespace(weight=weight), x
                )

    def test_hc_three_rows_equal_four_row_prefix(self):
        torch.manual_seed(53)
        with (
            envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True),
            torch.inference_mode(),
        ):
            x = torch.randn(4, 10240, device="cuda", dtype=torch.float16)
            down_weight = (
                torch.randn(320, 10240, device="cuda", dtype=torch.float16) * 0.02
            )
            up_weight = (
                torch.randn(10240, 320, device="cuda", dtype=torch.float16) * 0.02
            )
            inject = torch.randn(4, 10240, device="cuda", dtype=torch.float16) * 0.02
            down3 = sm70_hc_mix.hc_down(x[:3], down_weight)
            down4 = sm70_hc_mix.hc_down(x, down_weight)
            torch.testing.assert_close(down3, down4[:3], atol=0, rtol=0)
            up3 = sm70_hc_mix.hc_up(down3, x[:3], up_weight)
            up4 = sm70_hc_mix.hc_up(down4, x, up_weight)
            torch.testing.assert_close(up3, up4[:3], atol=0, rtol=0)
            gated3, partial3 = sm70_hc_mix.hc_down_with_gate(x[:3], down_weight, inject)
            gated4, partial4 = sm70_hc_mix.hc_down_with_gate(x, down_weight, inject)
            torch.testing.assert_close(gated3, gated4[:3], atol=0, rtol=0)
            torch.testing.assert_close(partial3, partial4[:3], atol=0, rtol=0)
            y = torch.randn(4, 2560, device="cuda", dtype=torch.float16)
            combine3 = sm70_hc_mix.hc_apply_gate(y[:3], x[:3], partial3)
            combine4 = sm70_hc_mix.hc_apply_gate(y, x, partial4)
            torch.testing.assert_close(combine3, combine4[:3], atol=0, rtol=0)

    def test_gemm_three_rows_equal_four_row_prefix(self):
        torch.manual_seed(54)
        with (
            envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True),
            torch.inference_mode(),
        ):
            for (rows, n, k), _ in gemm._SMALL_CONFIGS.items():
                if rows != 3:
                    continue
                with self.subTest(n=n, k=k):
                    x = torch.randn(4, k, device="cuda", dtype=torch.float16)
                    weight = (
                        torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
                    )
                    out3 = gemm.linear_small(x[:3], weight)
                    # N=1 had a two-row specialization only; rows are independent.
                    out_ref = (
                        gemm.linear_small(x, weight)[:3]
                        if (4, n, k) in gemm._SMALL_CONFIGS
                        else torch.cat(
                            [
                                gemm.linear_small(x[:2], weight),
                                gemm.linear_small(x[2:], weight),
                            ]
                        )[:3]
                    )
                    torch.testing.assert_close(out3, out_ref, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
