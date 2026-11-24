"""The local PP2 result must follow the output ring's FIFO and event order."""

import sys
import unittest
from collections import deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.pipeline import (
        initialize_local_output,
        pack_output,
        receive_output,
        send_output,
        send_output_dict,
        set_local_relay,
        unpack_output,
    )


class TestV100Pipeline(CustomTestCase):
    def owner(self, *, last=True, spec=SpeculativeAlgorithm.NONE):
        return SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=last), spec_algorithm=spec
        )

    def test_decode_snapshot_can_commit_a_speculative_verify_graph(self):
        """Restored DECODE snapshots share copy graphs across weight/TP layouts."""
        from sglang_v100_plus.commit_graph import commit_relayed_states

        from sglang.srt.environ import envs
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        def tensor(shape):
            return SimpleNamespace(
                shape=torch.Size(shape),
                dtype=torch.int64,
                device=torch.device("cuda:0"),
                is_cuda=True,
            )

        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(mamba_pool=SimpleNamespace()),
                attn_backend=object(),
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["Qwen4ExpForConditionalGeneration"]
                    )
                ),
            )
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=tensor([1]),
            mamba_track_indices=None,
        )
        settings = dict(
            model_path="dummy",
            tp_size=4,
            ep_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_use_rejection_sampling=True,
        )
        states, commits = {}, []

        # CUDA allocation/capture is the dependency boundary. Exercise the real
        # selector/cache below; GPU tests exercise actual capture and state bits.
        def capture(original, owner, snapshot, accept, indices, width):
            state = SimpleNamespace(draft_tokens=width)
            state.run = lambda *args: commits.append((state, args))
            return state

        def eager(*_args):
            return "eager"

        with (
            envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.override(True),
            envs.SGLANG_ENABLE_PP_SPEC.override(True),
            patch("sglang_v100_plus.commit_graph.get_buffer", return_value=states),
            patch("sglang_v100_plus.commit_graph.CommitGraph", side_effect=capture),
        ):
            for quant, tp, ep in (("fp8", 4, 4), ("modelopt_fp4", 4, 1), (None, 2, 2)):
                for steps in (1, 2, 3):
                    width = steps + 1
                    accept, indices = tensor([1]), tensor([1, width])
                    with (
                        self.subTest(quant=quant, tp=tp, ep=ep, steps=steps),
                        get_context().override_server_args(
                            **(
                                settings
                                | dict(
                                    quantization=quant,
                                    tp_size=tp,
                                    ep_size=ep,
                                    speculative_num_steps=steps,
                                    speculative_num_draft_tokens=width,
                                )
                            ),
                        ),
                        get_parallel().override(pp_rank=0),
                    ):
                        self.assertIsNone(
                            commit_relayed_states(
                                eager, worker, batch, accept, indices, width
                            )
                        )
                        self.assertEqual(commits[-1][0].draft_tokens, width)
                        self.assertEqual(commits[-1][1], (batch, accept, indices))
            self.assertEqual(len(states), 3)
            self.assertIs(commits[0][0], commits[3][0])
            self.assertIs(commits[1][0], commits[4][0])

            with (
                get_context().override_server_args(
                    **settings,
                    quantization="fp8",
                    speculative_num_steps=2,
                    speculative_num_draft_tokens=3,
                ),
                get_parallel().override(pp_rank=0),
            ):
                accept, indices = tensor([1]), tensor([1, 3])
                previous = commits[1][0]
                # Replacing just the state pool under the same request pool
                # must not replay a graph captured against old state storage.
                worker.model_runner.req_to_token_pool.mamba_pool = SimpleNamespace()
                commit_relayed_states(eager, worker, batch, accept, indices, 3)
                self.assertIsNot(commits[-1][0], previous)
                self.assertEqual(len(states), 4)
                for fields, value in (
                    ({"pp_size": 3}, None),
                    ({"pp_async_batch_depth": 1}, None),
                    ({"max_running_requests": 2}, None),
                    ({"disable_overlap_schedule": False}, None),
                    ({"disaggregation_mode": "prefill"}, None),
                    ({"speculative_eagle_topk": 2}, None),
                    ({"speculative_use_rejection_sampling": False}, None),
                    ({}, (tensor([1]), tensor([1, 4]))),
                    ({}, (tensor([2]), tensor([2, 3]))),
                ):
                    with self.subTest(fields=fields, tensors=value):
                        before = len(commits)
                        with (
                            get_context().override_server_args(
                                **(
                                    settings
                                    | dict(
                                        quantization="fp8",
                                        speculative_num_steps=2,
                                        speculative_num_draft_tokens=3,
                                    )
                                    | fields
                                )
                            ),
                            get_parallel().override(pp_rank=0),
                        ):
                            result = commit_relayed_states(
                                eager, worker, batch, *(value or (accept, indices)), 3
                            )
                        self.assertEqual(result, "eager")
                        self.assertEqual(len(commits), before)

                # GLM Triton KDA shares the stable-row scatter contract. A
                # replacement linear backend must retain/capture new owners.
                worker.model_runner.model_config.hf_config.architectures = [
                    "Glm5NextForConditionalGeneration"
                ]
                backend = SimpleNamespace(
                    linear_attn_backend=SimpleNamespace(accept_lens_pool=None)
                )
                worker.model_runner.attn_backend = backend
                commit_relayed_states(eager, worker, batch, accept, indices, 3)
                first = commits[-1][0]
                backend.linear_attn_backend = SimpleNamespace(accept_lens_pool=None)
                commit_relayed_states(eager, worker, batch, accept, indices, 3)
                self.assertIsNot(commits[-1][0], first)
                for arch, accept_pool in (
                    ("Glm5NextForConditionalGeneration", object()),
                    ("OtherModel", None),
                ):
                    worker.model_runner.model_config.hf_config.architectures = [arch]
                    backend.linear_attn_backend.accept_lens_pool = accept_pool
                    before = len(commits)
                    self.assertEqual(
                        commit_relayed_states(eager, worker, batch, accept, indices, 3),
                        "eager",
                    )
                    self.assertEqual(len(commits), before)

    def test_nested_logprobs_do_not_put_storage_in_pickle_metadata(self):
        """Nested logprobs formerly bypassed extraction and restored on the sender GPU."""
        import pickle

        from sglang.srt.distributed.parallel_state import _split_tensor_dict

        tensors = {
            "ids": torch.tensor([7, 13]),
            "logprobs": [
                {
                    "values": torch.tensor([-0.0, -0.125]),
                    "indices": (torch.tensor([7, 13]), None),
                }
            ],
            "empty": torch.empty(1, 0),
            "scalar": torch.tensor(True),
        }
        packed = pack_output(tensors)
        metadata, payloads = _split_tensor_dict(packed)

        def contains_tensor(value):
            if isinstance(value, torch.Tensor):
                return True
            if isinstance(value, dict):
                return any(contains_tensor(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return any(contains_tensor(item) for item in value)
            return False

        self.assertFalse(contains_tensor(metadata))
        self.assertEqual(len(payloads), 1)
        restored_metadata = dict(pickle.loads(pickle.dumps(metadata)))
        restored_metadata["__v100_pp_output_payload__"] = payloads[0].clone()
        restored = unpack_output(restored_metadata)
        self.assertIsInstance(restored["logprobs"], list)
        self.assertIsInstance(restored["logprobs"][0]["indices"], tuple)
        self.assertIsNone(restored["logprobs"][0]["indices"][1])
        for actual, expected in (
            (restored["ids"], tensors["ids"]),
            (restored["logprobs"][0]["values"], tensors["logprobs"][0]["values"]),
            (
                restored["logprobs"][0]["indices"][0],
                tensors["logprobs"][0]["indices"][0],
            ),
            (restored["empty"], tensors["empty"]),
            (restored["scalar"], tensors["scalar"]),
        ):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(
                torch.equal(
                    actual.reshape(-1).view(torch.uint8),
                    expected.reshape(-1).view(torch.uint8),
                )
            )

    def test_glm_mtp_retains_exact_proposal_and_index_seed(self):
        """GLM's FP4/TP4 transport must preserve q and its DSA seed locally."""
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        settings = dict(
            model_path="dummy",
            quantization="modelopt_fp4",
            tp_size=4,
            ep_size=1,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            speculative_use_rejection_sampling=True,
        )
        owner = self.owner(spec=SpeculativeAlgorithm.EAGLE)
        with get_context().override_server_args(**settings):
            initialize_local_output(lambda _: None, owner)
        self.assertIsNotNone(owner._v100_pp_local_outputs)
        event = object()
        tensors = {
            "next_token_ids": torch.tensor([17, 19, 23]),
            "spec_accept_lens": torch.tensor([3]),
            "spec_next_draft_probs": torch.tensor(
                [[[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]]
            ),
            "draft_dsa_topk_indices": torch.tensor(
                [[31, 32, 63, 64, 129]], dtype=torch.int32
            ),
            "draft_hidden_states": torch.arange(16, dtype=torch.float32)[None],
        }
        proxy = PPProxyTensors(tensors)
        pending = deque([(event, proxy)])
        sent = []

        def wire(_, payload, **kwargs):
            sent.append(payload)
            return [object()]

        owner._pp_send_dict_to_next_stage = partial(send_output_dict, wire, owner)

        def original(_, mb, batches, queue, previous):
            forward_event, value = queue.popleft()
            return owner._pp_send_dict_to_next_stage(
                value.tensors, msg_type="output", ready_event=forward_event
            )

        send_output(original, owner, 0, [True], pending, None)
        local, ready = receive_output(None, owner)
        self.assertIs(local, tensors)
        self.assertIs(ready, event)
        decoded = unpack_output(sent[0])
        self.assertNotIn("spec_next_draft_probs", decoded)
        self.assertIs(local["spec_next_draft_probs"], tensors["spec_next_draft_probs"])
        for key, expected in tensors.items():
            if key != "spec_next_draft_probs":
                torch.testing.assert_close(decoded[key], expected, rtol=0, atol=0)

        # Unvalidated scheduling/topology variants retain their existing wire protocol.
        for changes in (
            {"pp_size": 3},
            {"pp_async_batch_depth": 1},
            {"max_running_requests": 2},
            {"speculative_eagle_topk": 2},
        ):
            with (
                self.subTest(changes=changes),
                get_context().override_server_args(**(settings | changes)),
            ):
                other = self.owner(spec=SpeculativeAlgorithm.EAGLE)
                initialize_local_output(lambda _: None, other)
                self.assertIsNone(other._v100_pp_local_outputs)

    def test_result_fifo_preserves_all_fields_and_original_events(self):
        """Removing the echo must not shift one result into the next token turn."""
        owner = self.owner()
        with get_context().override_server_args(
            model_path="dummy",
            tp_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
        ):
            initialize_local_output(lambda _: None, owner)
        first_event, second_event = object(), object()
        first = SimpleNamespace(
            tensors={
                "next_token_ids": torch.tensor([7]),
                "logprobs": torch.tensor([-0.1]),
            }
        )
        second = SimpleNamespace(tensors={"next_token_ids": torch.tensor([11])})
        queue = deque([(first_event, first), (second_event, second)])
        sent = []

        def send(_, mb, mbs, results, pp_outputs):
            if mbs[mb] is not None:
                sent.append(results.popleft())
                return [object()]
            return []

        # Empty slots do not consume or retain anything. Two pending results
        # must remain FIFO even if the receiver processes them later.
        send_output(send, owner, 0, [None, True], queue, None)
        send_output(send, owner, 1, [None, True], queue, None)
        send_output(send, owner, 1, [None, True], queue, None)
        for expected_event, expected_proxy in sent:
            tensors, event = receive_output(None, owner)
            self.assertIs(event, expected_event)
            self.assertIs(tensors, expected_proxy.tensors)
        self.assertFalse(owner._v100_pp_local_outputs)
        owner.pp_group.is_last_rank = False
        # Stage zero still receives from the original ring, but must not echo.
        self.assertEqual(send_output(None, owner, 0, [True], deque(), first), [])
        self.assertEqual(receive_output(lambda _: ({}, None), owner), ({}, None))

    def test_skipped_sends_and_unsupported_schedules_keep_protocol(self):
        """A skipped prefill output must not become a later decode result."""
        owner = self.owner()
        with get_context().override_server_args(
            model_path="dummy",
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
        ):
            initialize_local_output(lambda _: None, owner)
        queue = deque([(object(), SimpleNamespace(tensors={}))])

        def skip(_, mb, mbs, results, pp_outputs):
            results.popleft()
            return []

        self.assertEqual(send_output(skip, owner, 0, [True], queue, None), [])
        self.assertFalse(owner._v100_pp_local_outputs)
        for fields, spec in (
            ({"pp_size": 3}, SpeculativeAlgorithm.NONE),
            ({"pp_async_batch_depth": 1}, SpeculativeAlgorithm.NONE),
            ({"max_running_requests": 2}, SpeculativeAlgorithm.NONE),
            ({"disable_overlap_schedule": False}, SpeculativeAlgorithm.NONE),
            ({"disaggregation_mode": "prefill"}, SpeculativeAlgorithm.NONE),
            ({}, SpeculativeAlgorithm.EAGLE),
        ):
            with self.subTest(fields=fields, spec=spec):
                defaults = {
                    "model_path": "dummy",
                    "pp_size": 2,
                    "max_running_requests": 1,
                    "disable_overlap_schedule": True,
                }
                defaults.update(fields)
                with get_context().override_server_args(**defaults):
                    owner = self.owner(spec=spec)
                    initialize_local_output(lambda _: None, owner)
                self.assertIsNone(owner._v100_pp_local_outputs)
                self.assertEqual(
                    send_output(lambda *args: "send", owner, 0, [], deque(), None),
                    "send",
                )
                self.assertEqual(receive_output(lambda _: "receive", owner), "receive")

    def test_spec_receive_first_consumes_previously_sent_result(self):
        """PP2 receives the previous turn's result before sending this turn's.

        Retaining the current send queue at receive time selects the wrong
        result (or an empty queue). Preserve the original forward event and
        every speculative field across both turns without returning an echo.
        """
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardMode,
            PPProxyTensors,
        )

        owner = self.owner(spec=SpeculativeAlgorithm.EAGLE)
        owner._pp_spec_relay = True
        sent, received, waits = [], [], []
        owner.copy_stream_ctx = nullcontext()
        owner.schedule_stream = object()
        owner.copy_stream = SimpleNamespace(
            wait_stream=lambda stream: waits.append(stream),
            wait_event=lambda event: waits.append(event),
        )
        owner.device_module = SimpleNamespace(
            Event=lambda: SimpleNamespace(record=lambda _: None),
            current_stream=lambda: owner.copy_stream,
        )

        def wire_send(_, tensors, **kwargs):
            sent.append((kwargs["ready_event"], tensors))
            return [object()]

        def wire_receive(_):
            self.fail("The final stage must consume its local result")

        def prep(target, metadata, proxy):
            received.append(proxy.tensors)
            return SimpleNamespace(logits_output=None)

        owner._pp_send_dict_to_next_stage = partial(send_output_dict, wire_send, owner)
        owner._pp_send_output_to_next_stage = partial(
            send_output, SchedulerPPMixin._pp_send_output_to_next_stage, owner
        )
        owner._pp_recv_dict_from_prev_stage = partial(
            receive_output, wire_receive, owner
        )
        owner._pp_prep_batch_result = prep
        owner._pp_result_done_event = partial(
            SchedulerPPMixin._pp_result_done_event, owner
        )
        target = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        batches = [target, None]
        metadata = [SimpleNamespace(forward_done=None), None]
        queue = deque()

        settings = dict(
            model_path="dummy",
            quantization="fp8",
            tp_size=4,
            ep_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
            speculative_algorithm="EAGLE",
            speculative_num_steps=2,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=3,
            speculative_use_rejection_sampling=True,
        )
        for layout in (
            {},
            {"quantization": "modelopt_fp4", "ep_size": 1},
            {"tp_size": 2, "ep_size": 2},
            {"tp_size": 8, "ep_size": 8},
            {"quantization": None, "tp_size": 2, "ep_size": 1},
        ):
            with (
                get_context().override_server_args(**(settings | layout)),
                get_parallel().override(pp_rank=1),
            ):
                initialize_local_output(lambda _: None, owner)
                for token in (7, 11, 13):
                    event = object()
                    tensors = {
                        "next_token_ids": torch.tensor([token, token + 1]),
                        "spec_accept_lens": torch.tensor([2]),
                        "spec_new_seq_lens": torch.tensor([token + 100]),
                        "spec_bonus_tokens": torch.tensor([token + 1]),
                        "spec_accept_index": torch.tensor([[0, 1, -1]]),
                        "spec_next_chain": torch.tensor([[token + 1, token + 2, 0]]),
                        "spec_next_parents": torch.tensor([[-1, 0]]),
                        "spec_next_top_scores": torch.tensor([[0, 1]]),
                        "spec_next_draft_probs": torch.tensor(
                            [[[0.125, 0.875], [0.25, 0.75]]], dtype=torch.float32
                        ),
                    }
                    queue.append((event, PPProxyTensors(tensors)))
                    first = (
                        SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                            owner, 0, 1, batches, metadata, queue, None
                        )
                    )
                    self.assertEqual(len(first[-1]), 1)
                    self.assertFalse(queue)
                    # The next microbatch is empty, but must receive the previous
                    # result before its empty send in the upstream parity order.
                    second = (
                        SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors(
                            owner, 1, 0, batches, metadata, queue, None
                        )
                    )
                    self.assertEqual(second[-1], [])
                    self.assertIs(received[-1], tensors)
                    self.assertIs(waits[-1], event)
                    self.assertFalse(owner._v100_pp_local_outputs)
                    wire = unpack_output(sent[-1][1])
                    self.assertNotIn("spec_next_draft_probs", wire)
                    self.assertEqual(
                        set(wire), set(tensors) - {"spec_next_draft_probs"}
                    )
                    for key, value in wire.items():
                        self.assertTrue(torch.equal(value, tensors[key]))
        self.assertEqual(len(sent), 15)
        self.assertEqual(len(received), 15)
        # Keep the wire protocol for modes outside the chain/schedule
        # contract; enabling locality unconditionally could silently consume
        # graph buffers from a different request or unsupported relay.
        for unsupported in (
            {"speculative_eagle_topk": 2},
            {"speculative_num_steps": 4, "speculative_num_draft_tokens": 5},
            {"speculative_use_rejection_sampling": False},
            {"max_running_requests": 2},
        ):
            with self.subTest(unsupported=unsupported):
                with get_context().override_server_args(**(settings | unsupported)):
                    initialize_local_output(lambda _: None, owner)
                self.assertEqual(receive_output(lambda _: "wire", owner), "wire")
                send_output_dict(wire_send, owner, tensors, msg_type="output")
                self.assertIs(sent[-1][1], tensors)

    def test_complete_relay_replacement_preserves_q_and_partial_rows(self):
        """Replacing all live rows needs no gather/copy; subsets must keep peers."""
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput

        owner = self.owner(spec=SpeculativeAlgorithm.EAGLE)
        original = SchedulerPPMixin._pp_spec_set_relay

        def relay(rids, offset):
            rows = len(rids)
            return PPSpecRelayInput(
                rids,
                tokens=torch.arange(rows * 3).reshape(rows, 3) + offset,
                parents=torch.arange(rows * 2).reshape(rows, 2),
                top_scores=torch.arange(rows * 2).reshape(rows, 2) + 1,
                draft_probs=torch.arange(rows * 8, dtype=torch.float32).reshape(
                    rows, 2, 4
                )
                + offset,
            )

        with get_context().override_server_args(
            model_path="dummy",
            quantization="fp8",
            tp_size=4,
            ep_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
            speculative_algorithm="EAGLE",
            speculative_num_steps=2,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=3,
            speculative_use_rejection_sampling=True,
        ):
            initialize_local_output(lambda _: None, owner)
            for rids in (["first"], ["first", "second"]):
                with self.subTest(rids=rids):
                    relayed = relay(["first"], 10)
                    batch = SimpleNamespace(
                        reqs=[SimpleNamespace(rid=rid) for rid in rids],
                        spec_info=relay(rids, 100),
                    )
                    reference = SimpleNamespace(
                        reqs=batch.reqs, spec_info=relay(rids, 100)
                    )
                    original(owner, reference, relayed)
                    set_local_relay(original, owner, batch, relayed)
                    for field in ("tokens", "parents", "top_scores", "draft_probs"):
                        self.assertTrue(
                            torch.equal(
                                getattr(batch.spec_info, field),
                                getattr(reference.spec_info, field),
                            )
                        )
                    if len(rids) == 1:
                        self.assertIs(batch.spec_info, relayed)
                    else:
                        self.assertIsNot(batch.spec_info, relayed)
            # A cold row must retain the ordinary fallback's zero-q and
            # previous topology behavior when the incoming fields are absent.
            relayed = PPSpecRelayInput.degenerate(
                ["first"], torch.tensor([9]), num_draft_tokens=3
            )
            batch = SimpleNamespace(
                reqs=[SimpleNamespace(rid="first")], spec_info=relay(["first"], 100)
            )
            set_local_relay(original, owner, batch, relayed)
            self.assertFalse(torch.count_nonzero(batch.spec_info.draft_probs))
            self.assertIsNotNone(batch.spec_info.parents)


if __name__ == "__main__":
    unittest.main()
