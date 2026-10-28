import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeStream:
    def __init__(self, stream_id):
        self.cuda_stream = stream_id


class FakeEvent:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


def _make_scheduler(**attrs):
    scheduler = object.__new__(SchedulerPPMixin)
    scheduler.__dict__.update(attrs)
    return scheduler


class TestPPCommOverlap(CustomTestCase):
    @staticmethod
    def _rebuild_sampled_verify(scheduler, batch):
        # Only the CUDA tree construction is replaced. The packet, request
        # adoption/reordering and actual EagleVerifyInput construction run.
        tokens = batch.spec_info.tokens.flatten()
        indices = torch.arange(tokens.numel()).reshape_as(batch.spec_info.tokens)
        tree_outputs = (
            torch.empty(0, dtype=torch.bool),
            torch.arange(tokens.numel()),
            indices,
            torch.full_like(indices, -1),
            torch.full_like(indices, -1),
            tokens,
        )
        with (
            get_context().override_server_args(
                speculative_eagle_topk=1,
                speculative_num_steps=2,
                speculative_num_draft_tokens=3,
                speculative_use_rejection_sampling=True,
            ),
            patch(
                "sglang.srt.speculative.eagle_utils.build_tree_kernel_efficient",
                return_value=tree_outputs,
            ),
        ):
            scheduler._pp_spec_rebuild_verify_input(batch)
        return batch.spec_info

    @staticmethod
    def _relay_batch(rids):
        return SimpleNamespace(
            reqs=[SimpleNamespace(rid=rid) for rid in rids],
            spec_info=None,
            sampling_info=SimpleNamespace(is_all_greedy=False),
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: len(rids),
            seq_lens=torch.full((len(rids),), 10, dtype=torch.int64),
            seq_lens_cpu=torch.full((len(rids),), 10, dtype=torch.int64),
            seq_lens_sum=10 * len(rids),
        )

    def test_proposal_distribution_survives_pp_packet_and_request_reordering(self):
        """PP dropped the proposal q, so sampled verification had no correction.

        q must travel with the exact draft tokens and follow request identity,
        rather than the current microbatch's row order.
        """
        scheduler = _make_scheduler(
            _pp_spec_relay=True,
            device="cpu",
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(verify_mask=None),
                    model_config=SimpleNamespace(context_len=100, vocab_size=4),
                )
            ),
        )
        probs = torch.tensor([[[0.1, 0.2, 0.3, 0.4]] * 2, [[0.4, 0.3, 0.2, 0.1]] * 2])
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([1, 2]),
            accept_lens=torch.tensor([1, 1]),
            new_seq_lens=torch.tensor([11, 11]),
            next_draft_input=EagleDraftInput(bonus_tokens=torch.tensor([1, 2])),
            next_verify_chain=torch.tensor([1, 2, 3, 2, 3, 0]),
            next_verify_parent_list=torch.tensor([[-1, 0], [-1, 0]]),
            next_verify_top_scores_index=torch.tensor([[0, 1], [0, 1]]),
            next_verify_draft_probs=probs,
        )
        batch = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, return_logprob=False
        )
        payload = scheduler._pp_prepare_tensor_dict(result, batch)
        live = self._relay_batch(["b", "a"])
        with get_context().override_server_args(speculative_num_draft_tokens=3):
            scheduler._pp_spec_adopt_relayed_tree(
                live, ["a", "b"], PPProxyTensors(payload)
            )
        verify = self._rebuild_sampled_verify(scheduler, live)
        torch.testing.assert_close(verify.draft_probs, probs[[1, 0]], rtol=0, atol=0)
        self.assertEqual(verify.draft_token.tolist(), [2, 3, 0, 1, 2, 3])

    def test_proposal_rows_follow_merge_filter_and_subset_adoption(self):
        probs = torch.tensor([[[0.2, 0.8]] * 2, [[0.7, 0.3]] * 2])
        relay = PPSpecRelayInput(
            ["a", "b"], torch.tensor([[1, 2, 3], [4, 5, 6]]), draft_probs=probs
        )
        fresh = PPSpecRelayInput.degenerate(["c"], torch.tensor([7]), 3)
        relay.merge_batch(fresh)
        relay.filter_batch(torch.tensor([2, 0, 1]))
        torch.testing.assert_close(relay.draft_probs[0], torch.zeros_like(probs[0]))
        torch.testing.assert_close(relay.draft_probs[1:], probs)
        update = PPSpecRelayInput(
            ["b"], torch.tensor([[8, 9, 10]]), draft_probs=probs[:1]
        )
        relay.adopt(update)
        self.assertEqual(relay.rids, ["c", "a", "b"])
        torch.testing.assert_close(relay.draft_probs[1], probs[0])
        torch.testing.assert_close(relay.draft_probs[2], probs[0])
        relay.adopt(PPSpecRelayInput.degenerate(["b"], torch.tensor([11]), 3))
        torch.testing.assert_close(relay.draft_probs[2], torch.zeros_like(probs[0]))
        self.assertEqual(relay.tokens[2].tolist(), [11, 0, 0])

    def test_cold_pp_tree_resamples_target_without_inventing_proposal_probs(self):
        scheduler = _make_scheduler(
            device="cpu",
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(verify_mask=None),
                    model_config=SimpleNamespace(context_len=100, vocab_size=4),
                )
            ),
        )
        batch = self._relay_batch(["a"])
        batch.spec_info = PPSpecRelayInput.degenerate(["a"], torch.tensor([1]), 3)
        verify = self._rebuild_sampled_verify(scheduler, batch)
        torch.testing.assert_close(verify.draft_probs, torch.zeros(1, 2, 4))
        # Greedy verification never reads q; keep its existing allocation-free
        # path rather than constructing an unused full-vocabulary buffer.
        batch.spec_info = PPSpecRelayInput.degenerate(["a"], torch.tensor([1]), 3)
        batch.sampling_info.is_all_greedy = True
        verify = self._rebuild_sampled_verify(scheduler, batch)
        self.assertIsNone(verify.draft_probs)

    def test_chain_relay_keeps_accept_indices_for_recurrent_state_commit(self):
        """Earlier PP stages need accepted steps even when KV needs no compaction.

        Omitting chain indices left their recurrent state at the prompt and
        produced incorrect cold-cache Qwen tool arguments with PP + MTP.
        """
        scheduler = _make_scheduler(_pp_spec_relay=True)
        accept_indices = torch.tensor([[0, 1, -1, -1], [4, 5, 6, -1]])
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([[10, 11, -1, -1], [12, 13, 14, -1]]),
            accept_lens=torch.tensor([2, 3]),
            accept_index=accept_indices,
            new_seq_lens=torch.tensor([302, 403]),
            next_draft_input=EagleDraftInput(bonus_tokens=torch.tensor([11, 14])),
        )
        batch = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, return_logprob=False
        )
        with get_context().override_server_args(speculative_eagle_topk=1):
            payload = scheduler._pp_prepare_tensor_dict(result, batch)
        self.assertIn("spec_accept_index", payload)
        torch.testing.assert_close(payload["spec_accept_index"], accept_indices)

    def test_graph_proxy_send_records_forward_reuse_fence(self):
        comm_stream = FakeStream(4)
        work = Mock()
        works = [SimpleNamespace(work=work)]
        scheduler = _make_scheduler(
            pp_comm_stream=comm_stream,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
            device_module=SimpleNamespace(Event=FakeEvent),
        )

        scheduler._pp_commit_comm_work(works, fence_next_forward=True)

        work.wait.assert_called_once_with()
        self.assertEqual(works, [])
        self.assertIs(scheduler.pp_send_done_event.recorded_stream, comm_stream)

    def test_no_fence_event_without_comm_stream(self):
        scheduler = _make_scheduler(
            pp_comm_stream=None,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
        )

        scheduler._pp_commit_comm_work([SimpleNamespace(work=Mock())], True)

        self.assertIsNone(scheduler.pp_send_done_event)

    def test_forward_waits_for_graph_send_and_proxy_receive(self):
        schedule_stream = FakeStream(1)
        send_done_event = object()
        recv_event = object()
        forward_stream = Mock()
        scheduler = _make_scheduler(
            schedule_stream=schedule_stream,
            forward_stream=forward_stream,
            pp_send_done_event=send_done_event,
            pp_proxy_recv_event=recv_event,
        )

        scheduler._pp_wait_forward_dependencies()

        forward_stream.wait_stream.assert_called_once_with(schedule_stream)
        self.assertEqual(
            forward_stream.wait_event.call_args_list,
            [call(send_done_event), call(recv_event)],
        )
        self.assertIsNone(scheduler.pp_send_done_event)
        self.assertIsNone(scheduler.pp_proxy_recv_event)

    def test_inbox_returns_original_receive_event(self):
        recv_event = object()
        tensor_dict = {"__msg_type__": "output", "value": torch.arange(2)}
        scheduler = _make_scheduler(
            _pp_tensor_dict_inbox=defaultdict(
                deque, {"output": deque([(tensor_dict, recv_event)])}
            ),
        )

        received, event = scheduler._pp_recv_typed_dict("output")

        self.assertIs(received, tensor_dict)
        self.assertIs(event, recv_event)


if __name__ == "__main__":
    unittest.main()
