"""Linear MTP penalties must match ordinary decoding at each causal prefix."""

import sys
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
    apply_scaling_penalties,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.sampling import (
        apply_linear_penalties,
        copy_penalty_metadata,
        exact_eagle_sample,
    )

PENALIZERS = {
    BatchedFrequencyPenalizer,
    BatchedPresencePenalizer,
    BatchedRepetitionPenalizer,
    BatchedMinNewTokensPenalizer,
}


class Request:
    def __init__(self, history, **changes):
        self.output_ids = list(history)
        self.sampling_params = SimpleNamespace(
            **(
                dict(
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                    min_new_tokens=0,
                    stop_token_ids=None,
                )
                | changes
            )
        )
        self.eos_token_ids = {7}
        self.tokenizer = SimpleNamespace(eos_token_id=7, additional_stop_token_ids=None)


class Batch:
    def __init__(self, reqs):
        self.reqs, self.device = reqs, "cpu"
        self.forward_mode = SimpleNamespace(is_idle=lambda: False)
        orchestrator = BatchedPenalizerOrchestrator(8, self, PENALIZERS)
        self.sampling_info = SimpleNamespace(
            penalizer_orchestrator=orchestrator,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
        )


def ordinary_reference(batch, candidates, positions, logits):
    expected = logits.clone().view(len(batch.reqs), candidates.shape[1], 8)
    # Execute the existing ordinary penalizers on one independent request per
    # query row. Feed every generated token plus that row's actual draft prefix.
    for i, req in enumerate(batch.reqs):
        root = positions[i].min().item()
        for j in range(candidates.shape[1]):
            prefix = [
                candidates[i, k].item()
                for k in positions[i].argsort()
                if root < positions[i, k] <= positions[i, j]
            ]
            reference_req = Request(
                list(req.output_ids) + prefix, **vars(req.sampling_params)
            )
            reference_batch = Batch([reference_req])
            orch = reference_batch.sampling_info.penalizer_orchestrator
            # Preserve the existing ordinary order; set iteration is not a
            # documented ordering contract between different orchestrators.
            order = batch.sampling_info.penalizer_orchestrator.penalizers
            for token in reference_req.output_ids:
                orch.cumulate_output_tokens(torch.tensor([token]))
            with patch(
                "sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties",
                apply_scaling_penalties.__wrapped__,
            ):
                for kind in order:
                    orch.penalizers[kind].apply(expected[i, j : j + 1])
    return expected.reshape_as(logits)


class TestV100Sampling(CustomTestCase):
    def check_prefixes(self, reqs, candidates, positions):
        batch = Batch(reqs)
        width = candidates.shape[1]
        plan = SimpleNamespace(
            tree_topk=1,
            draft_token_num=width,
            max_tree_depth=width,
            draft_token=candidates.flatten(),
            positions=positions.flatten(),
        )
        logits = torch.linspace(-4.0, 4.0, len(reqs) * width * 8).reshape(-1, 8)
        expected = ordinary_reference(batch, candidates, positions, logits)
        with get_context().override_server_args(model_path="dummy"):
            apply_linear_penalties(plan, batch, logits)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        return batch, plan, expected

    def test_presence_includes_all_accepted_tokens_and_each_draft_prefix(self):
        """Repeating one base penalty misses accepted and newly proposed tokens."""
        self.check_prefixes(
            [Request([3, 3, 1], presence_penalty=1.5)],
            torch.tensor([[1, 4, 4]]),
            torch.tensor([[100, 101, 102]]),
        )

    def test_mixed_penalties_follow_reordered_rows_and_minimum_length(self):
        self.check_prefixes(
            [
                Request(
                    [4, 3, 3, 1],
                    frequency_penalty=0.3,
                    presence_penalty=1.5,
                    repetition_penalty=2.0,
                    min_new_tokens=6,
                ),
                Request(
                    [5],
                    frequency_penalty=-0.25,
                    presence_penalty=-1.5,
                    repetition_penalty=0.5,
                ),
            ],
            torch.tensor([[4, 1, 4], [2, 6, 5]]),
            torch.tensor([[101, 100, 102], [302, 301, 300]]),
        )

    def test_history_cache_handles_more_accepts_and_recycled_prefixes(self):
        req = Request([1], frequency_penalty=0.3, presence_penalty=1.5)
        for history in ([1], [1, 4, 4, 6], [1, 4, 4, 6, 6, 1], [3, 1], [5, 1]):
            req.output_ids = list(history)
            self.check_prefixes(
                [req], torch.tensor([[1, 4, 4]]), torch.tensor([[100, 101, 102]])
            )

    def test_native_array_history_preserves_sequential_frequency_rounding(self):
        req = Request([], frequency_penalty=0.3)
        req.output_ids = array("q", [3] * 17 + [1])
        self.check_prefixes(
            [req], torch.tensor([[1, 4, 4]]), torch.tensor([[100, 101, 102]])
        )
        req.output_ids.extend([4, 4, 6])
        self.check_prefixes(
            [req], torch.tensor([[6, 4, 4]]), torch.tensor([[100, 101, 102]])
        )

    def test_replaced_base_penalties_are_restored_after_sampling_failure(self):
        batch, plan, expected = self.check_prefixes(
            [Request([3, 1], presence_penalty=1.5)],
            torch.tensor([[1, 4, 4]]),
            torch.tensor([[100, 101, 102]]),
        )
        additive, scaling = torch.full((1, 8), 9.0), torch.full((1, 8), 2.0)
        batch.sampling_info.acc_additive_penalties = additive
        batch.sampling_info.acc_scaling_penalties = scaling
        logits = torch.linspace(-4.0, 4.0, 24).reshape(3, 8)

        def sample(plan, actual_batch, output, mask, **kwargs):
            torch.testing.assert_close(
                output.next_token_logits, expected, rtol=0, atol=0
            )
            self.assertIsNone(actual_batch.sampling_info.acc_additive_penalties)
            self.assertIsNone(actual_batch.sampling_info.acc_scaling_penalties)
            raise RuntimeError("sample failed")

        with self.assertRaisesRegex(RuntimeError, "sample failed"):
            exact_eagle_sample(
                sample, plan, batch, SimpleNamespace(next_token_logits=logits)
            )
        self.assertIs(batch.sampling_info.acc_additive_penalties, additive)
        self.assertIs(batch.sampling_info.acc_scaling_penalties, scaling)

    def test_unpenalized_sampling_keeps_original_inputs(self):
        batch = Batch([Request([1])])
        output = SimpleNamespace(next_token_logits=torch.ones(1, 8))
        self.assertIs(
            exact_eagle_sample(
                lambda plan, batch, output, mask, **kwargs: output, None, batch, output
            ),
            output,
        )

    def test_warmup_sampling_has_no_penalizer_orchestrator(self):
        batch = Batch([Request([1])])
        batch.sampling_info.penalizer_orchestrator = None
        self.assertEqual(
            exact_eagle_sample(lambda *_args, **_kwargs: "warmup", None, batch, None),
            "warmup",
        )
        del batch.sampling_info.penalizer_orchestrator
        self.assertEqual(
            exact_eagle_sample(
                lambda *_args, **_kwargs: "graph-view", None, batch, None
            ),
            "graph-view",
        )

    def test_watermark_options_reach_native_sampling(self):
        batch = Batch([Request([1])])
        output = SimpleNamespace(next_token_logits=torch.ones(1, 8))
        state = object()

        def native(plan, actual_batch, actual_output, mask, **kwargs):
            return (
                kwargs["watermark_state"],
                kwargs["watermark_full_mask"],
                actual_output,
            )

        with envs.SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH.override(False):
            for idle in (False, True):
                batch.forward_mode = SimpleNamespace(is_idle=lambda idle=idle: idle)
                for watermark in (None, state):
                    with self.subTest(idle=idle, watermark=watermark is not None):
                        actual = exact_eagle_sample(
                            native,
                            None,
                            batch,
                            output,
                            watermark_state=watermark,
                            watermark_full_mask=False,
                        )
                        self.assertIs(actual[0], watermark)
                        self.assertFalse(actual[1])
                        self.assertIs(actual[2], output)

    def test_watermark_context_bypasses_sampling_graph_and_preserves_penalties(self):
        batch, plan, expected = self.check_prefixes(
            [Request([3, 1], presence_penalty=1.5)],
            torch.tensor([[1, 4, 4]]),
            torch.tensor([[100, 101, 102]]),
        )
        output = SimpleNamespace(
            next_token_logits=torch.linspace(-4.0, 4.0, 24).reshape(3, 8)
        )
        state = object()

        def native(plan, batch, output, mask, **kwargs):
            return kwargs["watermark_state"], output.next_token_logits.clone()

        with envs.SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH.override(True):
            actual_state, actual_logits = exact_eagle_sample(
                native,
                plan,
                batch,
                output,
                watermark_state=state,
            )
        self.assertIs(actual_state, state)
        torch.testing.assert_close(actual_logits, expected, rtol=0, atol=0)
        # No penalties also must delegate instead of capturing watermark state.
        batch = Batch([Request([1])])
        with envs.SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH.override(True):
            actual_state, actual_logits = exact_eagle_sample(
                native,
                None,
                batch,
                output,
                watermark_state=state,
            )
        self.assertIs(actual_state, state)
        torch.testing.assert_close(
            actual_logits, output.next_token_logits, rtol=0, atol=0
        )

    def test_simulated_acceptance_options_survive_the_adapter(self):
        """New upstream verify kwargs must reach eager sampling, never a stale graph."""
        injected = torch.tensor([2])
        generator = torch.Generator().manual_seed(531)

        def native(plan, batch, output, mask, **kwargs):
            return (
                kwargs["simulated_acc_len"],
                kwargs["simulated_acc_len_rng"],
                kwargs["sparse_bonus_argmax"],
            )

        for penalized in (False, True):
            batch = Batch([Request([1], presence_penalty=1.5 if penalized else 0)])
            plan = SimpleNamespace(
                tree_topk=1,
                draft_token_num=1,
                max_tree_depth=1,
                draft_token=torch.tensor([1]),
                positions=torch.tensor([2]),
            )
            output = SimpleNamespace(next_token_logits=torch.ones(1, 8))
            with envs.SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH.override(True):
                actual = exact_eagle_sample(
                    native,
                    plan,
                    batch,
                    output,
                    simulated_acc_len=injected,
                    simulated_acc_len_rng=generator,
                    sparse_bonus_argmax=True,
                )
            self.assertIs(actual[0], injected)
            self.assertIs(actual[1], generator)
            self.assertTrue(actual[2])

    def test_forward_copy_keeps_exact_penalties_without_the_orchestrator(self):
        batch, plan, expected = self.check_prefixes(
            [Request([3, 1], presence_penalty=1.5, min_new_tokens=5)],
            torch.tensor([[1, 4, 4]]),
            torch.tensor([[100, 101, 102]]),
        )
        info = SamplingBatchInfo(
            temperatures=torch.ones(1, 1),
            top_ps=torch.ones(1),
            top_ks=torch.ones(1, dtype=torch.int32),
            min_ps=torch.zeros(1),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=8,
            penalizer_orchestrator=batch.sampling_info.penalizer_orchestrator,
            device="cpu",
        )
        batch.sampling_info = copy_penalty_metadata(
            SamplingBatchInfo.copy_for_forward, info
        )
        self.assertIsNone(batch.sampling_info.penalizer_orchestrator)
        logits = torch.linspace(-4.0, 4.0, 24).reshape(3, 8)
        apply_linear_penalties(plan, batch, logits)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_active_penalties_reject_incomplete_or_branched_trees(self):
        for topk, depth in ((2, 3), (1, 2)):
            batch = Batch([Request([1], presence_penalty=1.5)])
            with self.assertRaisesRegex(NotImplementedError, "linear chain"):
                apply_linear_penalties(
                    SimpleNamespace(
                        tree_topk=topk, draft_token_num=3, max_tree_depth=depth
                    ),
                    batch,
                    torch.ones(3, 8),
                )

    def test_idle_sampling_does_not_require_sampling_metadata(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True), sampling_info=None
        )
        self.assertEqual(
            exact_eagle_sample(lambda *_args, **_kwargs: "idle", None, batch, None),
            "idle",
        )


if __name__ == "__main__":
    unittest.main()
