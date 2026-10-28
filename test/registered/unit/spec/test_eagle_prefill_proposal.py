"""Rejection-sampling prefill must preserve per-request greedy proposals."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")


class TestEaglePrefillProposal(CustomTestCase):
    def test_mixed_greedy_and_sampled_prefill(self):
        """The first draft ignored top_k=1 and sampled even for greedy rows.

        Only model execution and CUDA forward metadata are fixtures; token
        rotation, actual proposal sampling and output assembly run unchanged.
        A fixed RNG seed makes the pre-fix failure reproducible.
        """
        torch.manual_seed(318)
        logits = torch.tensor([[0.04, 0.0]]).repeat(64, 1)
        sampling = SimpleNamespace(
            temperatures=torch.ones(64, 1),
            sampling_seed=None,
            top_ks=torch.tensor([1] * 32 + [TOP_K_ALL] * 32, dtype=torch.int32),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.zeros(64, dtype=torch.int64),
            extend_lens=[1] * 64,
            chunked_req_next_prompt_token=None,
            sampling_info=sampling,
            seq_lens=torch.ones(64, dtype=torch.int64),
        )
        hidden = torch.zeros(64, 4)
        runner = SimpleNamespace(
            canary_manager=None,
            forward=lambda fb: SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=logits, hidden_states=hidden
                )
            ),
        )
        worker = object.__new__(EagleDraftWorker)
        worker.draft_runner = runner
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        worker.seed_dsa_topk_from_draft_extend = False
        worker.topk = 1
        with (
            get_context().override_server_args(speculative_use_rejection_sampling=True),
            # Mainline's Gumbel argmax is a GPU kernel. Exercise the real
            # Torch sampler here so this CPU regression stays device-independent.
            envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.ForwardBatch.init_new",
                return_value=SimpleNamespace(forward_mode=ForwardMode.EXTEND),
            ),
        ):
            proposal = worker._draft_extend_for_prefill(
                batch, hidden, torch.ones(64, dtype=torch.int64)
            )
        self.assertEqual(proposal.topk_index[:32].flatten().tolist(), [0] * 32)
        self.assertTrue(bool((proposal.topk_index[32:] == 1).any()))
        torch.testing.assert_close(proposal.draft_probs, logits.softmax(-1))
        torch.testing.assert_close(
            proposal.topk_p, proposal.draft_probs.gather(1, proposal.topk_index)
        )


if __name__ == "__main__":
    unittest.main()
