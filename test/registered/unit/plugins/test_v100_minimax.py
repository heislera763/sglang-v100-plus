"""MiniMax adapters preserve fixed RoPE and live-prefix verify semantics."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus import minimax


class TestMiniMaxSM70Adapters(CustomTestCase):
    def test_rope_trimming_leaves_other_dflash_models_unchanged(self):
        config = SimpleNamespace(
            hidden_size=4096,
            architectures=["DFlash2DraftModel"],
            max_position_embeddings=131072,
            rope_parameters={"rope_type": "yarn", "factor": 4},
        )
        with get_context().override_server_args(
            model_path="dummy", context_length=4096
        ):
            result = minimax.draft_attention_init(lambda _, c: c, None, config)
        self.assertIs(result, config)
        self.assertEqual(result.max_position_embeddings, 131072)

    def test_expert_encoding_divisors_are_inverted_before_shared_repack(self):
        from sglang_v100_plus import quantization

        gate_up = torch.tensor([[2.0, 2.0], [0.5, 0.5]])
        down = torch.tensor([4.0, 0.25])
        layer = torch.nn.Module()
        for name, value in (
            ("w13_weight_packed", torch.zeros(2, 4, 8, dtype=torch.uint8)),
            ("w2_weight_packed", torch.zeros(2, 8, 2, dtype=torch.uint8)),
            ("w13_weight_global_scale", gate_up),
            ("w2_weight_global_scale", down),
        ):
            layer.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )
        with patch.object(quantization, "prepare_nvfp4_moe") as prepare:
            minimax.prepare_experts(None, None, layer)
        prepare.assert_called_once_with(None, layer, group_size=16)
        torch.testing.assert_close(layer.w13_weight_scale_2, gate_up[:, 0].reciprocal())
        torch.testing.assert_close(layer.w2_weight_scale_2, down.reciprocal())
        self.assertIsNone(layer.w13_weight_packed)
        self.assertIsNone(layer.w2_weight_packed)
        self.assertFalse(layer._mega_moe_nvfp4)

    def test_small_graph_policy_does_not_disable_a_qualified_shape(self):
        from sglang.srt.speculative.draft_worker_common import draft_graph_has_memory

        model = type("DSparkMiniMaxDraftModel", (), {})()
        model.config = SimpleNamespace(hidden_size=6144)
        with get_context().override_server_args(
            model_path="dummy", max_running_requests=1
        ):
            self.assertFalse(
                draft_graph_has_memory(0.6, draft_model=model, num_tokens_per_req=7)
            )
            self.assertTrue(
                minimax.draft_graph_memory(
                    draft_graph_has_memory, 0.6, draft_model=model, num_tokens_per_req=7
                )
            )
            self.assertFalse(
                minimax.draft_graph_memory(
                    draft_graph_has_memory,
                    0.6,
                    draft_model=model,
                    num_tokens_per_req=16,
                )
            )
        with get_context().override_server_args(
            model_path="dummy", max_running_requests=2
        ):
            self.assertFalse(
                minimax.draft_graph_memory(
                    draft_graph_has_memory, 0.6, draft_model=model, num_tokens_per_req=7
                )
            )

    def test_fixed_rope_trim_does_not_change_shared_config_or_frequencies(self):
        config = SimpleNamespace(
            max_position_embeddings=1048576,
            rope_theta=1e7,
            rope_parameters={"rope_type": "default", "rope_theta": 1e7},
        )

        def construct(_, observed):
            positions = torch.arange(observed.max_position_embeddings).double()
            frequencies = observed.rope_theta ** (-torch.arange(0, 64, 2).double() / 64)
            return torch.outer(positions, frequencies).cos()

        with get_context().override_server_args(model_path="dummy", context_length=137):
            actual = minimax.attention_init(construct, None, config)
        expected = torch.outer(
            torch.arange(137).double(), 1e7 ** (-torch.arange(0, 64, 2).double() / 64)
        ).cos()
        self.assertEqual(config.max_position_embeddings, 1048576)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        with get_context().override_server_args(
            model_path="dummy",
            context_length=137,
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=8,
        ):
            verify = minimax.attention_init(construct, None, config)
        expected = torch.outer(
            torch.arange(145).double(), 1e7 ** (-torch.arange(0, 64, 2).double() / 64)
        ).cos()
        torch.testing.assert_close(verify, expected, rtol=0, atol=0)
        self.assertEqual(config.max_position_embeddings, 1048576)

        config.rope_parameters = {"rope_type": "dynamic", "factor": 4}
        with get_context().override_server_args(model_path="dummy", context_length=137):
            with self.assertRaisesRegex(ValueError, "fixed frequencies"):
                minimax.attention_init(construct, None, config)

    def test_verify_uses_live_prefix_despite_generic_graph_placeholder(self):
        torch.manual_seed(531)
        rows, capacity = 8, 256
        q = torch.randn(rows, 8, 128, dtype=torch.float16)
        keys = torch.randn(capacity, 1, 128, dtype=torch.float16)
        values = torch.randn_like(keys)
        pool = SimpleNamespace(
            layer_router=SimpleNamespace(
                read=lambda _: SimpleNamespace(kv=lambda: (keys, values))
            ),
            get_index_k_buffer=lambda _: keys,
        )
        backend = SimpleNamespace(
            kv_pool=pool,
            disable_value_layer_ids={0},
            _is_sparse_kv_cached_by_fusion=lambda *_: True,
            req_to_token=torch.arange(capacity, dtype=torch.int32)[None],
            block_size_q=1,
            block_size_k=128,
            topk_blocks=16,
            init_blocks=0,
            local_blocks=1,
            score_type="max",
        )
        layer = SimpleNamespace(
            layer_id=0,
            **{
                s: None
                for s in (
                    "q_scale_float",
                    "k_scale_float",
                    "v_scale_float",
                    "idx_q_scale_float",
                    "idx_k_scale_float",
                    "idx_v_scale_float",
                )
            },
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: True),
            seq_lens=torch.tensor([129], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            spec_info=SimpleNamespace(custom_mask=torch.ones(64), draft_token_num=8),
            extend_seq_lens=None,
        )

        def dense_reference(_original, query, kc, vc, _sink, *args, **kwargs):
            lengths, prefix = args[7:9]
            result = torch.empty_like(query)
            for row in range(rows):
                end = min(int(lengths[0]), int(prefix[0]) + row + 1)
                score = query[row].double() @ kc[:end, 0].double().T * 128**-0.5
                result[row] = (score.softmax(-1) @ vc[:end, 0].double()).half()
            return None, result

        with (
            get_context().override_server_args(
                model_path="dummy",
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=8,
            ),
            patch.object(minimax, "sparse_prefill", dense_reference),
        ):
            for history in (129, 173):
                batch.seq_lens.fill_(history)
                _, actual = minimax.sparse_extend(
                    None,
                    backend,
                    q,
                    None,
                    None,
                    layer,
                    batch,
                    idx_q=q[:, :1],
                    idx_k=None,
                    idx_v=None,
                )
                expected = torch.stack(
                    [
                        (
                            (
                                q[row].double()
                                @ keys[: history + row + 1, 0].double().T
                                * 128**-0.5
                            ).softmax(-1)
                            @ values[: history + row + 1, 0].double()
                        ).half()
                        for row in range(rows)
                    ]
                )
                torch.testing.assert_close(
                    actual, expected.reshape(rows, -1), rtol=0, atol=0
                )
                self.assertEqual(batch.seq_lens.item(), history)
                self.assertIsNone(batch.extend_seq_lens)


if __name__ == "__main__":
    unittest.main()
