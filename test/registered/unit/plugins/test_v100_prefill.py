"""HC token partitions must preserve row ownership and forward lifetime."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus import prefill
    from sglang_v100_plus.dflash import SM70DFlashResidualNorm, forward_draft
    from sglang_v100_plus.dispatch import prefill_scope
    from sglang_v100_plus.glm_mhc import mhc_pre_dispatch
    from sglang_v100_plus.hc_state import SM70GatedResidualState


class TestV100Prefill(CustomTestCase):
    def test_glm_paired_mhc_keeps_full_rows_without_gathering(self):
        """Active chunk scheduling uses replicated PRE without a local gather.

        The chunk's full token rows and optional normalization must reach the
        native dispatch unchanged. A leaked schedule context would also make
        later ordinary prefill bypass the separately tested partition path.
        """
        from sglang_v100_plus import paired_prefill

        value = torch.arange(1024, dtype=torch.float16)[:, None, None]
        value = value.expand(1024, 4, 4096).contiguous()
        weight = torch.ones(4096, dtype=torch.float16)

        def full(residual, _fn, _scale, _base, *_args):
            norm_weight, norm_eps = _args[-2:]
            layer = residual[:, 0] * norm_weight + norm_eps
            return residual[:, :, :1].float(), residual[:, :, :4].float(), layer, True

        fields = dict(
            model_path="dummy",
            tp_size=8,
            pp_size=1,
            max_running_requests=1,
            disable_overlap_schedule=True,
            disable_prefill_cuda_graph=True,
        )
        with (
            get_context().override_server_args(**fields),
            envs.SGLANG_OPT_SM70_HC_PREFILL_SP.override(True),
            prefill_scope(True),
            patch(
                "sglang_v100_plus.prefill.quad_partition",
                side_effect=AssertionError("paired PRE attempted a gather"),
            ),
        ):
            slot = paired_prefill._slot()
            token = slot.set(object())
            try:
                post, comb, layer, norm_fused = mhc_pre_dispatch(
                    full,
                    value,
                    None,
                    None,
                    None,
                    1e-5,
                    1e-6,
                    1e-6,
                    2.0,
                    20,
                    weight,
                    0.5,
                )
                self.assertTrue(norm_fused)
                self.assertEqual(post.shape, (1024, 4, 1))
                self.assertEqual(comb.shape, (1024, 4, 4))
                torch.testing.assert_close(layer, value[:, 0] + 0.5)
            finally:
                slot.reset(token)
            self.assertFalse(paired_prefill.is_active())

    def test_glm_mhc_restores_fp32_coefficients_and_trims_padded_tokens(self):
        """POST needs dense coefficient strides and matching original token IDs.

        Leaving views of the packed 20-float rows, gathering in channel-major
        order or exposing padded tail tokens silently corrupts the write-back.
        Normalization must happen once, before restoring the complete input.
        """
        rows = 1025
        local_rows = (rows + 3) // 4
        residual = (
            torch.arange(rows, dtype=torch.float16)[:, None, None]
            .expand(rows, 4, 4096)
            .contiguous()
        )
        saved = residual.clone()
        channel = torch.arange(4, dtype=torch.float32)

        def operator(value, *_args):
            token = value[:, 0, 0].float()
            post = (token[:, None] + channel).unsqueeze(-1)
            comb = token[:, None, None] + channel[:, None] * 4 + channel[None, :]
            layer = value[:, 0].clone()
            return post, comb, layer

        def normalizer(value, _weight, _eps):
            return value + 0.5

        def shard_operator(value, *_args, rank, size):
            self.assertIs(value, residual)
            return operator(prefill.Partition(rows, rank, size).local(value))

        expected_post, expected_comb, expected_layer = operator(residual)
        expected_layer = normalizer(expected_layer, None, None)
        expected_coefficients = torch.cat(
            (expected_post.reshape(rows, 4), expected_comb.reshape(rows, 16)), dim=1
        )

        class Group:
            def __init__(self, rank):
                self.rank = rank

            def all_gather(group, value, dim):
                self.assertEqual(dim, 0)
                plan = prefill.Partition(rows, group.rank)
                local_post, local_comb, local_layer = operator(plan.local(residual))
                if value.shape[1] == 20:
                    expected_local = torch.cat(
                        (
                            local_post.reshape(local_rows, 4),
                            local_comb.reshape(local_rows, 16),
                        ),
                        dim=1,
                    )
                    full = expected_coefficients
                else:
                    expected_local = normalizer(local_layer, None, None)
                    full = expected_layer
                torch.testing.assert_close(value, expected_local, rtol=0, atol=0)
                # Poison the gathered padding so a missing trim is detectable.
                return torch.nn.functional.pad(full, (0, 0, 0, 3), value=123)

        fields = dict(
            model_path="dummy",
            tp_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
            disable_prefill_cuda_graph=True,
        )
        with (
            get_context().override_server_args(**fields),
            envs.SGLANG_OPT_SM70_HC_PREFILL_SP.override(True),
            prefill_scope(True),
            patch(
                "sglang.kernels.ops.layernorm.mhc_sm70.mhc_pre_shard_sm70",
                shard_operator,
            ),
            patch("sgl_kernel.rmsnorm", side_effect=normalizer) as norm,
        ):
            for rank in range(4):
                with get_parallel().override(
                    attn_tp_size=4, attn_tp_rank=rank, attn_tp_group=Group(rank)
                ):
                    post, comb, layer, norm_fused = mhc_pre_dispatch(
                        lambda *_args: self.fail("quad prefill was bypassed"),
                        residual,
                        None,
                        None,
                        None,
                        1e-5,
                        1e-6,
                        1e-6,
                        2.0,
                        20,
                        torch.ones(4096, dtype=torch.float16),
                        1e-5,
                    )
                    self.assertTrue(norm_fused)
                    for got, want in zip(
                        (post, comb, layer),
                        (expected_post, expected_comb, expected_layer),
                    ):
                        self.assertTrue(got.is_contiguous())
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
                    self.assertEqual(post.dtype, torch.float32)
                    self.assertEqual(comb.dtype, torch.float32)
                    self.assertEqual(layer.dtype, torch.float16)
                    self.assertEqual(norm.call_args.args[0].shape, (local_rows, 4096))
            self.assertEqual(norm.call_count, 4)
        torch.testing.assert_close(residual, saved, rtol=0, atol=0)

    def test_glm_mhc_verification_and_graphs_do_not_partition(self):
        """Short prompts, disabled partitions, verification and graphs stay full."""
        fields = dict(
            model_path="dummy",
            tp_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
            disable_prefill_cuda_graph=True,
        )
        for changes, prefill_active, rows, enabled in (
            ({}, False, 1024, True),
            ({"disable_prefill_cuda_graph": False}, True, 1024, True),
            ({"max_running_requests": 2}, True, 1024, True),
            ({"disable_overlap_schedule": False}, True, 1024, True),
            ({}, True, 1023, True),
            ({}, True, 1024, False),
        ):
            value = torch.empty(rows, 4, 4096, dtype=torch.float16)
            with envs.SGLANG_OPT_SM70_HC_PREFILL_SP.override(enabled):
                with (
                    self.subTest(changes=changes, prefill=prefill_active),
                    get_context().override_server_args(**(fields | changes)),
                    prefill_scope(prefill_active),
                    patch("sglang_v100_plus.prefill.quad_partition") as partition,
                ):
                    marker = object()
                    original = unittest.mock.Mock(return_value=marker)
                    self.assertIs(
                        mhc_pre_dispatch(
                            original,
                            value,
                            None,
                            None,
                            None,
                            1e-5,
                            1e-6,
                            1e-6,
                            2.0,
                            20,
                            None,
                            None,
                        ),
                        marker,
                    )
                    partition.assert_not_called()
                    original.assert_called_once()

    def test_fp8_route_alignment_matches_packing_and_native_tile(self):
        """The native row tile applies to either local expert bank/layout."""
        from sglang_v100_plus.fp8 import fp8_route_block_size

        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            select_marlin_moe_block_size,
        )

        hidden = SimpleNamespace(
            shape=(4352, 2560), ndim=2, dtype=torch.float16, is_cuda=True
        )
        ids = SimpleNamespace(shape=(4352, 10), dtype=torch.int32)
        w1 = SimpleNamespace(shape=(128, 160, 5120))
        w2 = SimpleNamespace(shape=(128, 40, 10240))
        scales = SimpleNamespace(dtype=torch.float16, _sm70_fp8_scale=True)
        fields = dict(
            model_path="dummy",
            tp_size=4,
            ep_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
        )
        for changes, tuning, tagged, expected in (
            ({}, False, True, 32),
            ({"ep_size": 1}, False, True, 32),
            ({"tp_size": 8, "ep_size": 8}, False, True, 32),
            ({"pp_size": 1}, False, True, 32),
            ({"max_running_requests": 2}, False, True, 64),
            ({}, True, True, 64),
            ({}, False, False, 64),
        ):
            with (
                self.subTest(changes=changes, tuning=tuning, tagged=tagged),
                get_context().override_server_args(**(fields | changes)),
                patch(
                    "sglang_v100_plus.kernels.moe_marlin._sm70_marlin_user_tuning",
                    tuning,
                ),
            ):
                scales._sm70_fp8_scale = tagged
                actual = fp8_route_block_size(
                    select_marlin_moe_block_size, hidden, ids, w1, w2, scales
                )
                self.assertEqual(actual, expected)
        with get_context().override_server_args(**fields):
            scales._sm70_fp8_scale = True
            for experts, down_experts, width, expected in (
                (64, 64, 5120, 32),
                (128, 128, 5120, 32),
                (64, 128, 5120, 64),
                (64, 64, 1280, 64),
            ):
                with self.subTest(
                    experts=experts, down_experts=down_experts, width=width
                ):
                    w1.shape = (experts, 160, width)
                    w2.shape = (down_experts, 40, 10240)
                    self.assertEqual(
                        fp8_route_block_size(
                            select_marlin_moe_block_size, hidden, ids, w1, w2, scales
                        ),
                        expected,
                    )

    def test_dflash_residual_addition_does_not_overflow_before_norm(self):
        norm = SM70DFlashResidualNorm(
            SimpleNamespace(
                weight=torch.nn.Parameter(torch.ones(4, dtype=torch.float16)),
                variance_epsilon=1e-5,
            )
        )
        x = torch.full((2, 4), 65504.0, dtype=torch.float16)
        self.assertFalse(torch.isfinite(x + x).all())
        output, residual = norm(x, x)
        self.assertEqual(residual.dtype, torch.float32)
        torch.testing.assert_close(
            residual, torch.full((2, 4), 131008.0), rtol=0, atol=0
        )
        torch.testing.assert_close(output, torch.ones_like(x), rtol=0, atol=0)

    def test_dflash_eager_entry_keeps_fp16_embedding_bits(self):
        embeddings = torch.tensor(
            [[1.0009765625, 1.0029296875, -1.0009765625]], dtype=torch.float16
        )
        rounded = embeddings.bfloat16()
        self.assertFalse(torch.equal(rounded.half(), embeddings))
        batch = SimpleNamespace(input_embeds=embeddings)
        actual = forward_draft(
            lambda owner, ids, positions, fb, input_embeds: input_embeds,
            None,
            None,
            None,
            batch,
            input_embeds=rounded,
        )
        self.assertTrue(
            torch.equal(actual.view(torch.int16), embeddings.view(torch.int16))
        )

    def test_boundary_keeps_each_stages_cached_gate_and_clears_it(self):
        """An absent gate must still reach mainline's three-value HC consumer."""
        from sglang.srt.layers.hyperconnection import GatedResidual

        residual = torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12)
        gate_a, gate_b = torch.full((4, 3), 2.0), torch.full((4, 3), 3.0)
        seen = []

        def mix_with_gate(gate):
            return lambda value: (value[:, :4], (value, value * 0.5, gate))

        def combine(output, values):
            seen.append(values)
            owner = SimpleNamespace(
                combine=lambda block, state: state[0] + block.repeat(1, 3)
            )
            updated, normed = GatedResidual.combine_and_normalize(
                owner, output, values, None
            )
            self.assertIsNone(normed)
            torch.testing.assert_close(updated, values[0] + output.repeat(1, 3))
            return updated

        state = SM70GatedResidualState(
            expand=lambda value: value,
            attn_mix=mix_with_gate(gate_a),
            ffn_mix=mix_with_gate(gate_b),
            attn_combine=combine,
            ffn_combine=combine,
        )
        ops = state.residual_ops()
        hidden, residual = ops.attn_readout.read(residual, None)
        _, residual = ops.ffn_readout.update_and_read(
            ops.attn_update, hidden, residual, None
        )
        self.assertIs(seen[0][2], gate_a)
        self.assertIs(state.gate, gate_b)
        ops.ffn_update.update(hidden, residual)
        self.assertIs(seen[1][2], gate_b)
        self.assertIsNone(state.gate)
        self.assertIsNone(state.normed)
        # A subsequent prefill/empty read returns the ordinary pair and must
        # never reuse a gate from an earlier captured or speculative read.
        state.gate = gate_a
        state.attn_mix = lambda value: (value[:, :4], (value, value * 0.25))
        hidden, residual = ops.attn_readout.read(residual, None)
        ops.attn_update.update(hidden, residual)
        self.assertIsNone(state.gate)
        self.assertEqual(len(seen[-1]), 3)
        self.assertIsNone(seen[-1][2])

    def test_boundary_slices_gate_rows_with_normalized_residuals(self):
        residual = torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12)
        normed = residual * 0.5
        gate = torch.arange(4 * 8 * 3).reshape(4, 8, 3)
        state = SM70GatedResidualState(
            expand=None,
            attn_mix=None,
            ffn_mix=None,
            attn_combine=None,
            ffn_combine=None,
            normed=normed,
            gate=gate,
        )
        parallel = SimpleNamespace(attn_tp_size=2, attn_tp_rank=1)
        with (
            patch("sglang_v100_plus.hc_state.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.layers.layer_boundary.residual.gated.get_parallel",
                return_value=parallel,
            ),
        ):
            selected = state.residual_ops().attn_update.slice_residual_attn_tp(residual)
        torch.testing.assert_close(selected, residual[2:])
        torch.testing.assert_close(state.normed, normed[2:])
        torch.testing.assert_close(state.gate, gate[2:])

    def test_ple_preparation_gathers_before_the_causal_convolution(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        full = torch.arange(256 * 5).reshape(256, 5)
        batch = SimpleNamespace(residual_stream=None)
        ple_batch = object()
        owner = SimpleNamespace(ple=object())

        class Group:
            def all_gather(_, value, dim):
                torch.testing.assert_close(value, full[64:128])
                self.assertEqual(dim, 0)
                return full

        def original(layer, hidden, actual_batch, actual_ple_batch):
            self.assertIs(layer, owner)
            self.assertIs(actual_batch, batch)
            self.assertIs(actual_ple_batch, ple_batch)
            residual_batch.stream_of(batch).check(hidden)
            return hidden

        with (
            get_context().override_server_args(model_path="dummy"),
            get_parallel().override(attn_tp_group=Group()),
            prefill.partition_scope(prefill.Partition(256, 1)),
        ):
            local = full[64:128]
            residual_batch.set_written(local, batch)
            actual = prefill.prepare_attention(original, owner, local, batch, ple_batch)
            self.assertIs(actual, full)
            owner.ple = None
            local = full[64:128]
            residual_batch.set_written(local, batch)
            self.assertIs(
                prefill.prepare_attention(original, owner, local, batch, ple_batch),
                local,
            )

    def test_only_eager_serialized_tp4_prefill_is_partitioned(self):
        """Only ordinary/MTP eager prefill may partition; verification stays full."""
        owner = SimpleNamespace(hc_count=4, hidden_size=2560)
        batch = SimpleNamespace(
            batch_size=1, forward_mode=SimpleNamespace(is_extend=lambda: True)
        )
        fields = {
            "model_path": "dummy",
            "quantization": "fp8",
            "tp_size": 4,
            "pp_size": 2,
            "max_running_requests": 1,
            "disable_overlap_schedule": True,
            "disable_prefill_cuda_graph": True,
        }
        with envs.SGLANG_OPT_SM70_HC_PREFILL_SP.override(True), prefill_scope(True):
            for changes in (
                {},
                {"pp_size": 1},
                {"tp_size": 8},
                {"quantization": "modelopt_fp4"},
                {"quantization": "modelopt_fp4", "pp_size": 1},
                {"pp_size": 3},
                {"max_running_requests": 2},
                {"disable_prefill_cuda_graph": False},
                {"enable_return_hidden_states": True},
                {"speculative_algorithm": "EAGLE"},
                {"speculative_algorithm": "DFLASH"},
            ):
                with (
                    self.subTest(changes=changes),
                    get_context().override_server_args(**(fields | changes)),
                    get_parallel().override(
                        attn_tp_size=changes.get("tp_size", 4), attn_tp_rank=2
                    ),
                ):
                    for rows in (128, 256, 257):
                        actual = prefill._partition(owner, torch.empty(rows), batch)
                        if rows >= 256 and changes in (
                            {},
                            {"pp_size": 1},
                            {"quantization": "modelopt_fp4"},
                            {"quantization": "modelopt_fp4", "pp_size": 1},
                            {"speculative_algorithm": "EAGLE"},
                        ):
                            self.assertEqual(actual, prefill.Partition(rows, 2))
                        else:
                            self.assertIsNone(actual)

    def test_partitions_gather_in_original_order_and_do_not_leak(self):
        """Offsets cannot cross ranks or survive a failed/nested forward."""
        full = torch.arange(256 * 5).reshape(256, 5)
        with get_context().override_server_args(model_path="dummy"):
            for rank in range(4):
                plan = prefill.Partition(256, rank)
                expected = full.tensor_split(4)[rank]

                class Group:
                    def all_gather(_, value, dim, expected=expected):
                        torch.testing.assert_close(value, expected)
                        self.assertEqual(dim, 0)
                        return full.clone()

                with get_parallel().override(attn_tp_group=Group()):
                    with prefill.partition_scope(plan):
                        local = prefill.local_input(full)
                        torch.testing.assert_close(local, expected)
                        self.assertIs(prefill.local_input(local), local)
                        torch.testing.assert_close(prefill.gather_input(local), full)
                        with prefill.partition_scope(None):
                            self.assertIs(prefill.local_input(full), full)
                        torch.testing.assert_close(prefill.local_input(full), expected)
                        with self.assertRaisesRegex(ValueError, "partition"):
                            prefill.local_input(full[:17])
                    self.assertIs(prefill.local_input(full), full)
            with (
                self.assertRaisesRegex(RuntimeError, "failed"),
                prefill.partition_scope(prefill.Partition(256, 0)),
            ):
                raise RuntimeError("failed forward")
            self.assertIs(prefill.gather_input(full), full)

    def test_tp8_hc_uses_cached_quad_groups_without_changing_model_tp(self):
        """Both quads own all tokens; subgroup lifetime follows its parent world."""
        owner = SimpleNamespace(hc_count=4, hidden_size=2560)
        batch = SimpleNamespace(
            batch_size=1, forward_mode=SimpleNamespace(is_extend=lambda: True)
        )
        fields = dict(
            model_path="dummy",
            tp_size=8,
            ep_size=8,
            pp_size=1,
            max_running_requests=1,
            disable_overlap_schedule=True,
            disable_prefill_cuda_graph=True,
        )
        with (
            get_context().override_server_args(**fields),
            envs.SGLANG_OPT_SM70_HC_PREFILL_SP.override(True),
            prefill_scope(True),
        ):
            for rank in range(8):
                parent = SimpleNamespace(ranks=list(range(8)), device_group=object())
                world = SimpleNamespace(ranks=list(range(8)), local_rank=rank)
                group = SimpleNamespace(rank_in_group=rank % 4)
                with (
                    self.subTest(rank=rank),
                    get_parallel().override(
                        attn_tp_size=8,
                        attn_tp_rank=rank,
                        attn_tp_group=parent,
                        world_group=world,
                    ),
                    patch("torch.distributed.get_backend", return_value="nccl"),
                    patch(
                        "sglang.srt.distributed.init_model_parallel_group",
                        return_value=group,
                    ) as create,
                ):
                    for rows in (257, 4352):
                        plan = prefill._partition(owner, torch.empty(rows), batch)
                        self.assertEqual(
                            plan, prefill.Partition(rows, rank % 4, group=group)
                        )
                    create.assert_called_once_with(
                        [list(range(4)), list(range(4, 8))],
                        local_rank=rank,
                        backend="nccl",
                        use_pynccl=True,
                        use_custom_allreduce=False,
                        use_mscclpp=False,
                        use_torch_symm_mem_allreduce=False,
                        group_name="v100_hc_prefill",
                    )
                    self.assertIs(get_parallel().attn_tp_group, parent)
                    self.assertEqual(get_parallel().attn_tp_size, 8)
                    world.ranks = list(range(16))
                    with self.assertRaisesRegex(RuntimeError, "complete TP8 world"):
                        prefill._partition(owner, torch.empty(257), batch)

    def test_indexer_reuses_only_the_matching_dcp_quad(self):
        """Physical selected IDs may cross replicated DCP ranks, never quads.

        A TP8 gather or an interleaved DCP group would silently feed another
        quad's page IDs to attention. Odd row counts must trim exchange padding.
        """
        fields = dict(
            model_path="dummy",
            tp_size=8,
            dcp_size=4,
            max_running_requests=1,
            disable_overlap_schedule=True,
        )
        with get_context().override_server_args(**fields):
            for rank, rows, pooled_keys in (
                (r, n, k)
                for r in range(8)
                for n, k in (
                    (64, 32768),
                    (65, 32768),
                    (128, 16384),
                    (256, 8192),
                    (513, 0),
                )
            ):
                quad = rank // 4
                full = torch.arange(rows * 3).reshape(rows, 3) + quad * 10000
                local_rows = (rows + 3) // 4
                padded = torch.nn.functional.pad(full, (0, 0, 0, local_rows * 4 - rows))

                class Group:
                    ranks = list(range(quad * 4, quad * 4 + 4))
                    rank_in_group = rank % 4

                    def all_gather(group, value, dim):
                        self.assertEqual(dim, 0)
                        first = group.rank_in_group * local_rows
                        torch.testing.assert_close(
                            value, padded[first : first + local_rows]
                        )
                        gathered = padded.clone()
                        gathered[rows:] = -777
                        return gathered

                group = Group()
                parent = SimpleNamespace(ranks=list(range(8)))
                with get_parallel().override(
                    attn_tp_size=8,
                    attn_tp_rank=rank,
                    attn_tp_group=parent,
                    dcp_group=group,
                ):
                    plan = prefill.indexer_partition(rows, pooled_keys)
                    self.assertIs(plan.group, group)
                    torch.testing.assert_close(plan.gather(plan.local(full)), full)
                    self.assertIsNone(prefill.indexer_partition(511))
                    self.assertIsNone(prefill.indexer_partition(63, 1 << 20))
                    self.assertIsNone(prefill.indexer_partition(64, 16384))
                    self.assertIsNone(prefill.indexer_partition(64, 32767))
                    self.assertIsNone(prefill.indexer_partition(128, 16383))
                    self.assertIsNone(prefill.indexer_partition(256, 8191))
                    group.ranks = [0, 2, 4, 6]
                    with self.assertRaisesRegex(RuntimeError, "consecutive"):
                        prefill.indexer_partition(513)
                    for changed in (
                        dict(attn_dcp_size=2),
                        dict(attn_tp_size=4, attn_cp_size=2, attn_tp_rank=rank % 4),
                        dict(pp_size=2),
                    ):
                        with get_parallel().override(**changed):
                            self.assertIsNone(prefill.indexer_partition(513))

    def test_explicit_quad_gather_cannot_mix_other_quads_or_use_model_tp(self):
        """Distinct quad data detects an accidental world gather, including PLE."""
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        class ModelGroup:
            def all_gather(*_args, **_kwargs):
                raise AssertionError("HC gather used the model TP8 group")

        with (
            get_context().override_server_args(model_path="dummy", tp_size=8),
            get_parallel().override(attn_tp_group=ModelGroup()),
        ):
            for quad in range(2):
                full = torch.arange(257 * 5).reshape(257, 5) + quad * 10000
                padded = torch.cat((full, full.new_zeros(3, 5)))
                for rank in range(4):
                    expected = padded[rank * 65 : (rank + 1) * 65]

                    class QuadGroup:
                        def all_gather(_, value, dim):
                            self.assertEqual(dim, 0)
                            torch.testing.assert_close(value, expected, rtol=0, atol=0)
                            return padded

                    plan = prefill.Partition(257, rank, group=QuadGroup())
                    with prefill.partition_scope(plan):
                        local = prefill.local_input(full)
                        torch.testing.assert_close(local, expected, rtol=0, atol=0)
                        torch.testing.assert_close(
                            prefill.gather_input(local), full, rtol=0, atol=0
                        )
                        batch = SimpleNamespace(residual_stream=None)
                        residual_batch.set_written(local, batch)
                        gathered = prefill.prepare_attention(
                            lambda owner, hidden, fb, ple: hidden,
                            SimpleNamespace(ple=object()),
                            local,
                            batch,
                            None,
                        )
                        torch.testing.assert_close(gathered, full, rtol=0, atol=0)
                        residual_batch.stream_of(batch).check(gathered)

    def test_ragged_partitions_pad_only_hc_and_trim_before_consumers(self):
        """Equal collective counts must retain every real row, including short tails."""
        with get_context().override_server_args(model_path="dummy"):
            for rows in (257, 325, 3994, 4165):
                with self.subTest(rows=rows):
                    full = torch.arange(rows * 12).reshape(rows, 4, 3)
                    count = (rows + 3) // 4
                    parts = [
                        prefill.Partition(rows, rank).local(full) for rank in range(4)
                    ]
                    padded = torch.cat(parts)
                    self.assertEqual(padded.shape[0], 4 * count)
                    torch.testing.assert_close(padded[:rows], full, rtol=0, atol=0)
                    self.assertTrue(
                        torch.equal(padded[rows:], torch.zeros_like(padded[rows:]))
                    )

                    class Group:
                        def all_gather(_, value, dim):
                            self.assertEqual(dim, 0)
                            self.assertEqual(value.shape[0], count)
                            return padded

                    for rank, part in enumerate(parts):
                        with (
                            get_parallel().override(attn_tp_group=Group()),
                            prefill.partition_scope(prefill.Partition(rows, rank)),
                        ):
                            self.assertIs(prefill.local_input(part), part)
                            torch.testing.assert_close(
                                prefill.gather_input(part), full, rtol=0, atol=0
                            )
        self.assertIs(prefill.gather_input(full), full)

    def test_inactive_forwards_hide_parent_partition_and_restore_it(self):
        """A decode/nested inactive call cannot inherit the previous prefill's rows."""
        full = torch.arange(256).reshape(256, 1)
        owner = SimpleNamespace()
        batch = SimpleNamespace()
        with (
            get_context().override_server_args(model_path="dummy"),
            envs.SGLANG_OPT_SM70_HC_PREFILL_SP.override(False),
            prefill.partition_scope(prefill.Partition(256, 2)),
        ):

            def original(*_args, **_kwargs):
                self.assertIs(prefill.local_input(full), full)
                raise RuntimeError("nested failure")

            with self.assertRaisesRegex(RuntimeError, "nested failure"):
                prefill.model_forward(original, owner, full, None, batch)
            torch.testing.assert_close(prefill.local_input(full), full[128:192])


if __name__ == "__main__":
    unittest.main()
