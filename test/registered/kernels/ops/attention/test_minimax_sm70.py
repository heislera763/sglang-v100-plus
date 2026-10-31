"""Volta block selection/attention preserve causal FP16 KV and request reuse."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="nightly", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0),
    "Requires Volta",
)
class TestMiniMaxSM70(CustomTestCase):
    def test_selected_blocks_keep_causal_local_block_and_padding(self):
        from sglang_v100_plus.minimax import _select_blocks

        torch.manual_seed(531)
        rows, context = 17, 4352
        q = torch.randn(rows, 1, 128, device="cuda", dtype=torch.float16)
        keys = torch.randn(context, 1, 128, device="cuda", dtype=torch.float16)
        mapping = torch.stack(
            [torch.arange(context, device="cuda", dtype=torch.int32)] * 2
        )
        mapping[1] = mapping[1].flip(0)
        slot = torch.tensor([1], device="cuda", dtype=torch.int32)
        for history in (0, 127, 2200, 4200):
            with self.subTest(history=history):
                prefix = torch.tensor([history], device="cuda", dtype=torch.int32)
                lengths = prefix + rows
                physical, chosen = _select_blocks(
                    q, keys, mapping, slot, prefix, lengths, history + rows, 16, 0, 1
                )
                self.assertTrue(torch.equal(physical, mapping[1]))
                expected = torch.full_like(chosen, -1)
                for row in range(rows):
                    count = history + row + 1
                    dots = (
                        q[row, 0].double()
                        @ keys[mapping[1, :count].long(), 0].double().T
                    )
                    dots = torch.nn.functional.pad(
                        dots, (0, (-count) % 128), value=-torch.inf
                    )
                    scores = dots.view(-1, 128).amax(-1)
                    scores[-1] = torch.inf  # The current block is always retained.
                    ids = scores.topk(min(16, scores.numel())).indices.sort().values
                    expected[0, row, : ids.numel()] = ids.int()
                torch.testing.assert_close(chosen, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _, replayed = _select_blocks(
                        q,
                        keys,
                        mapping,
                        slot,
                        prefix,
                        lengths,
                        history + rows,
                        16,
                        0,
                        1,
                    )
                slot.zero_()
                prefix.add_(128)
                lengths.add_(128)
                graph.replay()
                _, fresh = _select_blocks(
                    q,
                    keys,
                    mapping,
                    slot,
                    prefix,
                    lengths,
                    history + rows + 128,
                    16,
                    0,
                    1,
                )
                torch.testing.assert_close(replayed, fresh, rtol=0, atol=0)
                slot.fill_(1)

    def test_draft_window_keeps_query_specific_left_and_configured_causality(self):
        from sglang.kernels.ops.attention.minimax_sparse.sm70 import (
            minimax_window_attention_sm70,
        )

        torch.manual_seed(531)
        for rows, history, window, causal in [
            (7, 0, 4095, False),
            (8, 139, 2047, False),
            (8, 4200, 4095, False),
            (8, 139, 4095, True),
            (8, 4200, 4095, True),
        ]:
            with self.subTest(rows=rows, history=history, causal=causal):
                heads, cap = 8, 4352
                q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.float16)
                k = torch.randn(rows, 1, 128, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                kc = torch.randn(cap, 1, 128, device="cuda", dtype=torch.float16)
                vc = torch.randn_like(kc)
                ids = torch.randperm(cap, device="cuda", dtype=torch.int64)[
                    :history
                ].contiguous()
                ptr = torch.tensor([0, history], device="cuda", dtype=torch.int32)
                result = minimax_window_attention_sm70(
                    q, k, v, kc, vc, ids, ptr, window, causal=causal
                )
                full_k = torch.cat([kc[ids, 0], k[:, 0]]).double()
                full_v = torch.cat([vc[ids, 0], v[:, 0]]).double()
                expected = torch.empty_like(q)
                for row in range(rows):
                    start = max(0, history + row - window)
                    end = history + row + 1 if causal else history + rows
                    scores = q[row].double() @ full_k[start:end].T * 128**-0.5
                    expected[row] = (scores.softmax(-1) @ full_v[start:end]).half()
                torch.testing.assert_close(result, expected, rtol=0.001, atol=0.0002)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = minimax_window_attention_sm70(
                        q, k, v, kc, vc, ids, ptr, window, causal=causal
                    )
                ptr.fill_(0)
                k.mul_(0.5)
                graph.replay()
                fresh = minimax_window_attention_sm70(
                    q, k, v, kc, vc, ids, ptr, window, causal=causal
                )
                torch.testing.assert_close(replayed, fresh, rtol=0, atol=0)

    def test_fp16_grouped_norm_rope_keeps_value_heads(self):
        from sglang.kernels.ops.attention.minimax_qknorm_rope import (
            minimax_qknorm_rope_grouped,
        )

        torch.manual_seed(531)
        rows = 7
        packed = torch.randn(rows, 12, 128, device="cuda", dtype=torch.float16)
        before = packed.clone()
        groups = [
            (torch.randn(128, device="cuda", dtype=torch.float16), off, count)
            for off, count in [(0, 8), (8, 1), (10, 1), (11, 1)]
        ]
        positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 129
        angle = torch.randn(1024, 32, device="cuda", dtype=torch.float32)
        cache = torch.cat([angle.cos(), angle.sin()], -1)
        expected = before.double()
        for weight, off, count in groups:
            x = before[:, off : off + count].double()
            x = (
                x
                * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + weight.double())
            )
            cs = cache[positions].double()
            first, second = x[..., :32].clone(), x[..., 32:64].clone()
            x[..., :32] = first * cs[:, None, :32] - second * cs[:, None, 32:]
            x[..., 32:64] = second * cs[:, None, :32] + first * cs[:, None, 32:]
            expected[:, off : off + count] = x
        minimax_qknorm_rope_grouped(
            packed.reshape(rows, -1), groups, cache, positions, 1e-6
        )
        torch.testing.assert_close(packed, expected.half(), rtol=0.001, atol=0.0002)
        self.assertTrue(torch.equal(packed[:, 9], before[:, 9]))

    def test_index_scores_mask_future_and_reused_pages(self):
        from sglang.kernels.ops.attention.minimax_sparse.sm70 import (
            minimax_index_scores_sm70,
        )

        torch.manual_seed(531)
        capacity, context, rows = 640, 515, 17
        keys = torch.randn(capacity, 1, 128, device="cuda", dtype=torch.float16)
        q = torch.randn(rows, 1, 128, device="cuda", dtype=torch.float16)
        slots = torch.randperm(capacity, device="cuda", dtype=torch.int32)[
            :context
        ].contiguous()
        prefix = torch.tensor([397], device="cuda", dtype=torch.int32)
        length = prefix + rows
        output = minimax_index_scores_sm70(q, keys, slots, prefix, length, 5)
        dot = q[:, 0].double() @ keys[slots.long(), 0].double().T
        dot *= 128**-0.5 * 1.4426950408889634
        pos = torch.arange(context, device="cuda")
        mask = pos[None] <= prefix[0] + torch.arange(rows, device="cuda")[:, None]
        dot.masked_fill_(~mask, -torch.inf)
        dot = torch.nn.functional.pad(dot, (0, 640 - context), value=-torch.inf)
        expected = dot.reshape(rows, 5, 128).amax(-1).float()[None]
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=2e-5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = minimax_index_scores_sm70(q, keys, slots, prefix, length, 5)
        q.mul_(0.5)
        slots.copy_(slots.flip(0))
        prefix.fill_(127)
        length.copy_(prefix + rows)
        graph.replay()
        fresh = minimax_index_scores_sm70(q, keys, slots, prefix, length, 5)
        torch.testing.assert_close(replayed, fresh, rtol=0, atol=0)

    def test_sparse_attention_tail_split_and_graph(self):
        from sglang.kernels.ops.attention.minimax_sparse.sm70 import (
            minimax_sparse_attention_sm70,
        )

        torch.manual_seed(531)
        for rows, heads in [(1, 8), (8, 8), (17, 16), (65, 8)]:
            with self.subTest(rows=rows, heads=heads):
                capacity, context = 2688, 2561
                q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.float16)
                k = torch.randn(capacity, 1, 128, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                slots = torch.randperm(capacity, device="cuda", dtype=torch.int32)[
                    :context
                ].contiguous()
                prefix = torch.tensor([2300], device="cuda", dtype=torch.int32)
                length = prefix + rows
                blocks = (
                    torch.arange(16, device="cuda", dtype=torch.int32)[None, None]
                    .expand(1, rows, 16)
                    .contiguous()
                )
                blocks[:, :, 15] = 18
                # Include padded support and an entirely empty query.
                blocks[:, 0, 14:] = -1
                if rows > 1:
                    blocks[:, 1] = -1
                output = minimax_sparse_attention_sm70(
                    q, k, v, slots, prefix, length, blocks
                )
                expected = torch.zeros_like(output)
                for row in range(rows):
                    pos = (
                        blocks[0, row, :, None] * 128
                        + torch.arange(128, device="cuda")[None]
                    ).reshape(-1)
                    mask = (
                        (blocks[0, row, :, None] >= 0).expand(-1, 128).reshape(-1)
                        & (pos <= prefix[0] + row)
                        & (pos < length[0])
                    )
                    pos = pos[mask].long()
                    if pos.numel():
                        physical = slots[pos].long()
                        logits = q[row].double() @ k[physical, 0].double().T * 128**-0.5
                        expected[row] = (
                            logits.softmax(-1) @ v[physical, 0].double()
                        ).half()
                torch.testing.assert_close(output, expected, rtol=0.001, atol=0.0002)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = minimax_sparse_attention_sm70(
                        q, k, v, slots, prefix, length, blocks
                    )
                q.mul_(0.75)
                prefix.fill_(129)
                length.copy_(prefix + rows)
                blocks.fill_(-1)
                blocks[:, :, 0] = 0
                graph.replay()
                fresh = minimax_sparse_attention_sm70(
                    q, k, v, slots, prefix, length, blocks
                )
                torch.testing.assert_close(replayed, fresh, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
