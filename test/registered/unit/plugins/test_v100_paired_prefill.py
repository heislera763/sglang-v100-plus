"""Ownership and checkpoint boundaries for paired GLM prefill chunks."""

import gc
import sys
import unittest
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    from greenlet import getcurrent, greenlet
except ImportError:
    getcurrent = greenlet = None

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.iteration_ids import IterationIds
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus import paired_prefill as p


class TestV100PairedPrefill(CustomTestCase):
    def test_active_schedule_preserves_cpu_collectives(self):
        """CPU metadata sums must not enter CUDA-stream scheduling."""
        x = torch.arange(8, dtype=torch.float32)
        with (
            patch.object(
                p, "_slot", return_value=SimpleNamespace(get=lambda: object())
            ),
            patch.object(torch.cuda, "Event", side_effect=AssertionError("CUDA used")),
        ):
            result = p.all_reduce(
                lambda _group, value: value + 1, SimpleNamespace(world_size=8), x
            )
        torch.testing.assert_close(result, x + 1)

    def test_compiled_collective_keeps_eager_context_out_of_graph(self):
        """The global hook must preserve full-graph callers outside this profile."""
        group = SimpleNamespace(world_size=8)

        def original_model(_self, input_ids, *_args):
            return p.all_reduce(lambda _group, value: value + 1, group, input_ids)

        def operation(x):
            if p.is_active():
                raise AssertionError("compiled execution entered eager schedule")
            return p.model_forward(original_model, None, x, None, None)

        compiled = torch.compile(operation, backend="eager", fullgraph=True)
        x = torch.arange(8, dtype=torch.float32)
        torch.testing.assert_close(compiled(x), x + 1)

    def test_checkpoint_has_one_writer_at_child_boundary(self):
        """The +1 snapshot marker must move to final-state tracking at a cut."""
        for count, prefix, target in (
            (4096, 0, 4096),
            (3802, 4096, 6145),
            (3802, 4096, 7169),
        ):
            with self.subTest(count=count, prefix=prefix, target=target):
                batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    req_pool_indices=torch.tensor([0]),
                    seq_lens_sum=prefix + count,
                    batch_size=1,
                    input_ids=torch.arange(count),
                    positions=torch.arange(prefix, prefix + count),
                    out_cache_loc=torch.arange(count) + 1024,
                    seq_lens_cpu=torch.tensor([prefix + count]),
                    seq_lens=torch.tensor([prefix + count]),
                    extend_seq_lens=torch.tensor([count]),
                    extend_prefix_lens=torch.tensor([prefix]),
                    extend_seq_lens_cpu=[count],
                    extend_prefix_lens_cpu=[prefix],
                    extend_start_loc=torch.tensor([0]),
                    mrope_positions=None,
                    iteration_ids=None,
                    mamba_track_seqlens_cpu=[target],
                    mamba_track_seqlens=torch.tensor([target]),
                    mamba_prefill_track_mask_cpu=[True],
                    mamba_track_mask=torch.tensor([True]),
                )
                reader = RunnerIds(
                    req_to_token=torch.zeros(1, prefix + count, dtype=torch.int64),
                    token_to_kv_pool_allocator=None,
                    token_to_kv_pool=SimpleNamespace(
                        id_spaces=lambda: {IdSpaceKind.FULL: IdSpace(key="full")}
                    ),
                    page_size=256,
                    device="cpu",
                )
                batch.iteration_ids = IterationIds(
                    source=reader,
                    req_pool_indices=batch.req_pool_indices,
                    seq_lens=batch.seq_lens,
                    seq_lens_cpu=batch.seq_lens_cpu,
                    write_virtual=batch.out_cache_loc,
                )
                with patch.object(p, "mamba_cache_chunk_size", return_value=256):
                    a, b = p._split_batch(batch)
                for child in (a, b):
                    self.assertIs(child.iteration_ids, batch.iteration_ids)
                    torch.testing.assert_close(
                        child.iteration_ids.write_ids(
                            reader, cols=child.iteration_cols
                        ),
                        child.out_cache_loc,
                    )
                torch.testing.assert_close(
                    torch.cat((a.input_ids, b.input_ids)), batch.input_ids
                )
                torch.testing.assert_close(
                    torch.cat((a.positions, b.positions)), batch.positions
                )
                self.assertEqual(a.seq_lens_cpu.item(), prefix + 2048)
                self.assertEqual(b.extend_prefix_lens_cpu, [prefix + 2048])
                owners = [x for x in (a, b) if x.mamba_prefill_track_mask_cpu[0]]
                self.assertEqual(len(owners), 1)
                if target == 6145:
                    self.assertIs(owners[0], a)
                    self.assertEqual(a.mamba_track_seqlens_cpu, [6144])
                if target == 7169:
                    self.assertIs(owners[0], b)
                    self.assertEqual(b.mamba_track_seqlens_cpu, [7169])
                self.assertEqual(batch.mamba_track_seqlens_cpu, [target])

    def test_metadata_is_not_borrowed_from_the_next_chunk(self):
        """Refreshing a backend's tensor views cannot change an earlier frame."""

        @dataclass(frozen=True)
        class Metadata:
            lengths: torch.Tensor
            nested: list

        backing = torch.arange(8)
        source = Metadata(backing[::2], [backing[1::2]])
        captured = p._owned(source)
        backing.fill_(-1)
        self.assertEqual(captured.lengths.tolist(), [0, 2, 4, 6])
        self.assertEqual(captured.nested[0].tolist(), [1, 3, 5, 7])

    def test_every_layer_state_is_discovered(self):
        """Temporary traversal-list IDs must not skip later layers."""
        from sglang.srt.layers.layer_boundary.residual.mhc import MHCState

        @dataclass
        class Readout:
            state: MHCState

        states = [MHCState(4, None, None, None) for _ in range(45)]
        model = SimpleNamespace(
            layers=[
                SimpleNamespace(attn_boundary=Readout(s), ffn_boundary=Readout(s))
                for s in states
            ]
        )
        self.assertEqual([id(s) for s in p._states_of(model)], [id(s) for s in states])

    @unittest.skipIf(greenlet is None, "requires the V100 plugin's greenlet dependency")
    def test_suspended_chunks_restore_flags_and_release_contexts(self):
        """Force the exact A/B suspension that would leak plain forward flags."""
        s = p._Scheduler.__new__(p._Scheduler)
        flags = p.get_forward()
        s.parent = getcurrent()
        s.states = [SimpleNamespace(h_res=None, h_post=None)]
        s.mhc = [[(None, None)], [(None, None)]]
        s.original_flags = {key: getattr(flags, key) for key in flags._GRAPH_VISIBLE}
        s.flags = [dict(s.original_flags), dict(s.original_flags)]
        backend = SimpleNamespace(forward_metadata="parent")
        s.original_metadata = [(backend, "forward_metadata", "parent")]
        s.metadata = [
            [(backend, "forward_metadata", "A")],
            [(backend, "forward_metadata", "B")],
        ]
        s.outputs = [None, None]
        s.comm = object()
        order = []

        def chunk(index):
            self.assertIs(p._slot().get(), s)
            for step in range(2):
                value = (index, step)
                flags.set("mlp_reduce_scatter", bool(index))
                s.states[0].h_res = value
                s.save()
                order.append(value)
                s.parent.switch()
                self.assertEqual(flags.mlp_reduce_scatter, bool(index))
                self.assertEqual(s.states[0].h_res, value)
                self.assertEqual(backend.forward_metadata, "A" if index == 0 else "B")
            s.outputs[index] = torch.tensor([[index]])

        s.tasks = [greenlet(lambda: chunk(0)), greenlet(lambda: chunk(1))]
        refs = [weakref.ref(task) for task in s.tasks]
        with patch.object(
            torch.cuda,
            "current_stream",
            return_value=SimpleNamespace(wait_stream=lambda _: None),
        ):
            result = s.run()
        self.assertEqual(result.tolist(), [[0], [1]])
        self.assertEqual(order, [(0, 0), (1, 0), (0, 1), (1, 1)])
        self.assertEqual(backend.forward_metadata, "parent")
        self.assertIsNone(p._slot().get())
        self.assertEqual(
            flags.mlp_reduce_scatter, s.original_flags["mlp_reduce_scatter"]
        )
        self.assertFalse(s.tasks)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))


if __name__ == "__main__":
    unittest.main()
