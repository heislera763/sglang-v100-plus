import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.layers.attention.triton_backend import (
    ForwardMetadata,
    TritonAttnBackend,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestReplicatedDraftDCP(CustomTestCase):
    def test_draft_cache_writes_keep_all_virtual_slots_under_target_dcp(self):
        """Target DCP leaked into drafts, dropping rows and dividing their IDs.

        Draft KV heads/cache remain TP-sharded and token-replicated. Construct
        the real backend and exercise its actual write selector; targets must
        keep their DCP owner mask while drafts retain every virtual write.
        """
        for is_draft in (False, True):
            config = SimpleNamespace(
                attention_arch=AttentionArch.MHA,
                get_max_num_attention_heads=lambda: 32,
                get_total_num_kv_heads=lambda: 8,
                is_draft_model=is_draft,
                v_head_dim=128,
                swa_v_head_dim=128,
                context_len=1024,
                head_dim=128,
                is_encoder_decoder=False,
            )
            config.get_num_kv_heads = lambda tp, dcp: ModelConfig.get_num_kv_heads(
                config, tp, dcp
            )
            writes = []
            pool = SimpleNamespace(
                start_layer=0,
                get_value_buffer=lambda _: torch.empty(1, 1, 128),
                set_kv_buffer=lambda *args, **kwargs: writes.append((args, kwargs)),
            )
            pool.layer_router = KVLayerRouter.from_layer_maps(
                pool, [(IdSpaceKind.FULL, pool, {0: 0})]
            )
            pool.id_spaces = lambda: {IdSpaceKind.FULL: IdSpace(key="full")}
            runner = SimpleNamespace(
                req_to_token_pool=SimpleNamespace(
                    size=1, req_to_token=torch.zeros(1, 1024, dtype=torch.int64)
                ),
                sliding_window_size=2048,
                token_to_kv_pool=pool,
                token_to_kv_pool_allocator=None,
                page_size=256,
                kv_index_translator=None,
                decode_num_tokens_per_req=lambda: 8,
                model_config=config,
                is_draft_worker=is_draft,
                device=torch.device("cpu"),
                gpu_id=0,
            )
            with (
                self.subTest(is_draft=is_draft),
                get_context().override_server_args(
                    model_path="dummy", tp_size=8, dcp_size=4
                ),
                get_parallel().override(attn_dcp_rank=2),
                patch(
                    "sglang.srt.layers.attention.triton_backend.mambaish_config",
                    return_value=None,
                ),
                patch(
                    "sglang.srt.layers.attention.triton_backend.get_device_core_count",
                    return_value=80,
                ),
                patch(
                    "sglang.srt.layers.attention.triton_backend.is_arch_support_pdl",
                    return_value=False,
                    create=True,
                ),
            ):
                runner.runner_ids = RunnerIds(
                    req_to_token=runner.req_to_token_pool.req_to_token,
                    token_to_kv_pool_allocator=None,
                    token_to_kv_pool=pool,
                    page_size=256,
                    device="cpu",
                    is_draft_worker=is_draft,
                )
                backend = TritonAttnBackend(runner)
                expected_dcp = 1 if is_draft else 4
                self.assertEqual(backend.dcp_size, expected_dcp)
                self.assertEqual(backend.num_head, 4 * expected_dcp)
                self.assertEqual(backend.num_kv_head, expected_dcp)
                loc = torch.arange(64, 72)
                batch = SimpleNamespace(
                    out_cache_loc=loc,
                    out_cache_loc_is_physical=False,
                    positions=torch.arange(1000, 1008),
                )
                key = torch.empty(8, backend.num_kv_head, 128)
                backend._set_kv_buffer(
                    batch, SimpleNamespace(layer_id=0), KVWriteLoc(loc), key, key
                )
                args, kwargs = writes.pop()
                torch.testing.assert_close(args[1].loc, loc if is_draft else loc // 4)
                if is_draft:
                    self.assertNotIn("dcp_kv_mask", kwargs)
                else:
                    torch.testing.assert_close(
                        kwargs["dcp_kv_mask"],
                        batch.positions % 4 == backend.dcp_rank,
                    )


class _BonusTokenVerifyInput(SpecInput):
    def __init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        self.draft_token_num = 6
        self.num_tokens_per_req = 7


class _RunnerIds:
    is_translating = False

    def pack_read_stream(
        self,
        plan,
        *,
        req_pool_indices,
        seq_lens,
        indptr,
        out,
        kv_start_idx=None,
        kind=IdSpaceKind.FULL,
    ):
        out.zero_()
        return False


class _RecordingTritonBackend(TritonAttnBackend):
    def build_unified_kv_indices(
        self,
        _prefix_kv_indptr,
        _prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        _extend_kv_indices,
        batch_size,
    ):
        self.recorded_extend_start_loc = extend_start_loc.clone()
        self.recorded_extend_seq_lens = extend_seq_lens.clone()
        return (
            torch.zeros(batch_size + 1, dtype=torch.int32),
            torch.zeros(batch_size * 7, dtype=torch.int64),
            torch.zeros(batch_size, dtype=torch.int32),
        )

    def extend_attention_fwd_unified(self, *_args, **_kwargs):
        self.recorded_window_start_pos = _kwargs["window_start_pos"].clone()
        return None


def _make_backend(batch_size, capture_width=7):
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.device = torch.device("cpu")
    backend.num_draft_tokens = 9
    backend.target_verify_num_tokens_per_req = capture_width
    backend.max_context_len = 128
    backend.qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    backend.kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    backend.window_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    backend.mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64)
    backend.cuda_graph_kv_indices = torch.zeros(256, dtype=torch.int64)
    backend.runner_ids = _RunnerIds()
    backend.sliding_window_size = None
    backend.use_sliding_window_kv_pool = False
    backend._verify_mask = None
    return backend


def _make_forward_batch(batch_size, spec_info):
    seq_lens = torch.arange(32, 32 + batch_size, dtype=torch.int64)
    return SimpleNamespace(
        batch_size=batch_size,
        input_ids=torch.zeros(batch_size * 7, dtype=torch.int64),
        req_pool_indices=torch.arange(batch_size, dtype=torch.int64),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        seq_lens_sum=int(seq_lens.sum()),
        encoder_lens=None,
        spec_info=spec_info,
        forward_mode=ForwardMode.TARGET_VERIFY,
        out_cache_loc=torch.zeros(batch_size * 7, dtype=torch.int64),
        # A static pool's reads never touch the plan.
        iteration_ids=None,
    )


def _assert_verify_width(backend, batch_size):
    assert torch.equal(
        backend.forward_metadata.qo_indptr,
        torch.arange(0, (batch_size + 1) * 7, 7, dtype=torch.int32),
    )
    assert backend.forward_metadata.max_extend_len == 7


def test_eager_target_verify_uses_bonus_token_width():
    batch_size = 2
    backend = _make_backend(batch_size, capture_width=9)
    spec_info = _BonusTokenVerifyInput()

    backend.init_forward_metadata(_make_forward_batch(batch_size, spec_info))

    assert spec_info.draft_token_num == 6
    assert spec_info.num_tokens_per_req == 7
    _assert_verify_width(backend, batch_size)
    assert torch.equal(
        backend.forward_metadata.mask_indptr,
        torch.tensor([0, 273, 553], dtype=torch.int64),
    )


@pytest.mark.parametrize(
    ("batch_size", "raw_batch_size"),
    ((2, 2), (4, 3)),
)
@pytest.mark.parametrize("unset_width", (None, -1, 0))
def test_graph_capture_and_padded_replay_use_bonus_token_width(
    batch_size, raw_batch_size, unset_width
):
    backend = _make_backend(batch_size)
    capture_spec = None
    if unset_width is not None:
        capture_spec = SpecInput(SpecInputType.EAGLE_VERIFY)
        capture_spec.num_tokens_per_req = unset_width
    capture_batch = _make_forward_batch(batch_size, capture_spec)

    backend.init_forward_metadata_out_graph(
        capture_batch,
        in_capture=True,
    )
    _assert_verify_width(backend, batch_size)

    spec_info = _BonusTokenVerifyInput()
    replay_batch = _make_forward_batch(batch_size, spec_info)
    replay_batch.seq_lens[raw_batch_size:] = 1

    backend.init_forward_metadata_out_graph(
        replay_batch,
        in_capture=False,
    )

    _assert_verify_width(backend, batch_size)


@pytest.mark.parametrize("with_spec_info", (False, True))
def test_unified_target_verify_uses_resolved_metadata(with_spec_info):
    batch_size = 2
    backend = _RecordingTritonBackend.__new__(_RecordingTritonBackend)
    backend.device = torch.device("cpu")
    backend.dcp_size = 1
    backend.enable_deterministic = True
    backend.use_dense_fp8_chunked_prefill = False
    backend.allow_bidirectional_attention_in_extend = False
    backend.page_size = 1
    backend.use_sliding_window_kv_pool = False
    layer_kv = SimpleNamespace(
        key=lambda: torch.zeros((1, 1, 4)), value=lambda: torch.zeros((1, 1, 4))
    )
    backend.token_to_kv_pool = SimpleNamespace(
        layer_router=SimpleNamespace(read=lambda _layer_id: layer_kv)
    )
    backend.forward_metadata = ForwardMetadata(
        attn_logits=None,
        attn_lse=None,
        max_extend_len=7,
        num_kv_splits=None,
        kv_indptr=torch.zeros(batch_size + 1, dtype=torch.int32),
        kv_indices=torch.zeros(1, dtype=torch.int64),
        qo_indptr=torch.tensor([0, 7, 14], dtype=torch.int32),
        custom_mask=None,
        mask_indptr=None,
        window_kv_indptr=torch.tensor([0, 8, 20], dtype=torch.int32),
        window_kv_indices=torch.zeros(20, dtype=torch.int64),
        window_num_kv_splits=None,
        window_kv_offsets=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        qk_head_dim=4,
        v_head_dim=4,
        tp_q_head_num=1,
        k_scale=None,
        v_scale=None,
        logit_capping_method="tanh",
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=16,
        scaling=0.5,
        xai_temperature_len=None,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        forward_mode=ForwardMode.TARGET_VERIFY,
        mha_one_shot=False,
        out_cache_loc=torch.zeros(batch_size * 7, dtype=torch.int64),
        extend_seq_lens=None,
        extend_start_loc=None,
        extend_prefix_lens=None,
        seq_lens=torch.tensor([32, 64], dtype=torch.int64),
        spec_info=_BonusTokenVerifyInput() if with_spec_info else None,
    )
    q = torch.zeros((batch_size * 7, 4))

    output = backend.forward_extend(
        q,
        q,
        q,
        layer,
        forward_batch,
        save_kv_cache=False,
    )

    assert output.shape == q.shape
    assert torch.equal(
        backend.recorded_extend_seq_lens,
        torch.tensor([7, 7], dtype=torch.int32),
    )
    assert torch.equal(
        backend.recorded_extend_start_loc,
        torch.tensor([0, 7], dtype=torch.int32),
    )
    assert torch.equal(
        backend.recorded_window_start_pos,
        torch.tensor([24, 52], dtype=torch.int64),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
