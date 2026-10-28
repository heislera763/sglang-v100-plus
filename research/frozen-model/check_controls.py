"""Small CPU ownership/control checks; run before the GPU research pilot."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ffn_repeat import PassThrough, RepeatFFN, blend, request_policy

from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.runtime_context import get_forward
from sglang.test.test_utils import CustomTestCase, published_topology


class TestControls(CustomTestCase):
    def test_identity_does_not_touch_state_even_if_the_extra_ffn_reuses_storage(self):
        """Extra FFN scratch must not alias the canonical residual or leak flags."""
        for mode, strength, value in (
            ("off", 0, 1),
            ("repeat", 0, 1),
            ("repeat", 0.25, 5),
        ):
            hidden = torch.ones(2, 16384, dtype=torch.float16)
            saved = hidden.clone()
            batch = SimpleNamespace(
                batch_size=1,
                sampling_info=SimpleNamespace(
                    custom_params=[
                        {
                            "frozen_ffn": {
                                "case": "cpu",
                                "mode": mode,
                                "strength": strength,
                            }
                        }
                    ]
                ),
                residual_stream=ResidualStream(hidden),
                forward_mode=SimpleNamespace(is_extend=lambda: True),
            )

            class MLP:
                reduce_results = False

                def __call__(self, x, batch, allocator):
                    assert get_forward().moe_output_buffer is None
                    return 2 * x

            layer = SimpleNamespace(
                layer_id=23,
                config=SimpleNamespace(mhc=True, hc_mult=4),
                mlp=MLP(),
                post_attention_layernorm=SimpleNamespace(
                    weight=torch.ones(4096), variance_epsilon=1e-5
                ),
                hc_ffn_pre=lambda x, *_: (x[:, :4096].clone(), None, None, True),
                hc_post=lambda x, residual, *_: residual.add_(x.repeat(1, 4)),
            )
            output = (hidden, object())
            hook = RepeatFFN({"observations": "/unused", "layer": 23})
            with (
                published_topology(tp_size=8, ep_size=1, pp_size=1),
                patch("torch.cuda.is_current_stream_capturing", return_value=False),
                patch(
                    "sglang.srt.layers.moe.utils.sum_post_experts_output",
                    side_effect=lambda x: 8 * x,
                ),
            ):
                marker = object()
                with get_forward().scoped(moe_output_buffer=marker):
                    result = hook(layer, (torch.arange(2), hidden, batch), output)
                    self.assertIs(get_forward().moe_output_buffer, marker)
            self.assertTrue(torch.equal(hidden, saved))
            self.assertIs(result[1], output[1])
            self.assertTrue(torch.equal(result[0], torch.full_like(hidden, value)))
            self.assertIs(batch.residual_stream.residual, result[0])
            if strength == 0:
                self.assertIs(result, output)

    def test_zero_mix_and_noop_sampler_are_exact(self):
        anchor = torch.tensor([1.0, -2.0], dtype=torch.float16)
        self.assertIs(blend(anchor, torch.full_like(anchor, float("nan")), 0), anchor)
        self.assertIs(PassThrough()(anchor), anchor)
        for bad in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                blend(anchor, anchor, bad)

    def test_request_policy_fails_closed_for_ambiguous_arms(self):
        batch = SimpleNamespace(
            batch_size=1, sampling_info=SimpleNamespace(custom_params=None)
        )
        self.assertIsNone(request_policy(batch))
        batch.sampling_info.custom_params = [{"frozen_ffn": {"mode": "typo"}}]
        with self.assertRaises(ValueError):
            request_policy(batch)
        batch.sampling_info.custom_params = [{}, {}]
        with self.assertRaises(ValueError):
            request_policy(batch)


if __name__ == "__main__":
    unittest.main()
