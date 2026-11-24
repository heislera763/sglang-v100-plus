"""Misaligned NVFP4 shards must preserve both gated halves and scale bytes."""

import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.checkpoint import (
        decode_csf32_scales,
        expand_csf32_weights,
        expand_mxfp8_fixed_weights,
        load_glm_csf_weights,
        normalize_block_fp8_weights,
    )
    from sglang_v100_plus.quantization import (
        pad_nvfp4_expert_width,
        prepare_nvfp4_moe,
        sm70_nvfp4_marlin_process_scales,
    )


def checkpoint(width):
    layer = torch.nn.Module()
    for prefix, n, k in (("w13", 2 * width, 2560), ("w2", 2560, width)):
        weight = torch.randint(0, 256, (2, n, k // 2), dtype=torch.uint8)
        # Include arbitrary encoded bytes: padding must not do FP8 arithmetic.
        scales = torch.randint(0, 256, (2, n, k // 16), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        layer.register_parameter(
            prefix + "_weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        layer.register_parameter(
            prefix + "_weight_scale", torch.nn.Parameter(scales, requires_grad=False)
        )
    return layer


class TestV100NVFP4Padding(CustomTestCase):
    def test_marlin_releases_unused_backend_scale_storage(self):
        """Unused swizzled scales retained 17.72GiB across an eight-GPU GLM."""
        layer = checkpoint(160)
        layer.params_dtype = torch.float16
        layer.quant_config = SimpleNamespace(group_size=16)
        dead = []
        expected = {}
        for prefix in ("w13", "w2"):
            scales = getattr(layer, prefix + "_weight_scale")
            # All non-negative finite E4M3 encodings, including zero/subnormals.
            scales.data = (
                (torch.arange(scales.numel()) % 127)
                .to(torch.uint8)
                .reshape(scales.shape)
                .view(torch.float8_e4m3fn)
            )
            expected[prefix] = sm70_nvfp4_marlin_process_scales(
                scales.transpose(1, 2).contiguous(), torch.float16
            )[0].view(torch.uint8)
            layer.register_parameter(
                prefix + "_weight_scale_2",
                torch.nn.Parameter(torch.ones(2), requires_grad=False),
            )
            swizzled = torch.nn.Parameter(scales.clone(), requires_grad=False)
            setattr(layer, prefix + "_blockscale_swizzled", swizzled)
            dead.append(weakref.ref(swizzled))
        del swizzled
        with (
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=80),
            ),
            patch(
                "sglang_v100_plus.quantization._dense_repack",
                lambda weight: weight.view(torch.int32).clone(),
            ),
        ):
            prepare_nvfp4_moe(None, layer, group_size=16)
        for prefix, old in zip(("w13", "w2"), dead):
            self.assertIsNone(old(), "Unused backend Parameter must be released")
            self.assertTrue(
                torch.equal(
                    getattr(layer, prefix + "_weight_scale").view(torch.uint8),
                    expected[prefix],
                )
            )
            self.assertIs(
                getattr(layer, prefix + "_blockscale_swizzled"),
                getattr(layer, prefix + "_weight_scale"),
            )

    def test_explicit_scale_group_is_validated_before_repacking(self):
        """Upstream callers can supply the per-layer group outside ModelOpt config."""
        layer = SimpleNamespace(params_dtype=torch.float16)
        for explicit, configured in ((32, 16), (None, 32)):
            layer.quant_config = SimpleNamespace(group_size=configured)
            with self.subTest(explicit=explicit, configured=configured):
                with self.assertRaisesRegex(ValueError, "requires group_size=16"):
                    prepare_nvfp4_moe(None, layer, group_size=explicit)

    def test_each_half_and_fc2_scale_group_preserves_checkpoint_bytes(self):
        layer = checkpoint(80)
        originals = {
            name: value.detach().clone().view(torch.uint8)
            for name, value in layer.named_parameters()
        }
        pad_nvfp4_expert_width(layer)
        self.assertEqual(tuple(layer.w13_weight.shape), (2, 192, 1280))
        self.assertEqual(tuple(layer.w2_weight.shape), (2, 2560, 48))
        for suffix in ("_weight", "_weight_scale"):
            first = getattr(layer, "w13" + suffix).view(torch.uint8)
            old = originals["w13" + suffix]
            self.assertTrue(torch.equal(first[:, :80], old[:, :80]))
            self.assertTrue(torch.equal(first[:, 96:176], old[:, 80:]))
            self.assertEqual(first[:, 80:96].count_nonzero().item(), 0)
            self.assertEqual(first[:, 176:].count_nonzero().item(), 0)
            second = getattr(layer, "w2" + suffix).view(torch.uint8)
            old = originals["w2" + suffix]
            self.assertTrue(torch.equal(second[..., : old.shape[-1]], old))
            self.assertEqual(second[..., old.shape[-1] :].count_nonzero().item(), 0)

    def test_aligned_shards_keep_their_original_storage(self):
        layer = checkpoint(160)
        pointers = {name: value.data_ptr() for name, value in layer.named_parameters()}
        pad_nvfp4_expert_width(layer)
        self.assertEqual(
            pointers,
            {name: value.data_ptr() for name, value in layer.named_parameters()},
        )

    def test_padding_does_not_reinterpret_another_weight_format(self):
        layer = checkpoint(80)
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.to(torch.int32), requires_grad=False
        )
        pointers = {name: value.data_ptr() for name, value in layer.named_parameters()}
        pad_nvfp4_expert_width(layer)
        self.assertEqual(
            pointers,
            {name: value.data_ptr() for name, value in layer.named_parameters()},
        )


def csf32_fixture(rows=160, columns=65):
    """Scalar wire-format fixture independent of the vectorized decoder.

    Clipped tiles, odd columns and 34 tile groups exercise the page boundary.
    Each block replaces its first/last scale, including signed/NaN byte codes.
    """
    bases = torch.arange(rows).remainder(100).byte()
    expected = torch.empty(rows, columns, dtype=torch.uint8)
    fields, records, prefixes, page_bases = [], [], [], []
    group = 0
    for tr in range(0, rows, 128):
        for tc in range(0, columns, 4):
            if group % 32 == 0:
                page_bases.append(2 * len(records))
                page_start = len(records)
                prefixes.append(0)
            for pair in range(tc, min(tc + 4, columns), 2):
                for rb in range(tr, min(tr + 128, rows), 32):
                    extent = min(2, columns - pair)
                    for rr in range(32):
                        for cc in range(extent):
                            field = (3 * (rb + rr) + pair + cc) % 16
                            fields.append(field)
                            expected[rb + rr, pair + cc] = int(bases[rb + rr]) + field
                    for local, value in [(0, 255), (62 + extent - 1, 0)]:
                        records.append(local | value << 8)
                        expected[rb + local // 2, pair + local % 2] = value
                    prefixes.append(len(records) - page_start)
            group += 1
    codes = [lo | hi << 4 for lo, hi in zip(fields[::2], fields[1::2])]
    parts = dict(
        bases=bases,
        codes=torch.tensor(codes, dtype=torch.uint8),
        page_bases=torch.tensor(page_bases, dtype=torch.uint32),
        offsets=torch.tensor(prefixes, dtype=torch.uint16),
        exceptions=torch.tensor(records, dtype=torch.uint16),
    )
    return parts, expected


class TestV100CSF32Checkpoint(CustomTestCase):
    def test_mxfp8_indexer_weighting_retains_fp32_operand_precision(self):
        prefix = "model.language_model.layers.45.self_attn.indexer.weights_proj"
        weight = torch.full((2, 64), 0.5).to(torch.float8_e4m3fn)
        scales = torch.tensor([[100, 120], [110, 125]], dtype=torch.uint8)
        stream = [(prefix + ".weight", weight), (prefix + ".weight_scale", scales)]
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=45),
            quant_config=SimpleNamespace(
                quantized_layers={
                    "model.decoder.self_attn.indexer.weights_proj": {
                        "quant_algo": "MXFP8"
                    }
                }
            ),
        )
        actual = load_glm_csf_weights(
            lambda model, stream, **kw: list(stream), model, stream, is_nextn=True
        )
        expected = weight.double() * torch.exp2(
            scales.double() - 127
        ).repeat_interleave(32, 1)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0][1].dtype, torch.float32)
        self.assertTrue(torch.equal(actual[0][1], expected.float()))
        self.assertFalse(torch.equal(actual[0][1], expected.half().float()))

    def test_modelopt_block_scales_remove_only_serialized_singletons(self):
        prefix = "model.language_model.layers.3.mlp.shared_experts.down_proj"
        scales = torch.arange(12).reshape(4, 1, 3, 1).float()
        for suffix in ("weight_scale", "weight_scale_inv"):
            actual = list(
                normalize_block_fp8_weights([(prefix + "." + suffix, scales)], [prefix])
            )[0][1]
            self.assertEqual(actual.data_ptr(), scales.data_ptr())
            self.assertTrue(torch.equal(actual, scales.reshape(4, 3)))
        with self.assertRaisesRegex(ValueError, "Invalid ModelOpt"):
            list(
                normalize_block_fp8_weights(
                    [(prefix + ".weight_scale", torch.zeros(4, 2, 3, 1))], [prefix]
                )
            )

    def test_glm_mtp_shares_quantized_head_with_its_method_and_scales(self):
        from sglang.srt.models.glm5_next_nextn import (
            Glm5NextForConditionalGenerationNextN,
        )

        draft = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )
        torch.nn.Module.__init__(draft)
        draft.lm_head = torch.nn.Linear(4, 8)
        target = torch.nn.Linear(4, 8)
        target.quant_method = object()
        target.register_buffer("weight_scale_inv", torch.ones(8, 1))
        draft.set_lm_head_from_target(target)
        self.assertIs(draft.lm_head, target)

    def test_glm_mtp_fused_projections_keep_source_quantization(self):
        """Missing draft mappings loaded raw MXFP8 into FP16 without its scales."""
        from sglang_v100_plus.fp8 import SM70MXFP8LinearMethod, mixed_quant_method

        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptMixedPrecisionConfig,
        )
        from sglang.srt.models.glm5_next_nextn import (
            Glm5NextForConditionalGenerationNextN,
        )

        cls = Glm5NextForConditionalGenerationNextN
        config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "packed_modules_mapping": cls.packed_modules_mapping,
                "quantized_layers": {
                    "model.language_model.layers.45." + name: {"quant_algo": "MXFP8"}
                    for name in (
                        "self_attn.q_a_proj",
                        "self_attn.kv_a_proj_with_mqa",
                        "mlp.shared_experts.gate_proj",
                        "mlp.shared_experts.up_proj",
                    )
                },
            }
        )
        config.apply_weight_name_mapper(
            cls.get_hf_to_sglang_mapper(SimpleNamespace(num_hidden_layers=45))
        )
        for prefix in (
            "model.decoder.self_attn.fused_qkv_a_proj_with_mqa",
            "model.decoder.mlp.shared_experts.gate_up_proj",
        ):
            with self.subTest(prefix=prefix):
                method = mixed_quant_method(
                    ModelOptMixedPrecisionConfig.get_quant_method,
                    config,
                    LinearBase(128, 128),
                    prefix,
                )
                self.assertIsInstance(method, SM70MXFP8LinearMethod)

    def test_mxfp8_absorbed_mla_preserves_fp16_operands(self):
        prefix = "model.language_model.layers.45.self_attn.kv_b_proj"
        weight = torch.linspace(-2, 2, 128).reshape(2, 64).to(torch.float8_e4m3fn)
        scales = torch.tensor([[120, 123], [125, 126]], dtype=torch.uint8)
        expected = (
            weight.double() * torch.exp2(scales.double() - 127).repeat_interleave(32, 1)
        ).half()
        for reverse in (False, True):
            weights = [(prefix + ".weight", weight), (prefix + ".weight_scale", scales)]
            if reverse:
                weights.reverse()
            actual = list(expand_mxfp8_fixed_weights(weights, {prefix: torch.float16}))
            self.assertEqual(len(actual), 1)
            self.assertEqual(actual[0][0], prefix + ".weight")
            self.assertTrue(torch.equal(actual[0][1], expected))
        with self.assertRaisesRegex(ValueError, "Incomplete MXFP8"):
            list(
                expand_mxfp8_fixed_weights(
                    [(prefix + ".weight", weight)], {prefix: torch.float16}
                )
            )
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=45),
            quant_config=SimpleNamespace(
                quantized_layers={
                    "model.decoder.self_attn.kv_b_proj": {"quant_algo": "MXFP8"}
                }
            ),
        )
        actual = load_glm_csf_weights(
            lambda model, stream, **kw: list(stream), model, weights, is_nextn=True
        )
        self.assertTrue(torch.equal(actual[0][1], expected))

    def test_mixed_mxfp8_selects_a16_without_blackwell_dispatch(self):
        """KDA layout queries must resolve MXFP8 without constructing an SM100 kernel."""
        from sglang_v100_plus.fp8 import SM70MXFP8LinearMethod, mixed_quant_method

        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptMixedPrecisionConfig,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.self_attn." + p: {"quant_algo": "MXFP8"}
                    for p in ("q_proj", "k_proj", "v_proj")
                },
                "packed_modules_mapping": {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            }
        )
        linear = LinearBase(128, 128)
        original = ModelOptMixedPrecisionConfig.get_quant_method
        self.assertIsInstance(
            mixed_quant_method(
                original, config, linear, "model.layers.0.self_attn.qkv_proj"
            ),
            SM70MXFP8LinearMethod,
        )
        self.assertIsInstance(
            mixed_quant_method(
                original, config, linear, "model.layers.0.self_attn.b_proj"
            ),
            UnquantizedLinearMethod,
        )

    def test_mtp_does_not_decode_unconsumed_target_scales(self):
        """Draft loading must discard target CSF before allocating decoded scales."""
        recipe = {"quant_algo": "NVFP4", "weight_scale_encoding": "csf_32x2_v1"}
        model = SimpleNamespace(
            quant_config=SimpleNamespace(
                quantized_layers={"model.layers.3.mlp.experts": recipe}
            ),
            config=SimpleNamespace(num_hidden_layers=45),
        )
        kept = torch.ones(1)
        weights = [
            (
                "model.language_model.layers.3.mlp.experts.0.down_proj.weight_scale.csf32_bases",
                torch.empty(0),
            ),
            ("model.language_model.layers.45.eh_proj.weight", kept),
        ]

        def consume(model, weights, is_nextn=False):
            return is_nextn, dict(weights)

        nextn, result = load_glm_csf_weights(consume, model, weights, is_nextn=True)
        self.assertTrue(nextn)
        self.assertEqual(list(result), [weights[1][0]])
        self.assertIs(result[weights[1][0]], kept)

    def test_exact_scale_bytes_across_clipped_tiles_and_pages(self):
        """A linear block order silently scrambles CSF32 scale rows/columns."""
        for rows, columns in [(32, 1), (96, 3), (128, 4), (160, 65)]:
            with self.subTest(rows=rows, columns=columns):
                parts, expected = csf32_fixture(rows, columns)
                actual = decode_csf32_scales(parts)
                self.assertEqual(actual.dtype, torch.float8_e4m3fn)
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected))

    def test_streaming_preserves_weights_and_rejects_incomplete_planes(self):
        """The ordinary loader must see each exact scale once before TP slicing."""
        parts, expected = csf32_fixture()
        prefix = "model.language_model.layers.3.mlp.experts.7.down_proj"
        scale_name = prefix + ".weight_scale"
        stream = [
            (scale_name + ".csf32_" + field, value) for field, value in parts.items()
        ]
        weight = torch.zeros(1, dtype=torch.uint8)
        stream.insert(2, (prefix + ".weight", weight))
        encoded = {"model.layers.3.mlp.experts"}
        result = dict(expand_csf32_weights(iter(stream), encoded))
        self.assertIs(result[prefix + ".weight"], weight)
        self.assertTrue(torch.equal(result[scale_name].view(torch.uint8), expected))
        for label, malformed, declared, capacity in [
            ("missing", stream[:-1], encoded, 256 << 20),
            ("duplicate", stream + [stream[0]], encoded, 256 << 20),
            ("undeclared", stream, set(), 256 << 20),
            ("capacity", stream, encoded, 1),
        ]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                list(
                    expand_csf32_weights(
                        malformed, declared, max_pending_bytes=capacity
                    )
                )

    def test_malformed_exception_metadata_fails_closed(self):
        """Bad page origins/positions cannot leave partially initialized scales."""
        for corrupt in ("bases", "page_bases", "offsets", "exceptions"):
            parts, _ = csf32_fixture()
            if corrupt == "bases":
                parts[corrupt][0] = 241
            elif corrupt == "page_bases":
                parts[corrupt][1] = 1
            elif corrupt == "offsets":
                parts[corrupt][0] = 1
            else:
                parts[corrupt][1] = parts[corrupt][0]
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                decode_csf32_scales(parts)


if __name__ == "__main__":
    unittest.main()
