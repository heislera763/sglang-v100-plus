# sglang-v100-plus

[SGLang](https://github.com/sgl-project/sglang) adapted for **eight V100 SXM2 32 GB GPUs
in two NVLink quads**. We rebuilt useful SM70 support from
[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) on mainline,
effectively rebasing it into an independent fork. Qwen3.8-Flash-Next FP8 and
GLM-5.3-Flash NVFP4 work is consolidated on `main`.

`main` contains validated changes; `dev` is the integration branch. Both include
the qualified unified GLM attention work. The [FA fork](https://github.com/heislera763/flash-attention-v100)
owns native CUDA sparse, vision and KDA attention; SGLang owns thin adapters,
metadata and cache/state lifetimes. Advance `main` after correctness and model
throughput checks. Qwen retains its existing backend during this work.
Clone remotes: `origin` is this fork; `upstream` is `sgl-project/sglang`.

Integrated upstream: [59af99a8](https://github.com/sgl-project/sglang/commit/59af99a873f39c3b281df0f9c609b3a0d3270726), integrated 2026-10-28.
This update brings 479 upstream commits, including cache/state ownership changes,
hybrid MTP fixes, seeded draft proposals and segmented prompt rendering. SM70
adapters honor the new routing/zeroing contract; upstream's corrected Marlin bias
handling replaces our old patch. Paired prefill uses the new iteration-ID owner.
Hardware adapters live in [v100_plus/sglang_v100_plus](v100_plus/sglang_v100_plus),
reusable operators in [python/sglang/kernels/ops](python/sglang/kernels/ops), and
build/Marlin changes in [v100_plus/aot](v100_plus/aot) and [v100_plus/patches](v100_plus/patches).
[Provenance](v100_plus/provenance.json) and [recorded core changes](v100_plus/core-patches.json)
make upstream updates auditable. Services, benchmark clients, traces and project notes
stay outside Git; the separate `experimental` branch keeps its frozen-model
research journal and implementation. This assumes familiarity with SGLang.

## Checkpoints

Pinned benchmark sources; sizes are weight-file totals, **not runtime VRAM**.

| Checkpoint | Revision | Composition | Files |
| --- | --- | --- | ---: |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13) | `236dfdf28582` | Official 128×128 block FP8, BF16 exclusions; FP8 PLE/MTP experts | 172.78 GiB |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348) | `da920bb0b9f4` | ModelOpt abs-max NVFP4; main shared experts remain BF16; FP8 KV metadata | 190.40 GiB |
| [nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/tree/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5) | `76348ddb2be2` | QAD mixed precision: CSF32-compressed NVFP4 expert scales, MXFP8 projections, block-FP8 shared experts and quantized vision/MTP | 165.33 GiB |

FP4-Local is the current GLM target. It uses QAD mixed precision and needs its
**own NVFP4 MTP**, not the FP8 draft exported from the original checkpoint.
The loader expands CSF32 scale bytes once at startup; expert weights stay packed.
MXFP8 projections use native group32 W8A16. Absorbed MTP MLA and indexer weighting
projections are reconstructed once into FP16/FP32. Format references:
[Local Inference Lab vLLM](https://github.com/local-inference-lab/vllm/tree/8244d1c44348ae8a5bbfc9f1b607021652283ed4),
[CSF32 specification](https://github.com/local-inference-lab/flashinfer/blob/35c0f234c4b82f4b3b30258132c8bf718e222309/docs/csf-scale-blocks.md).
The original NVIDIA checkpoint remains available; publisher alone does not establish
quantization quality. NVIDIA Qwen NVFP4 and full-checkpoint requantization are deferred.

| Current GLM session | Value |
| --- | --- |
| Layout | TP8 / DCP4 / EP1; each consecutive four-rank group is one NVLink quad |
| Architecture | 45layers:34KDA recurrent,11DSA sparse; pool4 learned index selection |
| Context / physical pool | 524288tokens /131584rows per rank, exposing526336logical slots |
| KV / state | FP16, including prefix snapshots; separate model-native FP8 learned-index keys |
| KV/index allocation | About1.56GiB/rank at524K, excluding weights, recurrent state and execution scratch |
| Speculation / vision | Own MTP3, four verification positions; native FP16 vision with CPU image preprocessing |
| Prefill | 4096token admission, paired internal2048token chunks; same recurrent persistence boundaries |
| Prefix reuse | Four recurrent slots,2048token checkpoint spacing; retain reasoning with `clear_thinking=false` |

A previous-source full1M runtime check completed1,047,546input+1,024output tokens,
then image/prefix reuse. Current hosting/development uses524K; that capacity check
is separate from comprehensive long-context accuracy evaluation.

## SM70 execution

Volta supports FP16 Tensor Cores. FP4/FP8/BF16 checkpoint storage is decoded or cast
in software; it does not imply native arithmetic in those formats.

| Operation | Implementation / contract |
| --- | --- |
| NVFP4 GEMMs | [marlin-v100](https://github.com/heislera763/marlin-v100) at `95b1e62e` plus local patches: W4A16, FP16 WMMA with FP32 accumulation; block/global scales remain distinct. TP4/TP8 routed prefill uses64×256×32 cached-row Marlin tiles. Small routed groups at both TP sizes use [Volta MMA](python/sglang/kernels/ops/gemm/sm70_nvfp4_routed.py) with M16 tiles and K-major decoded weights; FP16 dequantization/output and FP32 accumulation/scaling remain. Unused alternate-backend MoE scale buffers alias active scales, saving17.72GiB across GLM's8GPUs. |
| FP8 GEMMs | W8A16 with block128 or MXFP8 group32 scales. [Small-row GEMV](python/sglang/kernels/ops/gemm/sm70_fp8_gemv.py) reuses reconstructed FP16 weights across1–4 rows and reduces K in FP32. Short-K projections fuse their finishing reduction; large four-row group32 projections decode four packed bytes together in native CUDA and retain scales across each group. [Prefill](python/sglang/kernels/ops/gemm/sm70_fp8_gemm.py) transiently unpacks one projection for native FP16 Tensor Cores/FP32 accumulation; no persistent FP16 weight bank. Intermediate row counts retain SM70 Marlin. Qwen block-FP8 experts retain their separate small-row kernels; EP4/EP8 preserves whole experts. |
| FP16 routed experts | [Volta MMA GEMM](python/sglang/kernels/ops/moe/sm70_fp16.py), retained for the original NVIDIA checkpoint's unquantized MTP experts. FP16 inputs/weights/output, FP32 accumulation and router multiply; upstream routing, activation and expert sum retained. Automatically selected by the SM70 plugin; unsupported epilogues fail under strict dispatch. |
| Qwen QSA/GDN/HC | Direct FP16-cache sparse prefill/decode/verification, masked Tensor Core prefill, SM70 GDN and gated-residual kernels; HC eager prefill partitions within each quad, including TP8. |
| GLM attention | [Canonical V100 CUDA backend](python/sglang/kernels/ops/attention/v100.py) from the [FA fork](https://github.com/heislera763/flash-attention-v100): selected-key FP16 sparse prefill/decode/MTP, FP32 reductions and FP16 probabilities/partials. Empty tiles skip identity work. The model-native FP8 indexer uses [live-history MMA decode scoring](python/sglang/kernels/ops/attention/dsa/sm70_indexer_decode.py) and [fused shared-history prefill scoring](python/sglang/kernels/ops/attention/dsa/sm70_indexer_prefill.py). Prefill keeps ReLU/head weighting/reduction on-chip, converts the shared index bank once per chunk and reuses64-key tiles across query pairs with24KiB shared scratch; no global per-head score tensor. Whole-chunk query selection is shared within existing DCP4 quads; only final selected IDs are exchanged. Decode/verification gathers initialize only live counted prefixes; future scratch is masked by the scorer. |
| GLM KDA / mHC | Native CUDA full-K FP32 recurrence, FP16 state and per-step verification carry. KDA prefill prepares normalized Q/K and gates once; a Tensor Core chunk formulation is future work. Paired prefill uses exact fused full-row mHC preparation and avoids the local gathers; other profiles retain quad partitions. Its learned FP32 projection stays replicated to preserve reduction order. |
| Storage | Unquantized FP16 attention KV for both models; Qwen FP16 QSA index, GLM's separate model-native FP8 indexer. KV-cache quantization is excluded from the project recipes. FP16 recurrent state is retained. |
| Cold module loading | Return idle Torch allocator blocks to CUDA when first-use Triton loads have little headroom; leave live tensors unchanged. Skip capture/custom arenas; steady kernel replay has no callback. |
| Vision / DFlash2 | GLM uses native FP16 D64 varlen vision attention; Qwen retains SDPA vision. [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) uses FP16 draft linears/KV and FP32 convolution/residual/norm for range safety. Its TP8/block8 draft uses native SM70 local GQA with FP32 split-K reduction; replicated draft storage stays outside target DCP. Eight-row target projections reuse packed FP8 weights. The Triton draft-backend option supplies metadata/cache management; arithmetic dispatches to native FA. Own MTP3 remains faster in the current screen. |


Strict dispatch verifies supported routes; it does not make every operation
V100-specific. Routing, activation, reduction, NCCL and some preparation still use
general CUDA/Torch/Triton implementations. The canonical FA package owns native
sparse, KDA and vision kernels. There is no legacy GLM attention fallback.
FP8 KV metadata in ModelOpt does **not** quantize this profile's FP16 KV cache.
Unsupported conflicting attention/backend/dtype choices fail before loading.

## Build and launches

Python3.12, `uv`, CUDA12.9. Setup builds pinned SM70 kernels, Marlin and
[sglang-fa-sm70 0.2.7](https://github.com/heislera763/flash-attention-v100/tree/4688df196ec3d71cb5099b37b6f4e538aedf53fe).
Use a fresh shell. Set device order from actual topology, keeping each NVLink quad
consecutive; enumeration is machine-specific. Use the pinned checkpoint revisions above.

```bash
git clone https://github.com/heislera763/sglang-v100-plus.git
cd sglang-v100-plus
git remote add upstream https://github.com/sgl-project/sglang.git
bash v100_plus/setup.sh
uv pip install --python .venv/bin/python fastokens==0.3.4
export CUDA_HOME=/usr/local/cuda-12.9 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH" OMP_NUM_THREADS=4
export SGLANG_PLUGINS=v100_plus SGLANG_V100_PLUS=1 SGLANG_DEBUG_V100_STRICT_DISPATCH=1
export SGLANG_V100_MARLIN_DIR="$PWD/artifacts/marlin-v100/vllm" SGLANG_SM70_DENSE_GEMV=1
export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
export NCCL_P2P_LEVEL=PHB NCCL_NVLS_ENABLE=0
```

GLM FP4-Local,524K/MTP3/vision/prefix caching (`http://<host>:9000/v1`, key `test-only`):

```bash
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_OPT_FUSE_MHC_POST_PRE=0
export SGLANG_OPT_SM70_HC_PREFILL_SP=1 SGLANG_OPT_SM70_NVFP4_MOE_GEMV=0 SGLANG_ENABLE_METADATA_GLUE_GRAPH=0
mkdir -p .cache
cc -std=c11 -O2 -fPIC -shared -I v100_plus/nccl/include v100_plus/nccl/quad_tuner.c -o .cache/libnccl-v100-quad-tuner.so
export NCCL_TUNER_PLUGIN="$PWD/.cache/libnccl-v100-quad-tuner.so"
curl -fsSL https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/chat_template.jinja -o .cache/glm-chat.jinja
python -m sglang_v100_plus \
  --model-path "$HOME/models/sglang/nvidia-GLM-5.3-Flash-FP4-Local" \
  --chat-template .cache/glm-chat.jinja --served-model-name glm5.3-flash \
  --host 0.0.0.0 --port 9000 --api-key test-only --trust-remote-code \
  --dtype float16 --quantization modelopt_mixed --fp4-gemm-backend marlin --moe-runner-backend marlin \
  --tensor-parallel-size 8 --expert-parallel-size 1 --dcp-size 4 --dcp-comm-backend a2a --no-dcp-replicate-q-proj \
  --attention-backend v100_plus --kv-cache-dtype auto --mamba-ssm-dtype float16 --page-size 256 \
  --context-length 524288 --max-total-tokens 131584 --mem-fraction-static 0.95 --max-running-requests 1 \
  --chunked-prefill-size 4096 --disable-overlap-schedule --disable-prefill-cuda-graph \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 --sleep-on-idle \
  --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 4 --mamba-track-interval 2048 \
  --mm-process-config '{"image":{"device":"cpu"}}' --tokenizer-backend fastokens \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 --speculative-use-rejection-sampling \
  --speculative-draft-model-path "$HOME/models/sglang/nvidia-GLM-5.3-Flash-FP4-Local" \
  --speculative-draft-model-quantization modelopt_mixed --sampling-backend pytorch \
  --reasoning-parser auto --tool-call-parser auto --random-seed 531 \
  --default-chat-template-kwargs '{"clear_thinking":false}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95}'
```

For Hub loading use the checkpoint ID and pinned `--revision` from the table.
The official template pin above renders equivalently to the optimized serving
copy on text/tool fixtures. MTP3 means three draft steps plus a target/bonus
position, with one linear branch and rejection sampling. Deployment units,
launch wrappers, captures and benchmark clients are intentionally outside Git.

Qwen FP8 TP4×PP2/EP4, ordinary decoding, thinking enabled. This reference keeps
radix reuse disabled, as in its recorded uncached performance screens:

```bash
export SGLANG_PP_LAYER_PARTITION=24,24 SGLANG_OPT_SM70_HC_PREFILL_SP=1 SGLANG_ENABLE_METADATA_GLUE_GRAPH=1
unset NCCL_TUNER_PLUGIN
python -m sglang_v100_plus \
  --model-path "$HOME/models/sglang/Qwen-Qwen3.8-Flash-Next-FP8" --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 --port 9000 --api-key test-only --dtype float16 --trust-remote-code \
  --quantization fp8 --json-model-override-args '{"language_model_only":true}' \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --expert-parallel-size 4 \
  --context-length 262144 --max-total-tokens 262272 --max-running-requests 1 \
  --chunked-prefill-size 4352 --attention-backend triton --mm-attention-backend sdpa \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype auto --qsa-indexer-dtype float16 --mamba-ssm-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer --disable-radix-cache \
  --disable-overlap-schedule --disable-prefill-cuda-graph --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 \
  --moe-runner-backend marlin --sampling-backend pytorch --reasoning-parser auto --tool-call-parser auto \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

Qwen TP8 uses TP8/EP8; omit PP and unset `SGLANG_PP_LAYER_PARTITION`/metadata flag.
For Qwen MTP3 add the shared speculative flags shown for GLM, omitting the explicit
GLM draft path/quantization; PP2 additionally needs `SGLANG_ENABLE_PP_SPEC=1`.
Lab sampling references: [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage),
[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes).

## Settings and topology

| Choice | Reason / limitation |
| --- | --- |
| TP8 versus TP4×PP2 | TP8 uses both quads concurrently in single-request decode. PP2 makes most layer collectives quad-local but serializes the two stages. FP4-Local's current tuning targets TP8. |
| DCP4 | Stripes target latent KV inside each NVLink quad; native index and draft storage remain replicated. Queries/FP32 normalizers are exchanged; full KV history is not. DCP8 crosses quads and lost its earlier screen. |
| Quad selector sharing | Identical queries/index keys/page IDs were proved in all11target layers and MTP. Shard whole-chunk scoring/top-k, gather selected IDs once; ≥512rows, or ≥64rows with rows×pooled keys≥2²¹. TP8/DCP4/max1 only; cheap appends stay local. |
| Paired prefill | Overlaps unchanged global TP sums with another chunk's compute; preserves metadata ownership and FP16 recurrence boundaries. Cross-quad NCCL still uses host staging; NCCL duration includes rank waiting. |
| Prefix caching | Reuses KV plus matching recurrent snapshots.2048spacing can require tail replay on tool turns; reducing it arbitrarily changes FP16 persistence/rounding. Exact append-only continuation is separate work. |
| Memory fraction / slots | .95 leaves measured execution headroom; explicit context/pool/state capacities prevent silent shrinking. Four recurrent slots for serving; eight only in private pinned-prefix experiments. No state-memory ratio is needed. |
| Image processing | CPU decode/transforms avoid a second CUDA patch bank; vision encoding remains on GPU. |
| Small-allreduce tuner | Tree/LL for small TP8 reductions; larger transfers retain NCCL selection. About2% shorter verification cycles in its original screen. |
| Other switches | GLM metadata/sample graphs and alternate NVFP4 MoE GEMV remain off without a model-level benefit. Qwen metadata graphs retain their qualified PP2 use. Generic split-K/decode and attention/expert-topology trials that lost were removed. |

## Performance and validation

Upstream integration screen (2026-10-28), same near8K/256-output profile below:

| Tokens/s, category mean | Before | After |
| --- | ---: | ---: |
| Prefill | 2,837 | 2,832 |
| Generation | 88.0 | 92.2 |

Prefill is flat. Generation increased in this short screen as MTP acceptance rose
67.1%→71.6%; mean verification-cycle time is essentially flat. This is not an
isolated kernel-speed claim. Images, prefix reuse and
T1/P.95 sampling pass. SM70 adapters adopt upstream's inactive-route contract and
new sampling/checkpoint arguments; no new full-context qualification is claimed.

Routed GEMM screen (2026-10-27): FP4-Local/own MTP3/TP8/DCP4/524K,
FP16 KV/state and vision, with both NVLink quads fully connected and all PCIe
links at Gen3x8. Same three near8K prompts, 32-token warmup, 256 T0/seed531
outputs; the compatible recorded control is reused.

| Tokens/s, category mean | Before | After |
| --- | ---: | ---: |
| Prefill | 2,838 | 2,837 |
| Generation | 84.4 | 88.0 |

Narrow routed expert gate/up now splits K eight ways with FP32 partial sums,
improving generation **4.3%**. Verification cycles shorten **6.6%** despite
acceptance falling69.9%→67.1%; prefill is unchanged. Reconstruction and output
remain FP16, with router/global scaling after the FP32 sum. Reduction order
changes, so this is not bit-exact parity. Fixed FP64-reference bounds,
changing-route CUDA graphs, canaries and sanitizers pass; images, prefix reuse
and lab-sampled execution pass. No new full-context throughput claim.

Earlier long-context selector screen (2026-10-22), with the same model/cache profile:
Before uses replicated selection on the qualified upstream merge; after shares
whole-chunk selection within each quad. Short-case baselines reuse the compatible
recorded process; deep append pairs use one resident process and exact pinned prefix.

| Tokens/s | Before | After |
| --- | ---: | ---: |
| Prefill, coding | 2,703 | 2,733 |
| Generation, coding | 98.4 | 100.0 |
| Prefill, writing | 2,710 | 2,748 |
| Generation, writing | 81.6 | 79.9 |
| Prefill, document analysis | 2,710 | 2,749 |
| Generation, document analysis | 87.9 | 87.6 |
| Prefill, short-case mean | 2,708 | 2,743 |
| Generation, short-case mean | 89.3 | 89.2 |
| Prefill, 200K history | 1,575 | 2,164 |
| Generation, 200K history | 76.9 | 79.0 |
| Prefill, 400K history | 1,273 | 1,943 |
| Generation, 400K history | 86.9 | 84.6 |

Deep-prefill gains are **37.4%/52.7%**. The pre-sharing native trace put selector scoring
at44–45% of paired-prefill wall windows; scoring/top-k improved3.5–4× in the
complete-operator screen including exchange. Model gains fit the cost budget.
Exchanging after every32-query scorer group was rejected because communication
erased most benefit. Short-context prefill is effectively flat. Decode kernels
and verification-cycle execution are unchanged; acceptance at200K/400K changes
69.8→72.4% /89.5→86.1%, explaining the opposing small generation changes.
The252-commit upstream integration itself established throughput parity.

Follow-up retuning keeps32-query launch batches and paired prefill. Larger128-query
batches improved the complete selector4–6%, but only about1% in the warm model;
serializing the same paired children reduced prefill1942→1733tokens/s. Selector
scoring now occupies17.4% of native paired windows, so its remaining operator
gains have less model impact. The new history-aware cutoff helps short deep appends:

| Tokens/s, 195-token append at400K history | Before | After |
| --- | ---: | ---: |
| Prefill, mean of two matched comparisons | 531 | 561 |
| Generation, same requests | 75.5 | 75.6 |

This cutoff-only gain is5.6%; MTP3/decode kernels are unchanged. A first request
after idle was slower regardless of variant; an explicit warmup excluded that
effect from the final comparison. Its cause is unproven. No routine reverse pairs
or extra cold model starts were added.

Higher MTP depths are supported: the [upstream GLM preset](https://github.com/sgl-project/sglang/blob/0d6b283d7e6ac79edc10b2ca788f309e972f527b/docs/src/snippets/configs/zai-org/glm-5.3-flash.jsx)
uses five steps, one branch and six verification slots, repeatedly using the
checkpoint's single MTP head. Current near8K/256-output/T0 results:

| Tokens/s, category mean | Before | After |
| --- | ---: | ---: |
| Prefill, MTP4 | 2,740 | 2,741 |
| Generation, MTP4 | 72.2 | 77.7 |
| Prefill, MTP5 | 2,741 | 2,735 |
| Generation, MTP5 | 73.9 | 80.9 |

Widening existing native sparse split attention from four to eight query rows
improves MTP4/MTP5 generation7.6%/9.5%, with gains in all three categories and
effectively flat prefill. FP16 KV is preserved. Wider rows now use the existing FP16
normalized-partial/FP32 LSE reduction, changing rounding relative to unsplit
prefill. Fixed FP64/reference bounds, changing-input graphs, canaries and
sanitizers pass; images, prefix reuse and lab-sampled execution pass. MTP3 keeps
its one-through-four-row verification schedule and remains the default; its
compatible reference leads at2,732prefill/89.1generation tokens/s.
Indexer/routed dispatch for wider verification still needs separate cost attribution.

DFlash2 block8's FA0.2.6 screen added native windowed FA and packed FP8 target
projections through eight rows; it predates the wider target split schedule and
has not been rerun. Same FP4-Local target/TP8/DCP4/524K, FP16 draft/KV/state,
near8K inputs and256 outputs at T0; the compatible baseline is reused:

| Tokens/s, DFlash2 | Before | After |
| --- | ---: | ---: |
| Prefill, coding | 2,349 | 2,352 |
| Generation, coding | 34.0 | 84.6 |
| Prefill, writing | 2,359 | 2,360 |
| Generation, writing | 22.8 | 52.3 |
| Prefill, document analysis | 2,366 | 2,364 |
| Generation, document analysis | 21.8 | 56.1 |
| Prefill, mean | 2,358 | 2,359 |
| Generation, mean | 26.2 | 64.3 |

Generation improves2.46×; complete cycles fall141→55.6ms while output/cycle
changes3.72→3.59. Native profiling identified a generic four-CTA, FP32-FFMA
draft-attention kernel taking about15.1ms per layer. The replacement uses
132 Tensor Core CTAs with bounded device-counted KV gathering and preserves
each query’s resolved left window. Fixed numerical bounds, graph/canary and
sanitizer checks pass; images and repeated-prefix reuse pass. Floating reduction
order changes: two of three greedy texts differ, so this is not bit-exact model
parity. Prefill remains unchanged; DFlash is outside paired prefill. Own MTP3
still leads this short suite at **2,732 prefill /89.1 generation tokens/s**.


Prefill counts only newly processed tokens divided by native prefill time;
generation is `(outputs-1)/native decode time`, obtained through opt-in
`return_meta_info`. Compilation, profiling and cold prefix filling are excluded
from append timing. Comparative runs use T0/seed531; serving uses lab sampling.
Reusing compatible baselines avoids repeated model startups and automatic reverse pairs.

The three short cases are coherent coding/writing/document-analysis prompts from
[SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1),
about8K inputs,32-output warmup and256 measured outputs. Deep tests grow one real
prefix to204800/409600 tokens, then process8267new tokens and128outputs; private
baseline/candidate share identical weights, state, prefix and process.

Historical Qwen FP8, FP16-KV/MTP3 lab-sampled screens (T1/P.95/K20/xhigh,512outputs):

| Tokens/s | TP8/EP8 | TP4×PP2/EP4 |
| --- | ---: | ---: |
| Prefill | 4831 | 7588 |
| Generation | 79.4 | 72.5 |

These are separate from the current GLM T0 comparison and were not rerun here.
Earlier intermediate results/rejected experiments are preserved in the
[previous README](https://github.com/heislera763/sglang-v100-plus/blob/487bb909ace96ca067200d9b695b814b58af9152/README.md).

Qualification uses current Marlin and independent FP64 references without
loosening established bounds (`rtol=.001, atol=.0002` where applicable), changing
inputs/routes/counts under CUDA graphs, output canaries and targeted Compute
Sanitizer runs. New collective-only selector integration reuses unchanged kernels;
its added checks cover eight-rank ownership/padding/concurrent reductions,
all-layer input identity, actual deep-context appends, images and request reuse.
The October22 integration passed330focused CPU cases,169runtime-context cases and58GPU
cases. October28 adds routing/masked-expert graph and canary coverage,22Marlin GPU
cases and deterministic sampling, plus targeted CPU adapter/ownership tests.
Five changed-route cases pass Compute Sanitizer with zero errors. Kernel
namespaces/core-patch inventory and affected checks remain auditable.

Useful checks: [core patch census](v100_plus/check-core-diff.py),
[indexer kernels](test/registered/kernels/ops/attention/test_sm70_indexer.py),
[partition contracts](test/registered/unit/plugins/test_v100_prefill.py),
[canonical FA tests](https://github.com/heislera763/flash-attention-v100/blob/4688df196ec3d71cb5099b37b6f4e538aedf53fe/csrc/sparse_mla/tests/test_attention.py).
Current524K image/reuse passes do not imply a new full-window accuracy campaign.
Qwen YaRN/image/video requalification, broader long-video work and DFlash2
full-window qualification remain separate tasks.

[Apache-2.0](LICENSE). Credit: SGLang, original V100 fork, Marlin/vLLM and model authors;
NCCL declarations retain their [license](v100_plus/nccl/LICENSE.txt).
