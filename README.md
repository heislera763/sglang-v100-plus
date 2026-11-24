# sglang-v100-plus

This SGLang fork targets eight 32 GB V100 SXM2 GPUs arranged as two four-GPU NVLink islands. It rebuilds the useful SM70 work from the original V100 fork on mainline SGLang and supports Qwen3.8 Flash Next FP8, GLM-5.3 Flash FP4-Local, Hy4 Flash FP8 and ERNIE 5.5 Flash Preview. [Original V100 fork](https://github.com/haohervchb/sglang-V100) · [SGLang upstream](https://github.com/sgl-project/sglang)

The shared baseline for `main` and `dev` incorporates upstream through November 24 and builds sglang-kernel 0.4.11 for SM70. `codex/volta-raw-inference` isolates raw prefill and decode experiments. TP8 spans both islands. PP2×TP4 keeps each stage’s tensor-parallel reductions on its own quad and sends activations across the pipeline boundary. The GLM backend uses native CUDA sparse, vision and KDA kernels from the companion FlashAttention fork. [Upstream base](https://github.com/sgl-project/sglang/commit/f4284f29ba56fffe2d09c9e6c5ac5e218e6cc053) · [FlashAttention V100](https://github.com/heislera763/flash-attention-v100)

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
| [Baidu ERNIE 5.5 Flash Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview/tree/9e1f08fd86d1a0c232f6572e617ea9cc98775ec6) | `9e1f08fd86d1` | Block FP8 projections, group32 MXFP4 routed experts, bundled DSpark | 154.51 GiB |
| [Tencent Hy4 Flash FP8](https://huggingface.co/tencent/Hy4-flash-FP8/tree/003107fa248f5892064ab1c43af0a7003dd68c9b) | `003107fa248f` | FP8 routed experts and projections, BF16 Engram and vision weights, built-in five-layer DFly2 | 267.36 GiB |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13) | `236dfdf28582` | Official 128×128 block FP8, BF16 exclusions; FP8 PLE/MTP experts | 172.78 GiB |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348) | `da920bb0b9f4` | ModelOpt abs-max NVFP4; main shared experts remain BF16; FP8 KV metadata | 190.40 GiB |
| [nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/tree/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5) | `76348ddb2be2` | QAD mixed precision: CSF32-compressed NVFP4 expert scales, MXFP8 projections, block-FP8 shared experts and quantized vision/MTP | 165.33 GiB |
| [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e) | `2f33a8b9ae49` | NVFP4 routed experts, group32 MXFP8 projections, BF16 exclusions; bundled dense MXFP8 DSpark | 232.94 GiB +2.14 GiB draft |
| [RadixArk/Minimax-M3.1-Flash-Dflash2](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2/tree/923bf1d4b567d5ccb71b0488c6e61b5e4794e92a) | `923bf1d4b567` | BF16 DFlash2, Gemma norms, partial RoPE, 4096-token window | 3.83 GiB |
| [incoai/MiniMax-M3.1-Flash-DFlash2](https://huggingface.co/incoai/MiniMax-M3.1-Flash-DFlash2/tree/ac07a7c1ecaae26a7ace595756124dca974b432d) | `ac07a7c1ecaa` | BF16 DFlash2, RMSNorm, full RoPE, 2048-token window | 3.96 GiB |

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
| FP8 GEMMs | W8A16 with block128 or MXFP8 group32 scales. [Small-row GEMV](python/sglang/kernels/ops/gemm/sm70_fp8_gemv.py) reuses reconstructed FP16 weights across1–8 rows and reduces K in FP32. Short-K projections fuse their finishing reduction; large group32 verification projections decode four packed bytes together in native CUDA and retain scales across each group. Five-to-eight-row batches use FP32 FMA; finite FP16 products are exact in FP32, preserving the accumulation result with fewer instructions. [Prefill](python/sglang/kernels/ops/gemm/sm70_fp8_gemm.py) transiently unpacks one projection for native FP16 Tensor Cores/FP32 accumulation; no persistent FP16 weight bank. Intermediate row counts retain SM70 Marlin. Qwen block-FP8 experts retain their separate small-row kernels; EP4/EP8 preserves whole experts. |
| FP16 routed experts | [Volta MMA GEMM](python/sglang/kernels/ops/moe/sm70_fp16.py), retained for the original NVIDIA checkpoint's unquantized MTP experts. FP16 inputs/weights/output, FP32 accumulation and router multiply; upstream routing, activation and expert sum retained. Automatically selected by the SM70 plugin; unsupported epilogues fail under strict dispatch. |
| Hy4 Flash | FP16 latent KV, 30 sliding-window and 12 IndexPool layers. Selected-key Tensor Core GEMMs preserve index order. GPU lookups read sharded host Engram tables. DFly2 shares target KV and selected keys. |
| Qwen QSA/GDN/HC | Direct FP16-cache sparse prefill/decode/verification, masked Tensor Core prefill, SM70 GDN and gated-residual kernels; HC eager prefill partitions within each quad, including TP8. |
| GLM attention | [Canonical V100 CUDA backend](python/sglang/kernels/ops/attention/v100.py) from the [FA fork](https://github.com/heislera763/flash-attention-v100): selected-key FP16 sparse prefill/decode/MTP, FP32 reductions and FP16 probabilities/partials. The upstream Triton sparse-MLA modules are unmodified; this profile uses the FA implementation throughout. The model-native FP8 indexer uses [one Volta MMA scorer](python/sglang/kernels/ops/attention/dsa/sm70_indexer_score.py) for per-query decode and shared-history prefill. Prefill keeps ReLU/head weighting/reduction on-chip, converts the shared index bank once per chunk and reuses64-key tiles across query pairs with24KiB shared scratch; no global per-head score tensor. Whole-chunk query selection is shared within existing DCP4 quads; only final selected IDs are exchanged. Decode/verification gathers initialize only live counted prefixes; future scratch is masked by the scorer. |
| GLM KDA / mHC | Native CUDA full-K FP32 recurrence, FP16 state and per-step verification carry. KDA prefill prepares normalized Q/K and gates once; a Tensor Core chunk formulation is future work. Paired prefill uses exact fused full-row mHC preparation and avoids the local gathers; other profiles retain quad partitions. Its learned FP32 projection stays replicated to preserve reduction order. |
| MiniMax M3.1 | [Volta Tensor Core index/sparse attention](python/sglang/kernels/ops/attention/minimax_sparse/sm70.py), original selected128-token blocks, FP16 main/index KV and FP32 online/split-K reductions. FP16 Gemma QK norm/partial RoPE, FP32 biased top4 routing, original SwiGLU-OAI/clamps. NVFP4 encoding divisors are inverted once at load; MXFP8 groups stay32. Small routed GEMMs reuse the M16/K-major SM70 kernel and full-K FP32 reduction. Vision retains FP16 SDPA. [Quad collectives](v100_plus/sglang_v100_plus/quad_collectives.py) retain TP8 computation, use NVLink inside each quad and stripe bulk/verification sums over three fast PCIe bridges; tiny sums and gathers share the existing global NCCL communicator. |
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

Qwen FP8 TP4×PP2/EP4 with MTP3, vision, native 262K context and prefix reuse.
This profile passed near-8K text, image and repeated-request checks. Full 262K
prefix filling remains unqualified.

```bash
export SGLANG_PP_LAYER_PARTITION=24,24 SGLANG_ENABLE_PP_SPEC=1
export SGLANG_OPT_SM70_HC_PREFILL_SP=1 SGLANG_ENABLE_METADATA_GLUE_GRAPH=1
export SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH=1
unset NCCL_TUNER_PLUGIN
python -m sglang_v100_plus \
  --model-path "$HOME/models/sglang/Qwen-Qwen3.8-Flash-Next-FP8" --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 --port 9000 --api-key test-only --dtype float16 --trust-remote-code \
  --quantization fp8 --tensor-parallel-size 4 --pipeline-parallel-size 2 --expert-parallel-size 4 \
  --context-length 262144 --max-total-tokens 262272 --max-running-requests 1 \
  --chunked-prefill-size 4352 --attention-backend triton --mm-attention-backend sdpa \
  --image-processor-backend pil \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype auto --qsa-indexer-dtype float16 --mamba-ssm-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 4 \
  --disable-overlap-schedule --disable-prefill-cuda-graph --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --speculative-use-rejection-sampling \
  --moe-runner-backend marlin --sampling-backend pytorch --reasoning-parser auto --tool-call-parser auto \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

TP8 uses TP8/EP8, omits PP and unsets `SGLANG_PP_LAYER_PARTITION` and
`SGLANG_ENABLE_PP_SPEC`. Its measured profile uses the quad NCCL tuner above.
The ordinary `v100_plus` plugin retains compact expert-location maps and loads
vision on its owning pipeline stage. Full-FP16 weight/cache sharing stays experimental.
Lab sampling references: [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage),
[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes).

The optional Huihui-v2 adapter is a local PEFT conversion of
[nazeshinjite's published GGUF](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-Abliterated-LoRA/tree/dee8e0177e4e61efdaa3fb7128c40d6ab7a085d4).
Its FP32 factor values retain the published strength and alpha/r=1. The
conversion restores GGUF's tiled GDN input columns to HF grouped-head order,
with an exact forward-permutation round trip and FP64/TP4 update checks. It
targets 96 attention/shared-expert projections across 48 layers.
The converted directory contains ordinary `adapter_config.json` and
`adapter_model.safetensors` files plus local provenance. Add its path and a
distinct served model name to the Qwen launch.

```bash
--served-model-name qwen3.8-flash-next-huihui \
--json-model-override-args "{\"v100_static_lora\":\"$HOME/models/sglang/nazeshinjite-Qwen3.8-Flash-Next-Abliterated-LoRA\"}"
```

The static loader applies each declared projection on its owning PP stage and
TP shard. FP32 source factors retain their precision. BF16 factors use FP16.
Coefficients and adapter updates use FP32 before the final FP16 output. Packed
FP8 base execution and ordinary TP/EP reductions handle the adapted model.

The PP2×TP4/EP4, MTP3, vision, 262K-allocation screen passes both captured agent
prompts and sampled Pi file-read, arithmetic and code-edit/test workflows at
T1/P.95/K20/xhigh. Image recognition passes. A political-history prompt still
receives an evasive answer. Broader behavior and full-window execution remain
unqualified. This pass measures coherence only.

The previous [nchapman release](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-Heretic-LoRA/tree/9cdda5b1c1f8ba46116a29bc9a22204b33c46c79)
produces incoherent agent responses at full strength, including with MTP
disabled and a merged-weight control. Its 0.5× derivative was a diagnostic
strength probe. Operator checks cover independent FP64 references, masked
routes, canaries and changed-input graph replay.

MiniMax M3.1, TP8/EP1 with bundled DSpark, images and FP16 main/index/draft KV:

```bash
export SGLANG_RAGGED_VERIFY_MODE=static SGLANG_VLM_CACHE_SIZE_MB=32
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m sglang_v100_plus \
  --model-path "$HOME/models/sglang/MiniMaxAI-MiniMax-M3.1-Flash" \
  --served-model-name minimax-m3.1-flash --trust-remote-code --dtype float16 \
  --host 127.0.0.1 --port 9003 --api-key test-only \
  --tensor-parallel-size 8 --expert-parallel-size 1 \
  --moe-runner-backend marlin --attention-backend triton --mm-attention-backend sdpa \
  --kv-cache-dtype auto --page-size 128 --sampling-backend pytorch \
  --context-length 10240 --max-total-tokens 10496 --max-running-requests 1 \
  --mem-fraction-static 0.99 --chunked-prefill-size 1024 \
  --disable-overlap-schedule --disable-prefill-cuda-graph \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 6 \
  --speculative-num-draft-tokens 7 \
  --speculative-draft-model-path "$HOME/models/sglang/MiniMaxAI-MiniMax-M3.1-Flash/dspark" \
  --speculative-draft-attention-backend triton \
  --mm-process-config '{"image":{"device":"cpu"}}' \
  --reasoning-parser auto --tool-call-parser auto --random-seed 531 \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95}'
```

MiniMax's Triton backend manages metadata; index scoring, sparse attention and
draft windows use SM70 FP16 Tensor Cores with FP32 reductions. Vision's D80 heads
use FP16 SDPA. The selected DSpark configuration uses six proposals plus the
anchor/target position; these are not six sequential MTP steps. Its group32 MXFP8 weights remain
packed. Bounded draft graphs use about 20MiB/rank and bypass the generic 1GiB free
memory heuristic. Actual allocation failures remain explicit. Fixed-frequency
RoPE tables are sized to the configured context without changing frequencies.
The current FP16-cache qualification is **10K**, not the checkpoint's native 1M;
60 full-KV layers and replicated TP8 KV heads make this model much tighter than GLM.
For the qualified Inco DFlash2 alternative, use its pinned draft path,
`--speculative-algorithm DFLASH`, and context4096/pool4352; keep the eight-position
block and FP16 caches. Its unquantized weights load as FP16. The large context
projection is column-sharded across TP ranks, saving 378MiB/rank without changing
the full-K reduction. Radix's equivalent saves 315MiB, but that draft remains
**unqualified** because acceptance collapses in the current FP16 integration.

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

**Single-request raw inference, PP2×TP4.** Both quads run FP16 KV with a 262,144-token configured window and the checkpoint's sampler. Drafting is disabled. The 8K rows average three uncached requests with up to 256 sampled output tokens. The 131K rows are one cold document request. ERNIE’s earlier prefill includes first-use kernel compilation at new history widths.

| Model and phase (tokens/s) | Before | After |
| --- | ---: | ---: |
| ERNIE, 8K prefill | 2,144 | 2,323 |
| ERNIE, 8K decode | 18.4 | 21.9 |
| ERNIE, 131K prefill | 929 | 1,726 |
| ERNIE, 131K decode | 10.0 | 21.1 |
| Hy4, 8K prefill | 3,198 | 4,945 |
| Hy4, 8K decode | 36.2 | 37.4 |
| Hy4, 131K prefill | 1,286 | 5,416 |
| Hy4, 131K decode | 27.4 | 36.3 |

ERNIE shares one FP32 softmax across its FP16 window and compressed-history banks. Hy4 reuses each unpacked pooled key across query rows and uses indexed GEMMs for the eight-head pipeline-stage geometry. Both use Volta FP16 Tensor Cores. Hy4 passes the same image check on both builds.

**ERNIE 5.5 Flash Preview on eight V100s.** TP8, one request, 131,072 context, FP16 KV, Baidu's seven-token DSpark and native reasoning/tool parsers. The checkpoint supplies temperature 1.0 and top-p 0.95. Volta uses FP16 activations, the MXFP4 Marlin expert backend and the PyTorch sampler. Set the static memory fraction to 0.86 and cap the token pool at 131,328 for this 32 GiB-per-GPU layout.

| Tokens/s | First working Volta port | Optimized Volta port |
| --- | ---: | ---: |
| Prefill | 573 | 1,499 |
| Sampled decode | 33.1 | 38.3 |

Means of two uncached 3,772-token document requests per build. DSpark averaged 3.56 and 3.55 tokens per verification cycle respectively. The optimized indexer decodes each E4M3 bank once and uses FP16 Tensor Cores with FP32 scores. Large prefills exchange FP16 head slices across ranks to compute cross-head mixing locally in FP32. A separate 8,131-token request measured 1,287 prefill and 42.3 decode tokens/s. Both documents returned the requested record, and the native tool parser returned a structured call.

**Hy4 Flash FP8 on eight V100s.** TP8/EP4, FP16 KV, 262,144 context, vision, batch one. Prefill is effective uncached input throughput on 8.4K code and 9.6K document prompts. Generation averages those prompts and a short thinking chat at 160 output tokens. Code is greedy. Document and chat use temperature 0.9 and top-p 1.

| Tokens/s | Before | Ordered SM70 attention and routing |
|---|---:|---:|
| Raw prefill | 1,919 | 2,468 |
| Raw decode | 39.1 | 42.6 |
| DFly2 prefill, three drafts | 1,886 | 2,480 |
| DFly2 decode, three drafts | 49.9 | 57.2 |

The raw path reproduced the control’s 160-token greedy completion and all 128 scored positions’ returned log probabilities. The DFly2 verifier uses four target rows and can produce a different greedy continuation. The comparison used the same prompts, draft depth, checkpoint and upstream integration.

MiniMax M3.1 screen (2026-10-31): TP8/EP1, bundled DSpark/8 verification positions,
10K context, FP16 main/index/draft KV and vision, chunk1024, one request, strict
dispatch. Three coherent near-8K prompts, 32-token warmup and 256 T0/seed531 outputs;
cache flushed per request, profiling excluded. These are preliminary local rates.

| Category | Prefill tokens/s | Generation tokens/s | Draft acceptance |
| --- | ---: | ---: | ---: |
| Coding | 1763 | 75.9 | 50.4% |
| Writing | 1811 | 53.5 | 30.7% |
| Document analysis | 1808 | 50.4 | 27.1% |
| Mean | 1794 | 59.9 | 36.0% |

Matching expert routing to the existing 64-row cached-input Marlin tile improves
prefill **17.6%** over [75b1fb1b0](https://github.com/heislera763/sglang-v100-plus/commit/75b1fb1b0)
(1525/59.6 tokens/s); generation is unchanged. All three messages and acceptance
values match. Generic 48-row alignment previously required two 32-row GEMM tiles.
The retained schedule beats plain 32-row alignment on captured gate/up and down banks,
with 1.78× aggregate operator speedup and bit-exact outputs.

Earlier transport and FP8 work raised the compatible [3b8321aa1 baseline](https://github.com/heislera763/sglang-v100-plus/commit/3b8321aa1)
from 824/45.4 to 1525/59.6 tokens/s with the same weights, caches and sampling.
NVLink distribution through three healthy PCIe endpoints avoids the
Gen1x8 uplink without excluding its GPU from compute. Four-CTA subgroup
communicators bound VRAM; eager gathers reuse the graph communicator. Reordered
FP16 sums change greedy sequences and acceptance, so the full generation gain
includes that effect. The subsequent FP8 FMA, packed-feature and communicator reuse changes add 4.3%
generation versus transport alone, with all three messages and acceptance values
identical. No persistent expanded weight bank is added.

Images, tool calls, T1/P.95 sampling, 8064-token prefix reuse and 10208 input+30 output
at the 10K boundary pass without retractions; SGLang reserves two admission slots.
FP64 kernel references, changing-input graph replay/canaries and sanitizer checks
cover packed projections. Both NVLink quads were complete NV2. A larger paired
prefill schedule exceeded VRAM at the unchanged context and was removed; direct
small-row Tensor Core and unpack-plus-cuBLAS alternatives lost to packed GEMV.
Coarser FP8 partial reductions helped isolated projections but slowed complete
speculative cycles and were removed.
The 2026-11-01 DSpark depth screen tested four, six, seven and eight verification
positions: **49.6 / 54.5 / 60.9 / 59.8 generation tokens/s**, respectively;
prefill was **1803 / 1794 / 1800 / 1803 tokens/s**. Seven narrowly leads the mean,
but trades slower coding for faster writing/document analysis, with changed
generated sequences. Seven is the selected launch setting; this short screen does
not establish a consistent seven-position advantage. Four/six shorten cycles but
lose too many tokens per cycle. This comparison includes one fresh eight-position
control because the Gen1 PCIe endpoint moved after reboot.
The same-day layout screen keeps TP8/EP1: EP4 and EP8 with DSpark8 averaged
**819 / 37.1** and **807 / 32.5 prefill/generation tokens/s**, versus **1803 / 59.8**
for EP1. EP4 splits experts across two-GPU pairs; EP8 uses whole experts.
Both currently miss the EP1-specific routed-kernel schedules and quad collectives;
this measures current implementations, not an optimized EP upper bound. Ordinary
TP4×PP2 failed during weight processing with 30/30, 31/29 and 32/28 layer splits,
before KV allocation; there is no PP2 throughput result. DSpark currently requires
PP1. MiniMax sparse attention has no implemented DCP KV ownership/distributed
attention reduction, so GLM's DCP settings do not transfer to this backend.
The initial generic Triton attention path exhausted VRAM during autotuning;
there is no completed stock-attention throughput baseline.

Historical pre-transport near3K/256-output DFlash screens (context4K, same three categories):
Inco averaged 831 prefill/43.7 generation tokens/s with 26.0% acceptance and passed
the same image/tool/sampling/reuse checks. Radix averaged 834/15.8 with 0.6%
acceptance; changing only its resolved draft causality did not repair it.
DSpark at 10K averaged 829/46.0 on these same short prompts.
These predate the transport update and use different prompt/capacity settings;
Radix's loadability is not a claim of working speculation.

Upstream integration screen (2026-10-28), same near-8K/256-output profile below:

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
links at Gen3x8. Same three near-8K prompts, 32-token warmup, 256 T0/seed531
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
checkpoint's single MTP head. Current near-8K/256-output/T0 results:

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
near-8K inputs and256 outputs at T0; the compatible baseline is reused:

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

November 15 Qwen FP8/MTP3, FP16 KV/index/state, T1/P.95/K20/xhigh,
64-output warmups and 512 sampled outputs per coherent near-8K category.
Values are arithmetic means across coding, writing and document analysis.

| Tokens/s | TP8, text 12K | PP2×TP4, text 12K | TP8, vision 262K + reuse | PP2×TP4, vision 262K + reuse |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 4840 | 7813 | 4503 | 7119 |
| Generation | 76.5 | 62.7 | 76.8 | 60.3 |

All measured text requests have an empty prefix cache. The 262K profiles also
passed a separate cached-prefix request and a 224px image check. Allocation
leaves about 10 GiB/GPU free at readiness on TP8 and 11–12 GiB on PP2, before
request workspace. These screens qualify allocation and short inference.
The former 4831/79.4 TP8 and 7588/72.5 PP2 figures used **FP8 E5M2 KV** according
to their archived settings/allocation logs. Their previous FP16-KV label was
incorrect. They have a different precision baseline from the current FP16 runs.
The experimental eager-gather communicator reuse was excluded after its PP2
T0 cycle time increased about 8%, while sampled cycle time stayed flat.
Earlier experiments remain in the
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
