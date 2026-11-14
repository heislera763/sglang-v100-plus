# Swift FP16 memory experiment

Target `ukisai/Swift1.5-Qwen3.8-Flash-Next` revision `0bd4fe22431372cdad1979267d3ab45aa7e6150a`, loaded from its BF16 checkpoint as FP16. The objective is one session with native MTP3, vision and 262144 context. Target/draft operands, KV, learned index and recurrent states retain their existing FP16 boundaries. Complete projection K reductions and existing attention selection are preserved.

Work is isolated on `experimental-memory`, based on main/dev `084527ce1`. The normal server remains stopped. All source changes belong to this fork, with captures and launchers under `~/.local/state/server-work/sglang-swift-fp16/memory-experimental`. Models, host configuration, networking and GPU policies were preserved.

## Implemented layout

| Component | Placement and checks |
| --- | --- |
| HC down/up matrices | One full-weight owner per TP group, balanced across layers. Broadcast FP16 outputs and the native FP32 verification gates. Prefill retains quad-local token partitions and transfers one normalized shard at a time through existing broadcasts. |
| Router, QSA index projection and PLE projections | One full-weight owner per NVLink quad. Preserve their original GPU operations and output placement. |
| Main QSA KV | One physical FP16 cache per identical-KV-head consumer group, with CUDA VMM peer mappings. TP8 uses four consumers, PP2 uses two. |
| Native MTP KV | Stripe physical pages across consumers behind the original contiguous address space. GPU ownership masks compose with caller validity masks. Prefix commits, cache moves and changed-input graph replay were checked. |
| Compressed learned index | One FP16 physical store per layer and quad. The designated owner performs compression. Request-local pending rings retain their existing role. |
| Input vocabulary and RoPE | Pinned host tables with GPU lookups. RoPE retains its complete FP32 table. Vision's RoPE is included because upstream expands that table to the text context too. Expert arithmetic and attention execute on GPUs. |
| MTP embedding/head binding | Bind target-shared weights before KV budgeting to remove temporary duplicate residency from the budget. PP2 retains its required stage-local input table in host RAM. |
| Existing first-pass changes | Compact expert-location metadata, omit the unused PP1 vision encoder and reuse PyNCCL for eager gathers. |

Select the separately registered `v100_memory` plugin through the existing `SGLANG_PLUGINS` setting. Default plugin discovery leaves this memory profile inactive. The profile validates SM70, full-FP16 Qwen4-Exp shapes, eight GPUs, TP8/EP8 or PP2xTP4/EP4, DCP1, one request, eager prefill and the supported memory layout.

The new upstream-core surface is nine changed lines in the existing embedding operator. Runtime ownership, process groups, VMM lifetimes and model integration reside in one plugin module. The existing CPU memory tests and embedding tests contain the new regression cases. No new test files or checkpoint formats were added.

## Model confirmation

One coherent 7984-token prompt, a 32-output warmup, a fresh uncached 256-output request and prefix reuse. Throughput requests use temperature 0 and seed 531. A separate 64-output temperature-1, top-p .95, top-k 20 request checks sampled execution. Image checks use a 224x224 synthetic image. Profiling stays outside timed requests.

| Layout | Context allocated | Prefill tokens/s | Generation tokens/s | Draft acceptance |
| --- | --- | --- | --- | --- |
| TP8, final broadcasts, chunk256 | 262144 | 661.2 | 51.2 | 66.3% |
| PP2xTP4, final broadcasts, chunk256 | 131072 | 1159.8 | 36.5 | 45.0% |

Before the final prefill changes, TP8 at 262144 with chunk128 measured 367.7 prefill and 42.7 generation tokens/s, with 51.2% acceptance. Changing prefill partitioning/chunking changes rounded activations and generated trajectories. Most of the generation-throughput increase follows the changed acceptance. The execution-cycle proxy improves about 2%, so these numbers do not establish a 20% faster decoder.

The earlier PP2 chunk256 screen measured 1225.2 prefill and 42.6 generation tokens/s, with 63.6% acceptance. Final prefill is 5.3% lower in this screen. Its generation-cycle proxy improves about 6% while acceptance decreases. Across fresh PP2 processes the continuations diverge after output token111. Within each final process, the warmup token IDs match the measured prefix and the prefix-reuse output exactly. These screens include trajectory and acceptance variation.

## Memory accountability and limits

TP8's final cache/index physical allocations total 7.059 GiB including VMM page rounding, about 20.59 GiB below consumer-replicated storage at the same capacity. HC ownership removes 8.545 GiB of duplicated down/up weights across eight GPUs.

The final TP8 snapshot averages 29997.04 MiB target parameters, 646.26 MiB unique draft parameters, 903.50 MiB physical KV/index pages, 54 MiB verification state and 33.75 MiB live/prefix GDN state per GPU. Driver-free memory averages 145.75 MiB. The per-serial ledgers reconcile every physical byte through named allocator blocks, unused allocator reservation, physical VMM pages, driver reservation and free framebuffer. About 584.88 MiB/GPU of CUDA/NCCL allocations remain jointly accounted outside Torch and the cache VMM. Their individual internal owners are unexposed in this run.

The existing host N-gram table now reserves its actual 95.37 GiB instead of 128 GiB through exact-size large pinned allocations. Input/RoPE host tables add their own smaller pinned allocations.

PP2's 262K attempt failed during budget validation. Its first-stage GPUs had only 723–751 MiB free after weight loading, while the other stage had about 1990 MiB. Main-cache backing alone requires roughly 816 MiB on the heaviest first-stage ranks before GDN, graph and activation workspace. Moving one layer transfers about 1.2 GiB of expert parameters per GPU, so the retained 25/23 split leaves substantial last-stage slack. PP synchronization also conservatively pairs global minimum free memory with each stage's cost. That underestimates the exact ceiling, while 262K still exceeds the physical first-stage budget. 131K is the tested PP2 size.

262K pool allocation and near8K inference passed on TP8. Filling the complete 262K context and large-image workloads remain unverified. There is little runtime slack at this capacity.

## Launch profile

| Setting | Value and reason |
| --- | --- |
| GPU order | `0,3,1,2,5,6,4,7`, consecutive NVLink quads |
| Precision | FP16 model/KV/index/state, original FP32 RoPE |
| MTP | EAGLE native head, steps3, top-k1, draft tokens4, classical rejection sampling |
| Pool capacity | Context plus128 tokens, page64, one request, four Mamba slots |
| Runtime | `mem_fraction_static=.997`, eager prefill chunk256, decode graph batch1, overlap disabled, CPU image preprocessing and vision SDPA |
| Communication | Reused PyNCCL, custom all-reduce disabled, `NCCL_MAX_CTAS=1`, `NCCL_BUFFSIZE=1048576`, P2P level PHB and NVLS disabled |
| Allocator | `expandable_segments:True,pinned_max_round_threshold_mb:1024` |
| Other | Strict V100 dispatch, HC token partitioning, NUMA-interleaved host allocations and zero multimodal embedding-cache reservation |

An initial point-to-point HC prefill prototype allocated another 8 MiB NCCL peer buffer during execution and failed under the tight TP8 budget. Existing broadcasts replace those sends/receives, and their transport setup runs before KV sizing. The larger prefill chunk then passed actual model inference.

## Verification and handback

63 CPU tests and 138 subtests passed with CUDA hidden. Eight-rank TP8 and PP2 checks passed bit-exact HC mix/combine, token tails, projection ownership, changed-input CUDA graphs, VMM KV writes/moves, compressed-index writes, rank-distinct ownership canaries and changing prefix-commit lengths. Seven FP16 embedding cases passed, including changed-ID/weight replay, vocabulary padding, invalid IDs and int32/int64 inputs. Ruff and diff-whitespace checks passed.

Raw requests, per-GPU allocation snapshots, source captures, commands and physical ledgers remain in the private experiment directory. Final model evidence is in `tp8-262k-final` and `pp2-131k-final`. Earlier directories document rejected layouts and capacity failures. No permanent diagnostic instrumentation was added.

All owned model and test workers are stopped. Hosting remains down, and main/dev retain the base revision. Transient user units managed the model checks. The task branch contains the experimental source and this handoff. Reverting the experiment uses the ordinary `v100_plus` plugin and the base branch. Live launchers and deployed service definitions were preserved.
