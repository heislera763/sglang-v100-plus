# Swift memory optimization

Target Swift1.5 Qwen3.8 Flash Next with vision, native MTP and 262144 context on eight V100 32 GiB GPUs. Preserve FP16 target/draft weights, KV and recurrent state during this pass. The owner excludes quantization and other lossy changes, accepting the existing BF16-to-FP16 conversion. Host offload is allowed only if necessary and measured fast.

Base main/dev is 084527ce1. Work belongs on experimental-memory. Hosting stays stopped. Model checks must enable MTP. No host, network, hardware-policy or model-storage changes.

Start with launch configuration, unused PP-stage vision residency and expert-location metadata. Reconcile each reduction against the existing all-rank memory audit before expanding into HC or KV sharding. Preserve numerical operations and routing. Keep captures, launchers and benchmark tooling outside Git under the existing SGLang private state/workspace directories.

Acceptance is measured memory savings with targeted regression checks and an MTP-enabled model attempt at the requested context. Report prefill and decode whenever model throughput can be measured, and explicitly record capacity or correctness blockers. Avoid spending full model startups on changes whose measured budget cannot fit.

Initial audit and per-GPU budgets are in ~/.local/state/server-work/sglang-swift-fp16/memory-audit. Current-layout projection is 33.07 GiB/GPU on PP0 and 35.09 GiB/GPU on PP1 before extra graph/activation workspace.

## First-pass results

The 262K/MTP3 PP2xTP4 attempt loaded the target and failed constructing the draft expert down bank. Its 400 MiB allocation exceeded the remaining 205–217 MiB of driver-free memory. KV allocation and inference were never reached. Prefill and decode throughput remain unmeasured for this configuration. All owned workers are stopped.

| Change | Before | After | Evidence |
| --- | --- | --- | --- |
| Replica map per GPU | 96 MiB | 0.1875 MiB | Current target reports shape 48x512x1; GPU graph replay survives placement update |
| Vision parameters per PP1 GPU | 215.61 MiB | 0 MiB | All four actual target parameter inventories; CPU/meta stage and encoder-only checks |
| Collective-check resident memory per GPU | 941.38 MiB | 721.38 MiB | Eight ranks, eager and graph gathers, changed inputs, exact results |

These component savings sum to about 3.31 GiB across eight GPUs. This is a component budget, with the communicator saving measured independently. The full model has not completed initialization with MTP.

## Configuration review

| Setting | Disposition |
| --- | --- |
| Existing CPU image preprocessing | Keep. It already places preprocessing on CPU. Switching to PIL offers no identified worker-VRAM saving here. |
| Existing FP16 KV/index/state and one-request pool sizes | Keep. Radix reuse, four state slots plus sentinel, and batch-one decode graphs remain useful. |
| `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024` | Installed PyTorch 2.13 allocator and mapped-GPU-gather checks passed. Allocations above 1 GiB use exact requested capacity. Projected n-gram pinned RAM drops from 128 to 95.37 GiB. A full-checkpoint run with this setting remains pending. |
| `NCCL_MAX_CTAS=2` | Capacity experiment only. Saved another 184 MiB/GPU in the collective check, while large gathers slowed. Model throughput needs validation before choosing this setting. |
| `CUBLAS_WORKSPACE_CONFIG=:16:8` | Rejected. Saved 8 MiB per handle, slowed several GEMMs and changed their rounded outputs. |
| Generic KV-sharding switch | Requires PD-prefill and other attention backends. Current QSA serving/MTP needs implementation work. |

The first model attempt predates the exact-size pinned-allocation setting. Its process group peaked near 239 GiB of cgroup-accounted host memory and 11.9 GiB of swap. The unit exited and its workers were cleaned up. Future loads should use the exact-size pinned allocation and retain NUMA interleaving.

## Larger remaining work

The current layout still needs substantial memory recovery. Quad-local HC output sharding can keep each projection's full K reduction and existing FP16 boundaries. Down outputs would gather the low-rank vector, while up weights would shard hidden coordinates across all HC streams and gather the final hidden output. This avoids a new FP16 partial-product reduction. Its interaction with the existing token-sharded HC prefill needs explicit treatment.

Quad-local distributed QSA is the next KV candidate. Each rank would retain a stripe of FP16 KV, gather the quad's query heads, compute local FP32 partial outputs/LSE, and merge into the original query-head shards. Sharing queries and attention results avoids gathering entire historical KV prefixes. MTP cache movement, prefix reuse, causal masks and graph replay need coverage. Implementation remains pending.

Private commands, allocation inventories, raw results and logs are under `~/.local/state/server-work/sglang-swift-fp16/memory-experimental`. Main/dev remain at the base revision. The experimental branch carries the source work and this handoff.
