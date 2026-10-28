# Frozen-model inference research journal

This journal belongs to `experimental`. It records hypotheses, source findings,
implementation decisions, controls, negative results and the next bounded step.
The owner's [original brief](BRIEF.txt) is the research charter. It is a set of
questions, not evidence that any proposed intervention already works.

## 2026-10-28 — Charter and starting point

Question: can the existing frozen GLM checkpoint support useful alternative
inference computations, and can we distinguish refinement from distortion?
Useful outcomes include better reasoning, better efficiency, a causal account of
an interesting failure, or evidence that ordinary execution is hard to improve.
Do not reduce this project to throughput tuning or aggregate benchmark scores.

Keep four claims separate:

1. Frozen-weight inference changes.
2. Calibration/configuration selection with frozen weights.
3. Separately fitted observational tools (including a Jacobian lens).
4. Learned gates, adapters or controllers around a frozen backbone.

A state change, a behavioral improvement and a causal explanation are different
findings. Vocabulary-labelled readouts are interpretive constructions, not a
literal thought transcript, correctness oracle or evidence of consciousness.

### Actual local architecture and execution

Verified against the local NVIDIA `GLM-5.3-Flash-FP4-Local` config and source:

| Property | Current value / consequence |
| --- | --- |
| Checkpoint | `nvidia/GLM-5.3-Flash-FP4-Local`, revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Decoder | 45 layers: 34 KDA, 11 sparse DSA |
| Pattern | Three KDA then one DSA; DSA indices 3, 7, …, 43; final KDA |
| Residual | Four mHC streams of width 4096; full intermediate width 16384 |
| Experts | First three FFNs dense, then 42 MoE layers; 288 routed experts, top eight, plus one shared expert |
| KDA | 64 heads × 128, convolution kernel four, lower gate bound −5 |
| Sparse attention | 2048 selected keys; indexer 32 heads × 128; shared KV rank 512 |
| MTP | One trained prediction layer reused for multiple draft steps |
| Maintained execution | TP8/DCP4/EP1, own MTP3, FP16 KV and recurrent state, vision; 524288 logical context / 131584 physical slots |
| Prefill | 4096-token admission with paired 2048-token internal chunks |
| Hardware | Eight V100 32 GB; two fully NV2-connected quads, PCIe Gen3 ×8; inter-quad NCCL uses host SHM |

Weights resident in HBM do not make repeated layers free: they may be read again,
and projections, expert routing and collectives execute again. Extra state can
also consume memory even when weights are shared. A capability benefit may pay
for itself through fewer reasoning tokens; that must be measured, not assumed.

The public frozen deployment remains stopped at `eae6773c`. Research must not
restore or change hosting. Strict V100 dispatch and FP16 KV/state remain in force;
no KV quantization. Private user-systemd tasks are allowed. Keep captures, raw
outputs and machine-specific tooling outside Git under `~/sglang-workspace`.
This journal and purposeful experimental code are explicitly allowed in this
branch by the owner.

### Code observations and state semantics

- `python/sglang/srt/models/glm5_next.py`: decoder forward has separate attention
  and FFN boundary prepare/finish operations; four-stream residuals can use lazy
  representations. Observe materialized boundaries, not an assumed 4096-vector.
- `moe_output_buffer_ctx(hidden_states_orig)` can reuse original residual storage.
  Captures must own their data; otherwise overwritten views can fake trajectories
  or convergence. Existing MTP/DFlash captures are not automatically full mHC
  captures.
- KDA state is keyed by physical layer and request, including convolution and
  recurrent state. Calling a physical layer twice does not define a correct
  repeated-depth history policy.
- DSA writes, selected IDs, positions, DCP ownership and radix reuse must have
  explicit semantics too. Do not infer that keeping token position unchanged is
  sufficient.
- The model already accepts `input_embeds`; this is a useful seam for soft-token
  experiments, not a complete implementation of sampling or stopping policy.
- The outer forward is `no_grad`; custom inference kernels are largely
  forward-only. Merely enabling gradients is insufficient for fitting a lens.

Two different loop models must not be conflated:

1. **Virtual-depth histories:** share weights while each repeated depth owns its
   own history/state, consistently across prefill and decode.
2. **Same-history trials:** restore the original state before each refinement
   trial, then perform a defined canonical state commit for the chosen result.

Restoring KV alone is insufficient for GLM. KDA, convolution, sparse metadata and
replay/prefix-cache ownership also matter. Neither policy should be silently
switched to make an experiment run. Research configurations must not share
prefix-cache entries when they compute different histories.

## 2026-10-28 — Literature/source review

Sources were opened and compared with the local implementation. Reported results
below are external evidence, not validation on our checkpoint.

| Lead | Finding | Consequence here |
| --- | --- | --- |
| [Inner Loop Inference](https://arxiv.org/html/2602.14759v1), [code](https://github.com/jonathanlys01/looped-transformer) | Repeated contiguous blocks with mixing/anchoring; naive repetition often degrades. Results on Gemma 2 / Llama 3 are modest and model-dependent. | Best direct starting hypothesis; no promised GLM gain. |
| [Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1) | Damped/anchored iterations; explicitly restores cached history to avoid phantom token appends. Reports several unsuccessful loop/integrator variants. | Cache policy is part of the algorithm. ODE analogy does not prove useful refinement. |
| [Jacobian lens / J-space](https://transformer-circuits.pub/2026/workspace/index.html), [implementation](https://github.com/anthropics/jacobian-lens) | Average downstream Jacobians support vocabulary-labelled readouts and causal interventions. Fitting requires backward computation; application is a different, cheaper workload. | Frozen weights are compatible with fitting a lens, but our runtime needs an activation-backward implementation first. |
| [Open-frontier replication](https://xiangchensong.github.io/blog/2026/jacobian-lens-global-workspace/), [code](https://github.com/xiangchensong/jacobian-lens-open-frontier) | GLM-5.2 and four-stream DeepSeek-V4-Flash; rectangular multi-stream mapping. Ablations also damage ordinary prediction, without a clean selective reasoning-workspace separation in their tested configurations. | DeepSeek geometry is more relevant than the GLM name alone. Released lens weights do not transfer to this checkpoint. |
| [J-CoT](https://arxiv.org/html/2607.21981v1) | Sparse semantic extraction, reconstruction at an earlier layer and repeated carriers. Zero variant still starts from a reasoning-adapted checkpoint; Train learns interface components. | A later bridge from observation to intervention. Reconstruction quality and out-of-distribution lens validity must be checked. |
| [Soft Thinking](https://arxiv.org/html/2505.15778v1), [repo](https://github.com/UCSB-AI/Soft-Thinking) | Probability-weighted token embeddings; paper describes SGLang 0.4.6.post1 implementation. Entropy/length stopping control matters; uncontrolled soft chains can repeat/collapse. | Practical second intervention. Existing input-embedding support helps, but porting scheduler/sampling/stopping remains real work. Repo code itself was not successfully inspected in this pass. |
| [SwiReasoning](https://arxiv.org/html/2510.05069v3) | Switches explicit and latent steps using entropy trends. | Separate mixture effects from stopping/allocation-controller effects. |
| [Soft-thinking critique](https://arxiv.org/abs/2508.03440) | Dominant token can govern the mixture's continuation. | A mixture is not independent parallel exploration of all token branches. |
| [Recirculation](https://arxiv.org/html/2608.17981v1) | Deeper-to-shallower feedback across tokens; fixed and learned variants. Prefill becomes serial. | Postpone for our long-context workload. Decode-only adaptation would be a new, unvalidated regime. |
| [Memory Inception](https://arxiv.org/html/2605.06225v2) | Text-derived attention memories, selected layers/heads, calibration. | Sparse selection must actually expose inserted memory; dense/GQA implementations are not drop-in KDA/DSA support. |
| [Architectural Sampling](https://arxiv.org/html/2610.01687v1) | Different repeat patterns improve candidate coverage in studied models. | Pass@k is not single-answer improvement; require a selector and compare total compute, not only candidate count. |
| [Beyond the Trace](https://arxiv.org/html/2608.17638v2) | Semantic/routing observations include a learned semantic frame and fitted routing proxy. | Expert IDs are not intrinsic semantic labels or a training-free correctness detector. Observation and controller validation are separate. |

Additional primary model references:
[official model](https://huggingface.co/zai-org/GLM-5.3-Flash),
[NVIDIA architecture/training path](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vision-language-models/thudm/GLM-5.3-Flash).
NVIDIA's published training stack is not a drop-in V100 gradient backend.

### Jacobian-lens feasibility details

Our source state has 16384 coordinates; a contracted final-state target has 4096.
A full FP32 4096×16384 map costs 256 MiB per boundary. Materializing all 154880
vocabulary directions in that source space as FP16 would cost about 4.73 GiB per
boundary; retain factored maps / selected directions instead where possible.
Sparse position captures are much smaller than all-token/all-layer logging.

The reference fitting implementation batches target-coordinate backward passes;
its default eight coordinates per batch implies 512 backward batches for width
4096 per calibration prompt. Alternative low-rank estimators are possible but
change the approximation. The replication's normalized per-prompt estimator and
straight-through activation-quantization derivatives also need explicit labels.

First prove an activation derivative against finite differences on a bounded
path. Hard expert routing and sparse selection introduce piecewise/discrete
choices. No claim that downloading the full BF16 checkpoint is necessary:
parameter gradients/updates are not needed, but an appropriate backward path is.
A lens calibrated on ordinary execution may misread repeated-depth activations.

### Sampling and correctness distinctions

Ordinary comparative performance tests use temperature zero / seed 531.
A soft embedding needs a non-degenerate internal probability distribution:
applying the usual T0 one-hot decision first would erase the intervention. A fixed
positive internal temperature with a deterministic weighted sum can coexist with
greedy final answers; this is a different, declared inference policy.

The implementation identity control should reproduce ordinary inference. An
intentional model intervention is evaluated for changed behavior, not rejected
merely because tokens differ. These are separate correctness questions.
MTP is initially off to isolate the modified target. Later MTP can be investigated
with a correct target verifier/state rollback; acceptance is not assumed stable.

## 2026-10-28 — Bounded first experiment design (not implemented yet)

1. **Identity control and owned observations.** Repetition disabled must match
   ordinary forward/cached continuation. Observe selected positions at completed
   four-stream boundaries, routing and output distributions. Instrumentation
   stays outside timed requests.
2. **Small intervention.** Primary hypothesis: one additional application of a
   short block window, with a declared damping/anchoring rule and explicit state
   semantics. A four-layer window can cover three KDA plus one DSA. An FFN-only
   repeat is a useful narrower control for expert transformation without advancing
   attention state; it is not a reproduction of the full-layer papers.
3. **Small behavioral pilot.** Short contexts, externally checkable arithmetic,
   multi-hop/state tracking and short-code cases. Separate configuration selection
   from held-out cases. Record wrong→right, right→wrong, reasoning length and
   failures. A small pilot is case evidence, not a reliable benchmark leaderboard.
4. **Cost accounting.** Prefill TPS, decode TPS and total completion time; compare
   promising variants with ordinary inference given comparable time. No automatic
   reverse-order pairs, redundant baselines or wide tuning grid.
5. **Causal follow-up only where warranted.** Remove/substitute an interesting
   internal change against matched controls. Check broad prediction damage too.
   Confidence, norm stability and readable labels alone are not success.

Do not start with learned adapters, a full J-CoT port, all-layer J fitting or
long-context qualification. A clear negative/mechanistic result is a valid end
point. One small experiment should finish before expanding the intervention set.

## 2026-10-28 — Maintenance prerequisite and recoverable handoff

Owner first requested main/dev synchronization, latest upstream integration and a
short GLM check, then this experimental playground. Owner subsequently asked to
create/journal the branch immediately so maintenance cannot lose the research
thread. That changes ordering, not the requirement to validate the integration.

- Last validated main/dev/origin: `3a7f04fb`, retaining routed NVFP4 FP32 split-K.
  Compatible near8K/MTP3 baseline: 2837.18 prefill / 87.99 decode TPS. These are
  historical measurements, not a new run for this project.
- Fetched both remotes. Origin had no additional changes. Upstream moved from
  `0d6b283d7` to `59af99a87` (479 commits).
- A merge into `dev` is in progress in `/home/alexander/sglang-server`.
  Fifteen textual conflicts were resolved/staged; the merge is NOT yet committed,
  validated or pushed. Main has not been advanced to unvalidated code.
- `experimental` was created from validated `3a7f04fb` in the separate worktree
  `/home/alexander/sglang-workspace/frozen-model-experimental`. Its first commit is
  this journal and the preserved brief, not the unfinished maintenance merge.
- Maintenance evidence/tooling: `~/sglang-workspace/frozen-model-20261028/`.
  `upstream-commits.txt` pins the incoming list; `resolve_merge.py` records the
  initial conflict resolution; `cpu/` holds per-file test output.
- Import smoke for plugin, GLM model and DSA backend passes on CPU. Per-file
  regression tests were started; do not describe the overall suite as passing
  until its final results are inspected. No model/GPU test has run this pass.

**Owner's explicit integration policy:** do not preserve inferior fork behavior
merely because it existed. Evaluate upstream changes; adopt better upstream
implementations and remove superseded patches. Retain only hardware support or
verified behavior still missing upstream. In particular:

- Upstream now fixes Marlin logical-order bias handling after crop and stops
  permuting dense bias at load. The entire upstream file was selected; remove its
  obsolete core-patch inventory entry once validated.
- Preserve upstream deterministic draft proposal seeding while keeping the
  existing per-request greedy proposal correction. Upstream's shared helper
  supports both; the conflict was resolved to call it with seeds and positions.
- Preserve new segmented prompt rendering; GLM literal media text still needs its
  exact special-token protection path when literals are present.
- Upstream cache/state ownership changed substantially. The plugin's paired
  prefill still references `kv_loc_plan`/`kv_loc_cols`; adapt to the new iteration
  ID owner API. The PP commit graph still reads `mamba_pool`; review the new
  recurrent-state owner. Do not add broad compatibility fallbacks.
- Review resolved DSA DCP/kernel changes, lazy imports and all automatic merges
  for semantic compatibility. New upstream support includes hybrid MTP padding,
  KDA graph beta fixes and cache ownership/capacity fixes. Do not claim a speed
  benefit without measurement or enable unrelated new settings indiscriminately.

Next maintenance steps: finish interface/patch audit; run affected tests; launch
one private GLM smoke with current TP8/DCP4/MTP3/FP16 state and image/prefix checks;
record prefill/decode honestly; commit and align/push main/dev. Then merge that
validated result into `experimental` and implement the bounded first experiment.
Keep public hosting down and stop owned workers after tests.

## Journal entry template

For each experiment: question; code/checkpoint revision; configuration and state
semantics; calibration vs held-out inputs; identity/numerical checks; observations;
behavior and cost; competing explanations; disposition; exact next bounded step.
Link local raw evidence, but do not commit large traces/captures or credentials.
Mark hypotheses, external claims and locally measured facts distinctly. Record
negative results and incomplete work as plainly as successes.
