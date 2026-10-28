# Frozen-model inference research journal

This journal belongs to `experimental`. It records hypotheses, source findings,
implementation decisions, controls, negative results and the next bounded step.
The owner's [original brief](BRIEF.txt) is the research charter. It is a set of
questions, not evidence that any proposed intervention already works.

**Resume here:** `main` and `dev` are aligned and pushed at validated `cdf91e2bf`,
including upstream `59af99a8`; that merge is now in `experimental` too.
The bounded FFN-only pilot is implemented in [ffn_repeat.py](ffn_repeat.py), with
[ownership controls](check_controls.py). Its first actual-model pilot is complete:
no demonstrated capability gain, with the long-prompt causal control unresolved.
See [results and disposition](#2026-10-28--ffn-only-pilot-results). All owned
workers are stopped and public hosting remains down. The
earlier integration handoff below is historical; do not reapply its patches.

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
- At journal creation, import smoke for plugin, GLM model and DSA backend passed
  on CPU; per-file regression tests had started and no model/GPU test had run.
  The later checkpoint below supersedes that test status.

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

## 2026-10-28 — Checkpoint after the first upstream smoke

The owner asked to preserve the branch/journal before getting further into
maintenance. The initial brief/journal commit `a53d08b3c` was pushed to
`origin/experimental`. Research worktree:
`/home/alexander/sglang-workspace/frozen-model-experimental`.
Main/dev/origin remain at validated `3a7f04fbbcbe11c588a4ab16751fa673da131ef9`;
the open merge targets upstream `59af99a873f39c3b281df0f9c609b3a0d3270726`.
Do not describe main/dev as updated or the merge as validated.

### What the integration has established

- Paired prefill now uses upstream `iteration_ids` / `iteration_cols` instead of
  the removed KV location-plan API. Its regression uses real `ForwardBatch`,
  `RunnerIds` and `IterationIds` ownership. The public `mamba_pool` property still
  delegates to the new state owner; it does not presently need a compatibility
  shim in the PP commit graph.
- Upstream added `a_output_slices` to the FP16 MoE interface. The SM70 adapter
  accepts the ordinary single slice and explicitly rejects unsupported multiple
  slices under strict dispatch. No silent generic fallback was introduced.
- Adapted per-file CPU regressions passed for strict dispatch, paired prefill,
  Triton verify metadata, decode graph runner, draft proposal/seeding, chat
  rendering and NVFP4 padding. Logs are in the maintenance `cpu/` directory.
  These passes predate the newest Marlin edits; they do not validate those edits.
- Installed upstream's required `tml-renderers==0.1.0` into the existing venv
  using `uv`; the chat regression then passed (175 cases, 255 subtests).
- One private TP8/DCP4/MTP3 model process loaded the weights (~195–209 seconds,
  22.69 GiB/rank) but failed in warmup: the SM70 Marlin adapter did not accept
  upstream's new `expert_ids_are_sorted` keyword. The controller stopped it.
  **No new prefill or decode TPS was measured.** Image, reuse and sampled smoke
  checks were not reached. Both owned units are inactive and GPU workers are gone.

### Why this needs a real adapter change

Upstream now supplies sorted-routing certification, optional two-word counts
`[padded_tokens, inactive_blocks]`, and `zero_inactive_output`. Its down projection
relies on the GEMM clearing valid routes assigned to inactive experts, while
preserving routes omitted from the descriptor. It removed a full output clear.
Dropping the new arguments would therefore be incorrect.

The native SM70 Marlin implementation skips negative experts without clearing
their output. The fork's small routed MMA implementation previously always
cleared them and accepted only a one-word count. Both must honor the new contract.
This is integration work, not a frozen-model intervention.

Unvalidated edits currently in the **dev** worktree:

- `kernels/moe_marlin.py`: accepts the new arguments, allocates zeroed output when
  needed, passes the first count word to legacy native kernels, forwards the
  inactive-output policy and avoids the old routed GEMV path for the new producer.
  Revisit that dispatch exclusion before accepting a performance regression.
- `sm70_nvfp4_routed.py`: conditional inactive-route writes, including split-K
  reduction, and two-word count acceptance. Known unfinished call site:
  `padded_count.reshape(1)` must instead select its first word before reshaping.
- New lazy `moe.zero_inactive_routes` Triton prototype: clears only inactive valid
  routes before native GEMMs that skip them. Check descriptor allocation capacity
  and masks: its exact `sorted_ids.numel() == expert_ids.numel() * block_size`
  validation may be too strict for the upstream producer. Do not assume it is safe
  until tested. Empty routes, unused tails and omitted output rows need coverage.
- The strict-dispatch routed-kernel test stub must accept the newly passed keyword
  argument. The latest prototype is not formatted or tested yet.

Exact local recovery copy, including staged and unstaged binary patches, the new
untracked source, merge hashes and checksums:
`~/sglang-workspace/frozen-model-20261028/resume-checkpoint-20261028T203928Z/`.
The original worktree still contains all of it; **do not blindly reapply patches**.
Raw tooling/captures and this recovery copy stay outside Git.

Next maintenance step: finish and test that routing contract (including graph
refresh/canaries), then rerun the single private model smoke. Preserve its first
failure log. Update provenance/core-patch inventory, drop the superseded Marlin
bias entry, commit the validated merge and align/push main/dev. Merge that result
into `experimental` before implementing the research pilot. Do not restore hosting.

### First-experiment implementation notes to preserve

The smallest state-safe exploratory control is one additional middle-layer
FFN/MoE + mHC update with damping. It intentionally excludes repeated attention;
it can help separate expert transformation effects before a four-layer loop.
It does **not** establish the behavior of a full repeated-depth transformer.

- Existing `model_executor/hook_manager.py` accepts `--forward-hooks` factories.
  Hooks are registered after CUDA-graph capture, so an eager pilot must explicitly
  disable decode graphs and MTP. Otherwise the intervention could appear enabled
  while replay silently omits it. Verify actual module names and hook execution.
- A middle decoder layer (candidate: layer 23) returns `(hidden_states,
  topk_indices)`. Work at a completed FFN boundary with all four residual streams
  materialized; assert width 16384 and no pending residual contribution. Avoid the
  final layer, whose contracted output has different semantics.
- Own a copy before an extra FFN call: the ordinary expert path can reuse residual
  storage. Reuse the layer's `hc_ffn_pre`, normalization, MLP and `hc_post` rather
  than inventing a new FFN equation. Its MLP has deferred TP reduction; establish
  the exact shared/routed scaling and reduction contract before explicitly summing
  partials. Two reductions or none would invalidate the experiment.
- Blend the completed result with the original state in FP32, cast once to FP16,
  and update `residual_batch.set_written(...)` as well as the returned tensor.
  Preserving only the Python return while leaving lazy residual ownership stale
  is not a valid intervention.
- Candidate controls: ordinary execution; extra computation with zero-strength
  output mixing; one damped application (e.g. 0.25); optionally undamped repetition
  as a contrasting control. Zero strength must preserve cached continuation too.
  Pick a bounded pilot, not a broad layer/strength sweep.
- Existing request `custom_params` could select arms within one eager process,
  saving repeated multi-minute loads. Current sampling code only retains them
  when a custom logit processor is supplied; an explicitly enabled no-op processor
  is one possible carrier. This is an untested implementation option, not a new
  public launch requirement. Baseline must use the same hook/processor overhead.
- Isolate prefix-cache entries by arm (`cache_salt`) and verify request reuse.
  Keep observations outside timed requests. Use short, checkable prompts and
  report both prefill/decode cost plus behavioral changes; no 524K qualification
  or claim of general capability improvement from a handful of cases.

## 2026-10-28 — Upstream integration completed

Merged 479 upstream commits through `59af99a873f3`; validated fork `cdf91e2bf` is
on main/dev/origin and merged here. All maintenance code remains outside this
research directory. Full details and failures:
`~/sglang-workspace/frozen-model-20261028/MAINTENANCE.md`.

Adopt upstream cache/state/iteration ownership, segmented rendering and seeded
draft proposals. Remove the superseded Marlin bias patch. Adapt inactive-route
counts/zeroing for native Marlin, routed MMA and GEMV, preserving omitted rows.
Do not exclude the single-token GEMV to hide a contract gap. Auditing all 69
registered hooks found additional checkpoint/sampling argument changes, now
forwarded or explicitly implemented. Upstream post-verify graphs exclude the
classical draft-probability path and do not replace our optional sampling graph.

The first full-model smoke failed on the new Marlin keywords; the second failed
on new sampling keywords. Both are fixed. During GPU contract testing, TileLang
hoisted an inactive-row store; an outer block guard fixes it. Twenty-two Marlin
GPU tests, deterministic sampling, targeted CPU ownership/adapter tests and five
Compute Sanitizer cases pass (zero memory errors). Core census remains 60 paths.
No unrelated upstream whitespace cleanup was mixed into the merge.

One successful private process reuses the compatible October27 baseline:
FP4-Local/own MTP3, TP8/DCP4/EP1, 524K/FP16 KV/state/vision, paired4096/2048,
three coherent near8K categories, 32-token warmup/256 T0 outputs, seed531.

| Tokens/s, category mean | Before | After |
| --- | ---: | ---: |
| Prefill | 2,837.18 | 2,832.13 |
| Decode | 87.99 | 92.25 |

Acceptance increases 67.11% to 71.58%; cycle-time proxy is nearly flat
34.16 to 33.98 ms. All three greedy messages change. Treat this as a functional
integration/performance screen, not an isolated kernel gain or bit-exact model
equivalence. Image execution, 6144-token image-prefix reuse and T1/P.95 sampling
pass. No new full-context or Qwen qualification. Public hosting remains down;
the smoke workers were stopped.

## 2026-10-28 — FFN-only pilot implementation

Question: does one damped extra FFN/mHC application at layer23 meaningfully change
behavior, and what does it cost? This is a state-safe exploratory control, not a
reproduction of repeated full transformer blocks or a fitted semantic lens.

The existing forward-hook API selects `model.layers.23`. Require eager execution
for both phases, no MTP, TP8/EP1/PP1 and a fully written FP16 four-stream boundary.
Clone that state; reuse the original FFN PRE/norm, shared+routed MLP, its owed TP
sum and FFN POST. Blend in FP32 and cast once. Never execute attention a second
time. Changed later-layer histories are then ordinary histories of this declared
intervention. Store the mixed state through the residual ownership API.

Three arms share one process and a no-op logit processor: off; extra computation
with zero-strength output mixing; repeat with strength0.25. Separate cache salts
isolate arms. The zero arm returns the exact original state, including when
scratch is borrowed. CPU controls exercise aliasing and forward-flag restoration.
Only explicit untimed requests synchronize for last-position state observations.

Bounded execution: 16K context/4608 physical slots, same checkpoint and kernels,
strict dispatch, one coherent near8K/256-output throughput request per arm plus
cached continuation; three fixed arithmetic/code/logic cases capped at512 outputs.
These cases are a tiny pilot, not a held-out accuracy estimate. Require exact
off/zero token identity, actual hook execution on all eight ranks, finite state
changes and prefix reuse. Logs, raw prompts/results and controller stay at
`~/sglang-workspace/frozen-model-20261028/ffn-pilot/`.

## 2026-10-28 — FFN-only pilot results

Source `e3349102f`, private `glm-frozen-ffn-pilot-20261028`. All eight ranks
registered and executed the hook; graph phases and MTP were disabled as intended.
The one-process pilot completed all three arms. Eager decode was only about4TPS,
far below the graph/MTP serving profile. Its initial900s supervisor timeout was
too short for this measured rate. A bounded guard suspended only the idle
supervisor until its client exited; model/client execution and timing were
untouched, and the original systemd hard deadline remained. No reload or grid.

Ordinary arithmetic/code/logic answers are5263/387/A and the zero-strength arm
matches their entire78/435/397-token outputs. The near8K and cached-continuation
sequences diverge at token24/12 despite zero mixing. Two additional64-token
off/off requests also differ from each other, the original off arm and the zero
arm. Thus exact long-prompt token agreement is not a valid isolated-intervention
test in this runtime. This demonstrates baseline variation, not its cause and
not that an extra FFN has no side effects. The two ordinary repeats first differ
at token24; all five actual near8K prompt-token arrays are identical (7903tokens).
The small-case identity results are stronger evidence for this pilot; wider
causal claims remain blocked. The original all-cases identity assertion therefore
fails, deliberately preserved in the controller result; this is not a CUDA crash.

The untimed zero observation reports finite candidate states and zero mixing
delta, but its `zero_identity` field compares the returned anchor with the mixed
alias. It does not independently prove the anchor's pre-call contents survived.
A future native mutation control must own a pre-call copy and compare it after
the extra FFN, alongside fixed-input logits/selected-key observations. Do not
present the existing observer as that proof.

All three damped answers remain correct. These easy cases already had a perfect
ordinary score, so they cannot establish an accuracy improvement.

| Case | Ordinary answer | Damped answer | Ordinary output tokens | Damped output tokens |
| --- | --- | --- | ---: | ---: |
| Arithmetic | 5263 | 5263 | 78 | 290 |
| Short code | 387 | 387 | 435 | 412 |
| Logic | A | A | 397 | 384 |

Arithmetic adds a second check and a longer final explanation. Two cases shorten
slightly, but total output grows910→1086tokens and total completion time
232.19→279.45seconds (+20.4%). No output cap was reached on these three cases.
This is case evidence, not a held-out benchmark or general quality estimate.

One coherent near8K/256-output throughput screen; both columns use the same eager,
MTP-off setup and no-op logit processor. Do not compare these decode rates with
the graph/MTP production results above.

| Tokens/s | Ordinary | Extra FFN, strength0.25 |
| --- | ---: | ---: |
| Prefill | 2,889.54 | 2,900.55 |
| Decode | 4.05 | 3.87 |

Prefill is effectively flat; decode is about4.4% lower. Zero-strength extra
computation measures2,914.46prefill/3.85decodeTPS. These are one-request cost
screens, not statistically established gains. Each arm reuses6144tokens on its
continuation. Do not claim the faster later continuation prefill as an
optimization: first-shape/cold effects were not isolated. Extra off/off controls
queued serially; native throughput intervals exclude their queue wait.

Untimed last-position observations are finite. An undamped repeat moves the
state by41.7%/43.2% of its norm at the sampled prefill/decode positions;0.25mixing
reduces this to10.4%/10.8%. Cosines before mixing are0.951/0.948. These describe
movement, not semantic refinement, and are only two sampled positions.

**Disposition:** retain the implementation as a research control on `experimental`
only. No demonstrated capability benefit, no production/default change, no reason
to expand a strength/layer sweep from this pilot. Main/dev stay at `cdf91e2bf`.

**Next bounded step:** compare fixed inputs with an independently owned pre-call
residual snapshot, post-call mutation checks and selected-key/logit observations.
Separate ordinary numerical variation from intervention effects before repeating
full blocks or claiming semantic improvement. Use short diagnostic outputs first;
any later graph support must explicitly capture the hook rather than silently
skipping it. Lens fitting and full repeated-attention histories remain deferred.

Raw evidence and full report:
`~/sglang-workspace/frozen-model-20261028/ffn-pilot/REPORT.md` and `analysis.json`.
The supervisor resumed, every owned worker stopped, and no GPU compute processes
remained after completion. Public hosting stayed down throughout.

## Journal entry template

For each experiment: question; code/checkpoint revision; configuration and state
semantics; calibration vs held-out inputs; identity/numerical checks; observations;
behavior and cost; competing explanations; disposition; exact next bounded step.
Link local raw evidence, but do not commit large traces/captures or credentials.
Mark hypotheses, external claims and locally measured facts distinctly. Record
negative results and incomplete work as plainly as successes.
