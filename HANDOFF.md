# Qwen Heretic LoRA

Task completed on `codex/qwen-heretic-lora`. Implementation `0ed458aed` is integrated into main/dev.

The scope covers the LoRA kernel namespace, Marlin down-projection wiring, V100 plugin integration, focused tests and README. Model data and runtime evidence stay outside Git. Hosting is down and all owned GPU workers are stopped.

The official Qwen FP8 model accepts the author's default nchapman rank-one adapter through `v100_static_lora` in the existing model-override JSON. One adapter is fixed at startup. All 144 targets are covered across PP2×TP4/EP4, with resident FP16 factors and original packed FP8 base execution. MTP3, FP16 KV/state, vision, prefix reuse and 262K allocation passed short model checks.

34 CPU tests and three SM70 GPU cases pass. GPU checks include independent FP64 results, native Marlin, masked/skewed routes, canaries and changing-input graph replay. Compute Sanitizer reports zero errors. Matched near8K/256-output T0 averages are 7141→6785 prefill TPS and70.9→70.4 generation TPS.

Full-window filling and a broad behavioral-quality/refusal evaluation remain unqualified. Private evidence and the excluded mismatched-warmup control are under `~/.local/state/server-work/qwen-heretic-lora/REPORT.md`. No shared host policy changes were required.
