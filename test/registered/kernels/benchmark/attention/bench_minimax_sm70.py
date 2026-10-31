"""Volta MiniMax indexer and sparse attention, including allocation/merge costs."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.minimax_sparse.sm70 import (
    minimax_index_scores_sm70,
    minimax_sparse_attention_sm70,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("rows", [1, 8, 1024], [1, 1024])
@marker.parametrize("history", [8192, 32768], [8192])
@marker.benchmark("operation", ["index", "attention"])
def benchmark(rows: int, history: int, operation: str):
    if torch.cuda.get_device_capability() != (7, 0):
        return marker.skip("Requires Volta")
    capacity = history + rows
    q = torch.randn(
        rows, 1 if operation == "index" else 8, 128, device="cuda", dtype=torch.float16
    )
    k = torch.randn(capacity, 1, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    slots = torch.randperm(capacity, device="cuda", dtype=torch.int32)
    prefix = torch.tensor([history], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([capacity], device="cuda", dtype=torch.int32)
    if operation == "index":
        fn = minimax_index_scores_sm70
        args = (q, k, slots, prefix, lengths, (capacity + 127) // 128)
    else:
        blocks = torch.arange(16, device="cuda", dtype=torch.int32)
        blocks = blocks[None, None].expand(1, rows, -1).contiguous()
        fn = minimax_sparse_attention_sm70
        args = (q, k, v, slots, prefix, lengths, blocks)
    # Rotate every input, including the page map. No repeated L2-hot weight or
    # cache bank; output allocation, score/partial scratch and merge are timed.
    return marker.do_bench(fn, input_args=args, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
