"""Clear only route rows assigned to inactive experts, preserving omitted rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_inactive(
    OUTPUT,
    IDS,
    EXPERTS,
    COUNT,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CAPACITY: tl.constexpr,
    PREFIX_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    # A sorted producer supplies [padded tokens, inactive blocks]. Most forwards
    # have none; read that count before touching route IDs or output memory.
    limit = tl.load(COUNT + 1) if PREFIX_COUNT else tl.cdiv(tl.load(COUNT), BLOCK_M)
    if block < limit:
        expert = tl.load(EXPERTS + block)
        if expert < 0:
            for tile in range(tl.cdiv(BLOCK_M * N, BLOCK)):
                x = tile * BLOCK + tl.arange(0, BLOCK)
                offset = x // N
                slot = block * BLOCK_M + offset
                route = tl.load(
                    IDS + slot, (offset < BLOCK_M) & (slot < CAPACITY), ROWS
                )
                valid = (offset < BLOCK_M) & (route >= 0) & (route < ROWS)
                tl.store(OUTPUT + route * N + x % N, 0, valid)


def zero_inactive_routes(
    output,
    sorted_ids,
    expert_ids,
    padded_count,
    block_size,
    *,
    inactive_prefix_count=False,
):
    """GPU-only route cleanup for a GEMM which skips negative expert blocks.

    Within the live padded count each valid route must occur once. Padded IDs
    are outside output rows; omitted rows and unused metadata tails are untouched.
    The optional prefix count certifies sorted IDs and a second int32 count word.
    """
    if not (
        output.is_cuda
        and output.ndim == 2
        and output.is_contiguous()
        and output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and block_size > 0
        and sorted_ids.ndim == expert_ids.ndim == 1
        and triton.cdiv(sorted_ids.numel(), block_size) == expert_ids.numel()
        and all(
            t.device == output.device and t.is_contiguous() and t.dtype == torch.int32
            for t in (sorted_ids, expert_ids, padded_count)
        )
        and padded_count.numel() >= (2 if inactive_prefix_count else 1)
    ):
        raise ValueError("Invalid inactive expert route layout")
    if not output.numel() or not expert_ids.numel():
        return
    _zero_inactive[(expert_ids.numel(),)](
        output,
        sorted_ids,
        expert_ids,
        padded_count,
        output.shape[0],
        output.shape[1],
        block_size,
        sorted_ids.numel(),
        inactive_prefix_count,
        256,
    )
