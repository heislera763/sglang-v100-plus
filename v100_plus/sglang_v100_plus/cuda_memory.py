"""SM70 memory guards for explicit session pools and cold module loads."""

import torch

from sglang.srt.runtime_context import get_context, get_exec, get_schedule


def require_single_session_capacity(original, configurator, token_capacity):
    """An explicit one-session reservation must not silently shrink.

    Run the upstream constraint/PP synchronization first so every stage checks
    the same final capacity. Other serving policies retain upstream semantics.
    """
    capacity = original(configurator, token_capacity)
    schedule = get_schedule()
    requested = schedule.max_total_tokens
    if schedule.max_running_requests == 1 and requested is not None:
        aligned_capacity = capacity // schedule.page_size * schedule.page_size
        if aligned_capacity < requested:
            raise RuntimeError(
                f"Single-session KV reservation requires {requested} tokens, but the "
                f"profiled, PP-synchronized and page-aligned budget holds only "
                f"{aligned_capacity}. "
                "Refusing to shrink the requested pool. Check weights, draft/state "
                "reservations and runtime headroom; increasing the static fraction "
                "cannot create additional physical memory."
            )
    return capacity


def reclaim_for_triton_load(_module, _function, _name, _metadata_group, _hash):
    # cuModuleLoadData allocates outside Torch. Its OOM cannot trigger the
    # caching allocator's usual retry/reclamation, even when live tensors fit.
    # This callback runs on first device-load, never on warmed kernel replay.
    if not get_context().is_config_namespace_published("exec"):
        return  # Direct kernel users have no serving allocator to reclaim.
    if get_exec().features.enable_memory_saver:
        return  # Custom allocation arenas have their own segment ownership.
    if torch.cuda.is_current_stream_capturing():
        return
    free, _ = torch.cuda.mem_get_info()
    if free >= 1 << 30:
        return
    idle = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if idle < 64 << 20:
        return
    # Only unused allocator blocks are returned; live state/graph tensors and
    # the selected kernel remain unchanged. A real capacity OOM still raises.
    torch.cuda.empty_cache()


def install():
    import triton.knobs as knobs

    # HookChain.add is idempotent and preserves other first-load callbacks.
    knobs.runtime.kernel_load_start_hook.add(reclaim_for_triton_load)
