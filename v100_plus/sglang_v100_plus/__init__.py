"""Opt-in SM70 compatibility for current SGLang."""

import os


def register():
    if os.environ.get("SGLANG_V100_PLUS") != "1":
        from .dispatch import reject_fallback

        reject_fallback(
            "plugin.activation",
            "strict dispatch requires the V100 plugin to be active (SGLANG_V100_PLUS=1)",
        )
        return
    from sglang.srt.plugins.hook_registry import HookRegistry

    from . import runtime

    required = getattr(runtime, "REQUIRED_HOOKS", None)
    if required and required.issubset(HookRegistry._hooks):
        return
    runtime.install()
