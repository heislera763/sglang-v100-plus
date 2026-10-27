"""Carry native HC injection partials through mainline's residual boundary."""

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.srt.runtime_context import get_parallel


@dataclass
class SM70GatedResidualState(GatedResidualState):
    gate: Optional[torch.Tensor] = None

    def _read(self, mix, residual, out_norm):
        if out_norm is not None:
            raise NotImplementedError(
                "a gated hyper-connection read with a separate norm"
            )
        hidden_states, values = mix(residual)
        if len(values) == 3:
            residual, self.normed, self.gate = values
        else:
            residual, self.normed = values
            self.gate = None
        return hidden_states, residual

    def apply_attn_combine(self, hidden_states, residual):
        return self.attn_combine(hidden_states, (residual, self.normed, self.gate))

    def apply_ffn_combine(self, hidden_states, residual):
        return self.ffn_combine(hidden_states, (residual, self.normed, self.gate))

    def clear_coefficients(self):
        super().clear_coefficients()
        self.gate = None

    def slice_residual_attn_tp(self, residual):
        if self.gate is not None:
            parallel = get_parallel()
            self.gate = self.gate.tensor_split(parallel.attn_tp_size)[
                parallel.attn_tp_rank
            ]
        return super().slice_residual_attn_tp(residual)
