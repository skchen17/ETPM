"""Stage 2C wrapper around the frozen Stage 1.5 transition and memory law."""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_3.events import ContinuousEvent
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_5.model import AnatomicalETRCM


VARIANTS = {"full": "B5_separate", "no_memory": "B0_no_memory",
            "gru": "B1_gru", "gamma_zero": "B6_gamma_zero",
            "f_only": "B5_separate", "m_disabled": "B5_separate"}


class BehavioralModel(nn.Module):
    """Predicts observable consequence, not the latent rule or correct action."""

    def __init__(self, config: Stage14Config, variant: str = "full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.config = config
        self.variant = variant
        self.core = AnatomicalETRCM(config, mode=VARIANTS[variant])
        self.consequence_head = nn.Linear(config.hidden_dim, 4)

    def initial_state(self, batch: int, device=None) -> LearnedState:
        return self.core.initial_state(batch, device=device)

    def step(self, state: LearnedState, event: ContinuousEvent | None,
             *, read_clamp: str = "none", write_block: bool = False):
        if read_clamp not in {"none", "F", "M", "FM"}:
            raise ValueError(read_clamp)
        if read_clamp != "none" and self.variant in {"no_memory", "gru"}:
            raise ValueError("read intervention requires F/M model")
        if write_block and event is not None:
            from dataclasses import replace
            event = replace(event, write_mask=torch.zeros_like(event.write_mask))
        if self.variant in {"m_disabled", "no_memory", "gru"}:
            state = LearnedState(state.H,
                                 torch.zeros_like(state.F) if self.variant in {"no_memory", "gru"} else state.F,
                                 torch.zeros_like(state.M),
                                 state.tau, state.external_time)
        fast_zero = torch.zeros(state.H.shape[0], self.config.value_dim,
                                device=state.H.device, dtype=state.H.dtype)
        slow_zero = fast_zero
        fast_override = fast_zero if read_clamp in {"F", "FM"} else None
        slow_override = slow_zero if read_clamp in {"M", "FM"} or self.variant in {"f_only", "m_disabled"} else None
        if fast_override is not None or slow_override is not None:
            new_state, diag = self.core.step_with_read(
                state, event, fast_read_override=fast_override,
                slow_read_override=slow_override)
        else:
            new_state, diag = self.core.step(state, event)
        if self.variant in {"m_disabled", "no_memory", "gru"}:
            new_state = LearnedState(new_state.H,
                                     torch.zeros_like(new_state.F) if self.variant in {"no_memory", "gru"} else new_state.F,
                                     torch.zeros_like(new_state.M),
                                     new_state.tau, new_state.external_time)
        return new_state, diag

    def consequence_logits(self, state: LearnedState) -> torch.Tensor:
        return self.consequence_head(self.core._pool(state.H))

    def parameter_digest(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in self.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
