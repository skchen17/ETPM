"""Finite state interventions for behavioral semantic-memory tests."""

from __future__ import annotations

import torch

from etrcm.stage1_1.model import LearnedState


def replace_memory(state: LearnedState, *, F: torch.Tensor | None = None,
                   M: torch.Tensor | None = None) -> LearnedState:
    return LearnedState(state.H.clone(), state.F.clone() if F is None else F.clone(),
                        state.M.clone() if M is None else M.clone(), state.tau, state.external_time)


def swap_component(a: LearnedState, b: LearnedState, component: str) -> tuple[LearnedState, LearnedState]:
    if a.F.shape != b.F.shape or component not in {"F", "M", "FM"}:
        raise ValueError("matched states and F/M/FM component required")
    return (replace_memory(a, F=b.F if component in {"F","FM"} else None,
                           M=b.M if component in {"M","FM"} else None),
            replace_memory(b, F=a.F if component in {"F","FM"} else None,
                           M=a.M if component in {"M","FM"} else None))


def derangement(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    if n < 2:
        raise ValueError("derangement requires at least two items")
    # A random non-zero cyclic shift is a guaranteed derangement.
    shift = int(torch.randint(1,n,(1,),generator=generator))
    return (torch.arange(n) + shift) % n


def shuffled_memory(state: LearnedState, permutation: torch.Tensor) -> LearnedState:
    batch = state.H.shape[0]
    if permutation.shape != (batch,) or bool((permutation == torch.arange(batch,device=permutation.device)).any()):
        raise ValueError("permutation must be a derangement")
    return replace_memory(state,F=state.F[permutation],M=state.M[permutation])


def random_norm_matched(state: LearnedState, generator: torch.Generator | None = None) -> LearnedState:
    def replacement(memory):
        random = torch.randn(memory.shape,device=memory.device,dtype=memory.dtype,generator=generator)
        original_norm = memory.norm(dim=(-2,-1),keepdim=True)
        return random * (original_norm / random.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-12))
    return replace_memory(state,F=replacement(state.F),M=replacement(state.M))
