"""Associative-memory primitives with explicit evidence/use separation."""

from __future__ import annotations

import torch


def normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector).clamp_min(eps)


def memory_read(F: torch.Tensor, M: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (F + M) @ normalize(q)


def external_delta_write(
    F: torch.Tensor,
    M: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write genuinely external evidence into F; never call on NULL_EVENT."""

    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0,1]")
    key = normalize(key)
    if F.shape != M.shape:
        raise ValueError("F and M must have the same shape")
    if key.shape != (F.shape[1],) or value.shape != (F.shape[0],):
        raise ValueError("key/value dimensions do not match memory")
    residual = value - (F + M) @ key
    update = eta * torch.outer(residual, key)
    return F + update, M, update


def readout_conserving_consolidation(
    F: torch.Tensor,
    M: torch.Tensor,
    query: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move the currently used fast direction to slow state without new evidence."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    q = normalize(query)
    delta = gamma * torch.outer(F @ q, q)
    return F - delta, M + delta, delta


def uniform_consolidation(
    F: torch.Tensor, M: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """B3: direction-independent transfer, conserving the total matrix."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    delta = gamma * F
    return F - delta, M + delta, delta


def nonconserving_replay(
    F: torch.Tensor,
    M: torch.Tensor,
    query: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """B4 negative control: replay adds another copy and changes F+M."""

    q = normalize(query)
    delta = gamma * torch.outer(F @ q, q)
    return F, M + delta, delta


def decay_memory(
    F: torch.Tensor, M: torch.Tensor, rho_fast: float, rho_slow: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 <= rho_fast <= 1.0 and 0.0 <= rho_slow <= 1.0):
        raise ValueError("decay coefficients must be in [0,1]")
    return rho_fast * F, rho_slow * M

