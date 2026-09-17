"""Pure query decompositions used for functional addressing interventions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QueryInterventions:
    original: torch.Tensor
    parallel: torch.Tensor
    perpendicular: torch.Tensor
    signflip: torch.Tensor
    random: torch.Tensor
    target_zero: torch.Tensor
    strongest_nontarget_zero: torch.Tensor
    signed_cosine: torch.Tensor
    absolute_cosine: torch.Tensor
    squared_projection: torch.Tensor
    target_nontarget_margin: torch.Tensor
    strongest_nontarget_index: torch.Tensor


def _unit(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(eps)


def decompose_query(
    query: torch.Tensor,
    target_key: torch.Tensor,
    non_target_keys: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> QueryInterventions:
    """Decompose q against a unit target and remove a matched non-target component."""

    if query.ndim != 2 or target_key.shape != query.shape:
        raise ValueError("query and target_key must be [batch,key_dim]")
    if non_target_keys.ndim != 3 or non_target_keys.shape[:1] != query.shape[:1]:
        raise ValueError("non_target_keys must be [batch,count,key_dim]")
    target = _unit(target_key)
    coefficient = (query * target).sum(-1, keepdim=True)
    parallel = coefficient * target
    perpendicular = query - parallel
    similarities = torch.einsum("bd,bnd->bn", query, _unit(non_target_keys))
    strongest_index = similarities.abs().argmax(-1)
    row = torch.arange(query.shape[0], device=query.device)
    strongest = _unit(non_target_keys)[row, strongest_index]
    strongest_coefficient = (query * strongest).sum(-1, keepdim=True)
    strongest_zero = query - strongest_coefficient * strongest
    random = torch.randn(
        query.shape,
        generator=generator,
        device="cpu",
        dtype=query.dtype,
    ).to(query.device)
    random = _unit(random) * torch.linalg.vector_norm(query, dim=-1, keepdim=True)
    signed = coefficient.squeeze(-1) / torch.linalg.vector_norm(query, dim=-1).clamp_min(1e-8)
    non_target_abs = similarities.abs().amax(-1)
    return QueryInterventions(
        original=query,
        parallel=parallel,
        perpendicular=perpendicular,
        signflip=-query,
        random=random,
        target_zero=perpendicular,
        strongest_nontarget_zero=strongest_zero,
        signed_cosine=signed,
        absolute_cosine=signed.abs(),
        squared_projection=coefficient.squeeze(-1).pow(2),
        target_nontarget_margin=signed.abs() - non_target_abs,
        strongest_nontarget_index=strongest_index,
    )

