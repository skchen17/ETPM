"""Data generators contain task structure, not remember/importance labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from etrcm.memory import normalize


@dataclass(frozen=True)
class Fact:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class GraphEpisode:
    adjacency: torch.Tensor
    start: int
    target: int
    reachable: bool
    distance: int


def generator(seed: int) -> torch.Generator:
    value = torch.Generator(device="cpu")
    value.manual_seed(int(seed))
    return value


def random_unit(size: int, rng: torch.Generator) -> torch.Tensor:
    return normalize(torch.randn(size, generator=rng, dtype=torch.float64))


def random_fact(key_dim: int, value_dim: int, rng: torch.Generator) -> Fact:
    return Fact(random_unit(key_dim, rng), random_unit(value_dim, rng))


def orthogonal_facts(
    count: int, key_dim: int, value_dim: int, rng: torch.Generator
) -> list[Fact]:
    raw = torch.randn(key_dim, min(count, key_dim), generator=rng, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    facts: list[Fact] = []
    for index in range(count):
        key = q[:, index] if index < q.shape[1] else random_unit(key_dim, rng)
        facts.append(Fact(key, random_unit(value_dim, rng)))
    return facts


def interference_stream(
    count: int, key_dim: int, value_dim: int, rng: torch.Generator
) -> list[Fact]:
    return [random_fact(key_dim, value_dim, rng) for _ in range(count)]


def chain_episode(seed: int, reachable: bool, distance: int, nodes: int = 12) -> GraphEpisode:
    rng = generator(seed)
    permutation = torch.randperm(nodes, generator=rng).tolist()
    adjacency = torch.zeros(nodes, nodes, dtype=torch.float64)
    start = permutation[0]
    if reachable:
        path = permutation[: distance + 1]
        for left, right in zip(path[:-1], path[1:]):
            adjacency[left, right] = 1.0
        target = path[-1]
    else:
        split = max(2, nodes // 2)
        left_nodes = permutation[:split]
        right_nodes = permutation[split:]
        for group in (left_nodes, right_nodes):
            for left, right in zip(group[:-1], group[1:]):
                adjacency[left, right] = 1.0
        start = left_nodes[0]
        target = right_nodes[-1]
    return GraphEpisode(adjacency, start, target, reachable, distance)


def advance_reachability(hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """One recurrent latent-compute tick (soft Boolean transitive closure)."""

    return torch.clamp(hidden + hidden @ adjacency, 0.0, 1.0)

