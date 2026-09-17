"""Deterministic synthetic episode generators with no future-utility markers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from etrcm.stage1_1.events import EventKind, StructuredEvent


@dataclass(frozen=True)
class FactBatch:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class GraphBatch:
    sources: torch.Tensor
    destinations: torch.Tensor
    start: torch.Tensor
    target: torch.Tensor
    label: torch.Tensor
    distance: torch.Tensor


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _unique_rows(
    batch_size: int, count: int, symbol_count: int, generator: torch.Generator
) -> torch.Tensor:
    scores = torch.rand(batch_size, symbol_count, generator=generator)
    return torch.argsort(scores, dim=1)[:, :count]


def sample_facts(
    batch_size: int,
    count: int,
    symbol_count: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> FactBatch:
    keys = _unique_rows(batch_size, count, symbol_count, generator)
    values = torch.randint(
        0, symbol_count, (batch_size, count), generator=generator
    )
    return FactBatch(keys.to(device), values.to(device))


def fact_event(keys: torch.Tensor, values: torch.Tensor) -> StructuredEvent:
    return StructuredEvent.create(
        kind=EventKind.FACT, key_id=keys, value_id=values, write=True
    )


def query_event(keys: torch.Tensor, *, target: torch.Tensor | None = None) -> StructuredEvent:
    return StructuredEvent.create(
        kind=EventKind.QUERY if target is None else EventKind.GRAPH_QUERY,
        key_id=keys,
        aux_id=target,
        write=False,
    )


def interruption_event(keys: torch.Tensor) -> StructuredEvent:
    return StructuredEvent.create(
        kind=EventKind.INTERRUPTION, key_id=keys, write=False
    )


def unknowable_event(keys: torch.Tensor) -> StructuredEvent:
    return StructuredEvent.create(kind=EventKind.UNKNOWABLE, key_id=keys, write=False)


def sample_graphs(
    batch_size: int,
    min_distance: int,
    max_distance: int,
    symbol_count: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> GraphBatch:
    if max_distance + 3 > symbol_count:
        raise ValueError("symbol_count too small for unique graph nodes")
    distances = torch.randint(
        min_distance, max_distance + 1, (batch_size,), generator=generator
    )
    labels = torch.randint(0, 2, (batch_size,), generator=generator)
    nodes = _unique_rows(batch_size, max_distance + 3, symbol_count, generator)
    sources = torch.zeros(batch_size, max_distance, dtype=torch.long)
    destinations = torch.zeros_like(sources)
    targets = torch.zeros(batch_size, dtype=torch.long)
    for row in range(batch_size):
        distance = int(distances[row])
        path = nodes[row, : distance + 1]
        sources[row, :distance] = path[:-1]
        destinations[row, :distance] = path[1:]
        # Padding edges are self-loops on an unused symbol; they are masked later.
        if distance < max_distance:
            pad = nodes[row, max_distance + 1]
            sources[row, distance:] = pad
            destinations[row, distance:] = pad
        targets[row] = path[-1] if int(labels[row]) == 1 else nodes[row, max_distance + 2]
    return GraphBatch(
        sources.to(device),
        destinations.to(device),
        nodes[:, 0].to(device),
        targets.to(device),
        labels.float().to(device),
        distances.to(device),
    )


def graph_edge_event(
    graph: GraphBatch, edge_index: int
) -> StructuredEvent:
    active = graph.distance.gt(edge_index)
    return StructuredEvent.create(
        kind=EventKind.FACT,
        key_id=graph.sources[:, edge_index],
        value_id=graph.destinations[:, edge_index],
        write=active,
    )


def split_for_episode(split_salt: str, experiment: str, episode_id: str) -> str:
    digest = hashlib.sha256(
        f"{split_salt}:{experiment}:{episode_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket <= 69:
        return "train"
    if bucket <= 84:
        return "development"
    if bucket <= 92:
        return "validation"
    return "formal_test"

