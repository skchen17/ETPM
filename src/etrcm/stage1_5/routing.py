"""Historical-only oracle bank and finite-benefit routing supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_3.events import ContinuousEvent
from .model import AnatomicalETRCM


@dataclass(frozen=True)
class HistoricalReadBank:
    # Every entry comes from an actual early external event and memory state.
    raw_reads: torch.Tensor  # [item,batch,value_dim]
    key_ids: torch.Tensor  # [item,batch]
    source_event_indices: torch.Tensor  # [item]
    valid: torch.Tensor  # [item,batch]


@torch.no_grad()
def build_historical_bank(
    model: AnatomicalETRCM, events: tuple[ContinuousEvent, ...],
    *, history_length: int = 4,
) -> tuple[LearnedState, HistoricalReadBank]:
    """Build candidates before bridge; no target or future-event argument exists."""
    if history_length < 1 or history_length > len(events):
        raise ValueError("invalid history length")
    batch = int(events[0].key_id.shape[0])
    state = model.initial_state(batch, device=events[0].key_id.device)
    reads, keys, valid = [], [], []
    for index, event in enumerate(events[:history_length]):
        event.validate(batch)
        state, _ = model.step(state, event)
        key = model.target_key(event.key_id.clamp_min(0))
        raw = torch.einsum("bvk,bk->bv", state.F + state.M, key)
        reads.append(raw.detach().clone())
        keys.append(event.key_id.detach().clone())
        valid.append(event.write_mask.detach().clone())
    return state, HistoricalReadBank(
        torch.stack(reads), torch.stack(keys),
        torch.arange(history_length, device=state.H.device), torch.stack(valid),
    )


def _eligible(bank: HistoricalReadBank, bridge: ContinuousEvent) -> torch.Tensor:
    # Bridge value is a genuinely observed B in C→B, not a future target A.
    bridge.validate(bank.raw_reads.shape[1])
    eligible = bank.valid & bank.key_ids.eq(bridge.value_id[None, :])
    if not bool(eligible.any(dim=0).all()):
        raise ValueError("no observed historical source for at least one bridge cue")
    return eligible


def static_oracle_read(
    bank: HistoricalReadBank, bridge: ContinuousEvent,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Last actual historical write matching the observed bridge cue."""
    eligible = _eligible(bank, bridge)
    index = torch.where(
        eligible,
        torch.arange(len(bank.source_event_indices), device=eligible.device)[:, None],
        -1,
    ).max(dim=0).values
    batch_index = torch.arange(bank.raw_reads.shape[1], device=index.device)
    return bank.raw_reads[index, batch_index], bank.source_event_indices[index]


def closed_loop_oracle_read(
    model: AnatomicalETRCM, H: torch.Tensor,
    bank: HistoricalReadBank, bridge: ContinuousEvent,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-score eligible historical reads from the *current* H every tick.

    This is a diagnostic H-dependent oracle-bank policy, not a learned route.
    It never sees a future label. Its candidate set is only true history.
    """
    eligible = _eligible(bank, bridge)
    pooled = model._pool(H)
    width = bank.raw_reads.shape[-1]
    context = torch.tanh(pooled[:, :width])
    scores = torch.einsum("ibv,bv->ib", bank.raw_reads, context)
    scores = scores.masked_fill(~eligible, -torch.inf)
    probabilities = torch.softmax(scores, dim=0)
    selected = torch.einsum("ib,ibv->bv", probabilities, bank.raw_reads)
    source = scores.argmax(dim=0)
    return selected, bank.source_event_indices[source]
