"""Equal-budget multi-task training for Stage-1.3 streaming toys."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as Fnn

from etrcm.stage1_3.events import ContinuousEvent, Stage13EventKind, context_event, evidence_event
from etrcm.stage1_3.model import ContinuousETRCM


@dataclass
class TrainResult:
    logs: list[dict[str, float]]
    validation_loss: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ids(batch: int, count: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randint(0, count, (batch,), generator=generator).to(device)


def _mixed_evidence(
    target_key: torch.Tensor,
    target_value: torch.Tensor,
    support_mask: torch.Tensor,
    generator: torch.Generator,
    symbol_count: int,
) -> ContinuousEvent:
    device, batch = target_key.device, target_key.shape[0]
    keys = _ids(batch, symbol_count, generator, device)
    values = _ids(batch, symbol_count, generator, device)
    keys = torch.where(support_mask, target_key, keys)
    values = torch.where(support_mask, target_value, values)
    scalars = torch.zeros(batch, 4, device=device)
    scalars[:, 0] = 0.25
    return ContinuousEvent.create(
        kind=Stage13EventKind.EVIDENCE,
        key_id=keys,
        value_id=values,
        write=True,
        scalars=scalars,
    )


def accumulation_loss(
    model: ContinuousETRCM,
    *,
    batch: int,
    length: int,
    sufficient_count: int,
    insufficient_count: int,
    generator: torch.Generator,
    device: torch.device,
    emit_weight: float,
    content_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    symbols = model.config.symbol_count
    keys = _ids(batch, symbols, generator, device)
    values = _ids(batch, symbols, generator, device)
    sufficient = torch.rand(batch, generator=generator).to(device).lt(0.5)
    counts = torch.where(
        sufficient,
        torch.full((batch,), sufficient_count, device=device),
        torch.full((batch,), insufficient_count, device=device),
    )
    order = torch.rand(batch, length, generator=generator).argsort(dim=1).to(device)
    support = torch.zeros(batch, length, dtype=torch.bool, device=device)
    for index in range(length):
        support.scatter_(1, order[:, index : index + 1], (counts > index)[:, None])
    state = model.initial_state(batch, device=device)
    cumulative = torch.zeros(batch, dtype=torch.long, device=device)
    already_expressed = torch.zeros(batch, dtype=torch.bool, device=device)
    loss = torch.zeros((), device=device)
    content_terms = 0
    score_sum = 0.0
    for tick in range(length):
        active = support[:, tick]
        event = _mixed_evidence(keys, values, active, generator, symbols)
        state, output = model.step(state, event)
        cumulative = cumulative + active.long()
        should_emit = sufficient & cumulative.ge(sufficient_count) & ~already_expressed
        loss = loss + emit_weight * Fnn.binary_cross_entropy_with_logits(
            output["expression_logit"], should_emit.float()
        )
        if bool(should_emit.any()):
            loss = loss + content_weight * Fnn.cross_entropy(
                output["content_logits"][should_emit], values[should_emit]
            )
            content_terms += 1
            state = model.apply_self_output_feedback(state, values, should_emit)
            already_expressed |= should_emit
        score_sum += float(output["expression_score"].detach().mean())
    loss = loss / (length + content_terms)
    loss = loss + 0.002 * output["access"].mean()
    return loss, {"score_mean": score_sum / length, "positive_fraction": float(sufficient.float().mean())}


def memory_recovery_loss(
    model: ContinuousETRCM,
    *, batch: int, generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    symbols = model.config.symbol_count
    keys, values = _ids(batch, symbols, generator, device), _ids(batch, symbols, generator, device)
    state = model.initial_state(batch, device=device)
    loss = torch.zeros((), device=device)
    for _ in range(2):
        state, output = model.step(state, evidence_event(keys, values, support=0.5))
        loss = loss + 0.25 * Fnn.binary_cross_entropy_with_logits(
            output["expression_logit"], torch.zeros(batch, device=device)
        )
    for _ in range(2):
        state, output = model.step(state, None)
    for _ in range(12):
        noise_keys = _ids(batch, symbols, generator, device)
        noise_values = _ids(batch, symbols, generator, device)
        state, _ = model.step(state, evidence_event(noise_keys, noise_values, support=0.0))
    state = model.reset_active(state)
    state, output = model.step(state, context_event(keys))
    loss = loss + Fnn.binary_cross_entropy_with_logits(
        output["expression_logit"], torch.ones(batch, device=device)
    )
    loss = loss + Fnn.cross_entropy(output["content_logits"], values)
    return loss / 2.5 + 0.002 * output["access"].mean()


def revision_loss(
    model: ContinuousETRCM,
    *, batch: int, generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    symbols = model.config.symbol_count
    keys = _ids(batch, symbols, generator, device)
    old = _ids(batch, symbols, generator, device)
    offset = 1 + _ids(batch, symbols - 1, generator, device)
    new = (old + offset).remainder(symbols)
    state = model.initial_state(batch, device=device)
    for _ in range(4):
        state, output = model.step(state, evidence_event(keys, old, support=0.25))
    loss = Fnn.binary_cross_entropy_with_logits(output["expression_logit"], torch.ones(batch, device=device))
    loss = loss + Fnn.cross_entropy(output["content_logits"], old)
    state = model.apply_self_output_feedback(state, old)
    for _ in range(4):
        state, output = model.step(state, evidence_event(keys, new, support=0.25))
    loss = loss + Fnn.binary_cross_entropy_with_logits(output["expression_logit"], torch.ones(batch, device=device))
    loss = loss + Fnn.cross_entropy(output["content_logits"], new)
    return loss / 4.0 + 0.002 * output["access"].mean()


def train_model(
    model: ContinuousETRCM,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    stream_length: int,
    sufficient_count: int,
    insufficient_count: int,
    emit_weight: float,
    content_weight: float,
    weight_decay: float,
    gradient_clip: float,
) -> TrainResult:
    seed_everything(seed)
    model.to(device).train()
    generator = torch.Generator(device="cpu").manual_seed(seed + 91_000)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    logs: list[dict[str, float]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        task = step % 3
        if task == 0:
            loss, aux = accumulation_loss(
                model,
                batch=batch_size,
                length=stream_length,
                sufficient_count=sufficient_count,
                insufficient_count=insufficient_count,
                generator=generator,
                device=device,
                emit_weight=emit_weight,
                content_weight=content_weight,
            )
        elif task == 1:
            loss = memory_recovery_loss(model, batch=batch_size, generator=generator, device=device)
            aux = {}
        else:
            loss = revision_loss(model, batch=batch_size, generator=generator, device=device)
            aux = {}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            logs.append({"step": float(step), "loss": float(loss.detach()), **aux})
    model.eval()
    with torch.no_grad():
        validation, _ = accumulation_loss(
            model,
            batch=batch_size,
            length=stream_length,
            sufficient_count=sufficient_count,
            insufficient_count=insufficient_count,
            generator=torch.Generator(device="cpu").manual_seed(seed + 191_000),
            device=device,
            emit_weight=emit_weight,
            content_weight=content_weight,
        )
    return TrainResult(logs, float(validation))
