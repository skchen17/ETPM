"""Nine small, deterministic experiments for the Stage-1 decision gates."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from etrcm.memory import (
    decay_memory,
    external_delta_write,
    memory_read,
    nonconserving_replay,
    readout_conserving_consolidation,
    uniform_consolidation,
)
from etrcm.metrics import empty_record, retention_projection, safe_cosine
from etrcm.toys.generators import (
    advance_reachability,
    chain_episode,
    generator,
    interference_stream,
    orthogonal_facts,
    random_fact,
)

Tensor = torch.Tensor


def _params(config: dict[str, Any]) -> dict[str, Any]:
    memory = config["memory"]
    experiments = config.get("experiments", {})
    training = config.get("training", {})
    return {
        "dk": int(memory["key_dim"]),
        "dv": int(memory["value_dim"]),
        "gamma": float(memory["gamma"]),
        "rho_f": float(memory["rho_fast"]),
        "rho_m": float(memory["rho_slow"]),
        "eta": float(memory["eta_external"]),
        "interference": int(experiments.get("interference_steps", 32)),
        "toy3_steps": int(training.get("toy3_steps", 300)),
        "toy3_lr": float(training.get("toy3_lr", 0.08)),
    }


def _zeros(p: dict[str, Any]) -> tuple[Tensor, Tensor]:
    F = torch.zeros(p["dv"], p["dk"], dtype=torch.float64)
    return F, torch.zeros_like(F)


def _q_json(q: Tensor) -> str:
    return json.dumps([round(float(value), 8) for value in q.tolist()])


def _memory_record(
    *,
    toy: str,
    condition: str,
    seed: int,
    step: int,
    phase: str,
    F: Tensor,
    M: Tensor,
    q: Tensor | None = None,
    delta: Tensor | None = None,
    **values: Any,
) -> dict[str, Any]:
    a_norm = float(torch.linalg.vector_norm(F + M))
    transfer_fraction = None
    if delta is not None:
        transfer_fraction = float(torch.linalg.vector_norm(delta)) / max(
            float(torch.linalg.vector_norm(F + delta)), 1e-12
        )
    return empty_record(
        toy=toy,
        condition=condition,
        seed=seed,
        step=step,
        phase=phase,
        f_norm=float(torch.linalg.vector_norm(F)),
        m_norm=float(torch.linalg.vector_norm(M)),
        a_norm=a_norm,
        delta_norm=(float(torch.linalg.vector_norm(delta)) if delta is not None else 0.0),
        transfer_fraction=transfer_fraction,
        q_json=(_q_json(q) if q is not None else None),
        **values,
    )


def _write_and_use(
    F: Tensor, M: Tensor, key: Tensor, value: Tensor, p: dict[str, Any]
) -> tuple[Tensor, Tensor, Tensor]:
    F, M, _ = external_delta_write(F, M, key, value, p["eta"])
    F, M, delta = readout_conserving_consolidation(F, M, key, p["gamma"])
    return F, M, delta


def run_toy1(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    """Repeated real exposures followed by a shared interference protocol."""

    p = _params(config)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for exposures in (1, 2, 4, 8):
            rng = generator(10_000 + seed)
            target = random_fact(p["dk"], p["dv"], rng)
            stream = interference_stream(p["interference"], p["dk"], p["dv"], rng)
            F, M = _zeros(p)
            step = 0
            for _ in range(exposures):
                F, M, delta = _write_and_use(F, M, target.key, target.value, p)
                records.append(
                    _memory_record(
                        toy="toy1_repeated_exposure",
                        condition="full_etrcm",
                        seed=seed,
                        step=step,
                        phase="exposure",
                        F=F,
                        M=M,
                        q=target.key,
                        delta=delta,
                        parameter="exposures",
                        parameter_value=exposures,
                    )
                )
                F, M = decay_memory(F, M, p["rho_f"], p["rho_m"])
                step += 1
            for fact in stream:
                F, M, delta = _write_and_use(F, M, fact.key, fact.value, p)
                F, M = decay_memory(F, M, p["rho_f"], p["rho_m"])
                records.append(
                    _memory_record(
                        toy="toy1_repeated_exposure",
                        condition="full_etrcm",
                        seed=seed,
                        step=step,
                        phase="interference",
                        F=F,
                        M=M,
                        q=fact.key,
                        delta=delta,
                        parameter="exposures",
                        parameter_value=exposures,
                    )
                )
                step += 1
            slow_read = M @ target.key
            records.append(
                _memory_record(
                    toy="toy1_repeated_exposure",
                    condition="full_etrcm",
                    seed=seed,
                    step=step,
                    phase="final",
                    F=torch.zeros_like(F),
                    M=M,
                    q=target.key,
                    parameter="exposures",
                    parameter_value=exposures,
                    retention=retention_projection(slow_read, target.value),
                    accuracy=safe_cosine(slow_read, target.value),
                    memory_interference=float(
                        torch.linalg.vector_norm(slow_read - target.value)
                    ),
                )
            )
    return records


def _toy2_method(
    method: str,
    target,
    initial_facts,
    stream,
    reuses: int,
    p: dict[str, Any],
) -> tuple[Tensor, Tensor, float, float]:
    F, M = _zeros(p)
    A = torch.zeros_like(F)
    readout_drift = 0.0
    cumulative_access = 0.0

    if method == "single_persistent":
        for fact in initial_facts:
            update = p["eta"] * torch.outer(fact.value - A @ fact.key, fact.key)
            A = A + update
        for _ in range(reuses):
            A = p["rho_m"] * A
        for fact in stream:
            update = p["eta"] * torch.outer(fact.value - A @ fact.key, fact.key)
            A = p["rho_m"] * (A + update)
        return torch.zeros_like(A), A, 0.0, float(reuses)

    for fact in initial_facts:
        F, M, _ = external_delta_write(F, M, fact.key, fact.value, p["eta"])

    for _ in range(reuses):
        total_before = F + M
        if method == "full_etrcm":
            F, M, _ = readout_conserving_consolidation(F, M, target.key, p["gamma"])
            cumulative_access += 1.0
        elif method == "uniform_transfer":
            F, M, _ = uniform_consolidation(F, M, p["gamma"] / p["dk"])
        elif method == "nonconserving_replay":
            F, M, _ = nonconserving_replay(F, M, target.key, p["gamma"])
            cumulative_access += 1.0
        elif method == "no_idle":
            pass
        else:
            raise ValueError(method)
        readout_drift += float(torch.linalg.vector_norm((F + M) - total_before))
        F, M = decay_memory(F, M, p["rho_f"], p["rho_m"])

    for fact in stream:
        F, M, _ = external_delta_write(F, M, fact.key, fact.value, p["eta"])
        if method == "uniform_transfer":
            F, M, _ = uniform_consolidation(F, M, p["gamma"] / p["dk"])
        else:
            F, M, _ = readout_conserving_consolidation(F, M, fact.key, p["gamma"])
        F, M = decay_memory(F, M, p["rho_f"], p["rho_m"])
    return F, M, readout_drift, cumulative_access


def run_toy2(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    p = _params(config)
    records: list[dict[str, Any]] = []
    methods = (
        "full_etrcm",
        "uniform_transfer",
        "nonconserving_replay",
        "no_idle",
        "single_persistent",
    )
    for seed in seeds:
        rng = generator(20_000 + seed)
        facts = orthogonal_facts(8, p["dk"], p["dv"], rng)
        target = facts[0]
        stream = interference_stream(p["interference"], p["dk"], p["dv"], rng)
        for reuses in (0, 1, 2, 4, 8):
            for method in methods:
                F, M, drift, cumulative = _toy2_method(
                    method, target, facts, stream, reuses, p
                )
                slow_read = M @ target.key
                records.append(
                    _memory_record(
                        toy="toy2_repeated_use",
                        condition=method,
                        seed=seed,
                        step=p["interference"] + reuses,
                        phase="final",
                        F=torch.zeros_like(F),
                        M=M,
                        q=target.key,
                        parameter="reuses",
                        parameter_value=reuses,
                        reuse_count=reuses,
                        cumulative_access=cumulative,
                        retention=retention_projection(slow_read, target.value),
                        accuracy=safe_cosine(slow_read, target.value),
                        readout_drift=drift,
                        memory_interference=float(
                            torch.linalg.vector_norm(slow_read - target.value)
                        ),
                    )
                )
    return records


class _AccessPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, context: Tensor) -> Tensor:
        return torch.sigmoid(self.linear(context)).squeeze(-1)


def run_toy3(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    """Same fact event, different downstream task context and future loss."""

    p = _params(config)
    records: list[dict[str, Any]] = []
    contexts = torch.eye(2, dtype=torch.float64)
    for seed in seeds:
        torch.manual_seed(30_000 + seed)
        policy = _AccessPolicy().double()
        optimizer = torch.optim.Adam(policy.parameters(), lr=p["toy3_lr"])
        loss_value = math.nan
        for _ in range(p["toy3_steps"]):
            gates = policy(contexts)
            # Both episodes receive the same fact. Only the future task in row 0
            # repeatedly needs it; row 1 contributes a slow-state pollution cost.
            useful_retention = 1.0 - (1.0 - p["gamma"] * gates[0]) ** 4
            loss = (1.0 - useful_retention) ** 2 + 0.08 * gates[1] + 0.01 * gates[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
        gates = policy(contexts).detach()

        rng = generator(30_000 + seed)
        fact = random_fact(p["dk"], p["dv"], rng)
        for index, condition in enumerate(("future_useful", "future_unused")):
            F, M = _zeros(p)
            F, M, _ = external_delta_write(F, M, fact.key, fact.value, p["eta"])
            for _ in range(4):
                gamma = p["gamma"] * float(gates[index])
                F, M, _ = readout_conserving_consolidation(F, M, fact.key, gamma)
            slow_read = M @ fact.key
            records.append(
                _memory_record(
                    toy="toy3_same_input_future_utility",
                    condition=condition,
                    seed=seed,
                    step=p["toy3_steps"],
                    phase="final",
                    F=torch.zeros_like(F),
                    M=M,
                    q=fact.key,
                    parameter="learned_access_gate",
                    parameter_value=float(gates[index]),
                    cumulative_access=4.0 * float(gates[index]),
                    retention=retention_projection(slow_read, fact.value),
                    loss=loss_value,
                    notes="identical fact event; distinct unlabeled task-context vector",
                )
            )
    return records


def _graph_hidden(episode, ticks: int) -> tuple[Tensor, float, float]:
    hidden = torch.zeros(episode.adjacency.shape[0], dtype=torch.float64)
    hidden[episode.start] = 1.0
    initial = hidden.clone()
    last_change = 0.0
    for _ in range(ticks):
        updated = advance_reachability(hidden, episode.adjacency)
        last_change = float(torch.linalg.vector_norm(updated - hidden))
        hidden = updated
    return hidden, float(torch.linalg.vector_norm(hidden - initial)), last_change


def _graph_bank(seed: int, count: int = 128) -> list[Any]:
    bank = []
    for index in range(count):
        reachable = index % 2 == 0
        distance = 1 + (index % 8)
        bank.append(chain_episode(seed * 10_000 + index, reachable, distance))
    return bank


def run_toy4(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        bank = _graph_bank(40_000 + seed)
        for ticks in (0, 1, 2, 4, 8, 16):
            correct = 0
            changes = []
            convergence = []
            for episode in bank:
                hidden, total_change, last_change = _graph_hidden(episode, ticks)
                answer = bool(hidden[episode.target] > 0.5)
                correct += int(answer == episode.reachable)
                changes.append(total_change)
                convergence.append(last_change)
            records.append(
                empty_record(
                    toy="toy4_idle_reasoning",
                    condition="idle_compute",
                    seed=seed,
                    step=ticks,
                    phase="final",
                    parameter="idle_ticks",
                    parameter_value=ticks,
                    accuracy=correct / len(bank),
                    h_norm=float(torch.tensor(changes).mean()),
                    h_change=float(torch.tensor(changes).mean()),
                    state_convergence=float(torch.tensor(convergence).mean()),
                    idle_trajectory_length=ticks,
                    compute_budget=ticks,
                    latency_steps=0,
                )
            )
    return records


def run_toy5(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        bank = _graph_bank(50_000 + seed)
        for ticks in (0, 1, 2, 4, 8, 16):
            finals: dict[str, list[Tensor]] = {"idle_compute": [], "query_time_compute": []}
            accuracies: dict[str, int] = {"idle_compute": 0, "query_time_compute": 0}
            for episode in bank:
                idle_hidden, _, _ = _graph_hidden(episode, ticks)
                # Freeze during idle, then execute the exact same transition K times.
                query_hidden, _, _ = _graph_hidden(episode, ticks)
                finals["idle_compute"].append(idle_hidden)
                finals["query_time_compute"].append(query_hidden)
                accuracies["idle_compute"] += int(
                    bool(idle_hidden[episode.target] > 0.5) == episode.reachable
                )
                accuracies["query_time_compute"] += int(
                    bool(query_hidden[episode.target] > 0.5) == episode.reachable
                )
            state_diff = float(
                torch.stack(
                    [
                        torch.linalg.vector_norm(left - right)
                        for left, right in zip(
                            finals["idle_compute"], finals["query_time_compute"]
                        )
                    ]
                ).mean()
            )
            for condition in ("idle_compute", "query_time_compute"):
                records.append(
                    empty_record(
                        toy="toy5_matched_compute",
                        condition=condition,
                        seed=seed,
                        step=ticks,
                        phase="final",
                        parameter="compute_ticks",
                        parameter_value=ticks,
                        accuracy=accuracies[condition] / len(bank),
                        state_convergence=state_diff,
                        idle_trajectory_length=(ticks if condition == "idle_compute" else 0),
                        latency_steps=(0 if condition == "idle_compute" else ticks),
                        compute_budget=ticks,
                        notes="same transition count; state_convergence stores matched final-state difference",
                    )
                )
    return records


def run_toy6(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    p = _params(config)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        rng = generator(60_000 + seed)
        fact = random_fact(p["dk"], p["dv"], rng)
        for ticks in (0, 2, 4, 8, 16):
            F, M = _zeros(p)
            F, M, _ = external_delta_write(F, M, fact.key, fact.value, p["eta"])
            drift = 0.0
            for step in range(ticks):
                before = F + M
                F, M, delta = readout_conserving_consolidation(
                    F, M, fact.key, p["gamma"]
                )
                drift += float(torch.linalg.vector_norm((F + M) - before))
                records.append(
                    _memory_record(
                        toy="toy6_idle_consolidation",
                        condition="full_etrcm",
                        seed=seed,
                        step=step,
                        phase="idle",
                        F=F,
                        M=M,
                        q=fact.key,
                        delta=delta,
                        parameter="idle_ticks",
                        parameter_value=ticks,
                        reuse_count=step + 1,
                        cumulative_access=step + 1,
                        readout_drift=drift,
                    )
                )
                F, M = decay_memory(F, M, p["rho_f"], p["rho_m"])
            slow_read = M @ fact.key
            records.append(
                _memory_record(
                    toy="toy6_idle_consolidation",
                    condition="full_etrcm",
                    seed=seed,
                    step=ticks,
                    phase="final",
                    F=torch.zeros_like(F),
                    M=M,
                    q=fact.key,
                    parameter="idle_ticks",
                    parameter_value=ticks,
                    reuse_count=ticks,
                    cumulative_access=ticks,
                    retention=retention_projection(slow_read, fact.value),
                    accuracy=safe_cosine(slow_read, fact.value),
                    readout_drift=drift,
                    idle_trajectory_length=ticks,
                )
            )
    return records


def run_toy7(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    del config
    records: list[dict[str, Any]] = []
    sample_count = 4096
    for seed in seeds:
        rng = generator(70_000 + seed)
        targets = torch.randint(0, 2, (sample_count,), generator=rng)
        for ticks in (0, 1, 2, 4, 8, 16, 32):
            probability = torch.full((sample_count,), 0.5, dtype=torch.float64)
            prediction = torch.zeros(sample_count, dtype=torch.long)
            accuracy = float((prediction == targets).double().mean())
            confidence = float(torch.maximum(probability, 1.0 - probability).mean())
            calibration = abs(accuracy - confidence)
            records.append(
                empty_record(
                    toy="toy7_unknowable_bit",
                    condition="full_etrcm_null_evidence",
                    seed=seed,
                    step=ticks,
                    phase="final",
                    parameter="idle_ticks",
                    parameter_value=ticks,
                    accuracy=accuracy,
                    confidence=confidence,
                    calibration=calibration,
                    idle_trajectory_length=ticks,
                    compute_budget=ticks,
                    notes="no bit evidence; fixed maximum-entropy predictive head",
                )
            )
    return records


def run_toy8(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    p = _params(config)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        rng = generator(80_000 + seed)
        key = random_fact(p["dk"], p["dv"], rng).key
        old_value = torch.zeros(p["dv"], dtype=torch.float64)
        new_value = torch.zeros_like(old_value)
        old_value[0] = 1.0
        new_value[1] = 1.0
        for old_exposures in (1, 4, 8):
            for new_exposures in (1, 2, 4):
                for new_reuses in (0, 2, 8):
                    F, M = _zeros(p)
                    for _ in range(old_exposures):
                        F, M, _ = _write_and_use(F, M, key, old_value, p)
                        F, M = decay_memory(F, M, p["rho_f"], p["rho_m"])
                    # Give the old trace slow-state stability before revision.
                    for _ in range(8):
                        F, M, _ = readout_conserving_consolidation(
                            F, M, key, p["gamma"]
                        )
                    for _ in range(new_exposures):
                        F, M, _ = external_delta_write(F, M, key, new_value, p["eta"])
                        F, M = decay_memory(F, M, p["rho_f"], p["rho_m"])
                    for _ in range(new_reuses):
                        F, M, _ = readout_conserving_consolidation(
                            F, M, key, p["gamma"]
                        )
                    # Delayed revision is evaluated after the volatile trace is
                    # removed, so reuse must consolidate the correction into M.
                    read = M @ key
                    probability_new = float(torch.softmax(5.0 * read[:2], dim=0)[1])
                    records.append(
                        _memory_record(
                            toy="toy8_memory_revision",
                            condition="full_etrcm",
                            seed=seed,
                            step=old_exposures + new_exposures + new_reuses,
                            phase="final",
                            F=F,
                            M=M,
                            q=key,
                            parameter="new_exposures",
                            parameter_value=new_exposures,
                            reuse_count=new_reuses,
                            cumulative_access=new_reuses,
                            accuracy=probability_new,
                            confidence=max(probability_new, 1.0 - probability_new),
                            answer=probability_new,
                            target=1,
                            notes=f"old_exposures={old_exposures};new_reuses={new_reuses}",
                        )
                    )
    return records


def run_toy9(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    p = _params(config)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        rng = generator(90_000 + seed)
        local_fact, stable_fact = orthogonal_facts(2, p["dk"], p["dv"], rng)
        F, M = _zeros(p)
        for fact in (local_fact, stable_fact):
            F, M, _ = external_delta_write(F, M, fact.key, fact.value, p["eta"])
        # Both facts have the same input form. Future task statistics differ.
        for episode in range(8):
            F, M, _ = readout_conserving_consolidation(
                F, M, stable_fact.key, p["gamma"]
            )
            if episode == 0:
                F, M, _ = readout_conserving_consolidation(
                    F, M, local_fact.key, p["gamma"]
                )
            F, M = decay_memory(F, M, p["rho_f"], p["rho_m"])
        for condition, fact, uses in (
            ("episode_local", local_fact, 1),
            ("cross_episode_stable", stable_fact, 8),
        ):
            slow_read = M @ fact.key
            records.append(
                _memory_record(
                    toy="toy9_temporary_persistent",
                    condition=condition,
                    seed=seed,
                    step=8,
                    phase="final",
                    F=torch.zeros_like(F),
                    M=M,
                    q=fact.key,
                    parameter="future_uses",
                    parameter_value=uses,
                    reuse_count=uses,
                    cumulative_access=uses,
                    retention=retention_projection(slow_read, fact.value),
                    notes="input form matched; only cross-episode use statistics differ",
                )
            )
    return records


TOY_RUNNERS: dict[str, Callable[[dict[str, Any], list[int]], list[dict[str, Any]]]] = {
    "1": run_toy1,
    "2": run_toy2,
    "3": run_toy3,
    "4": run_toy4,
    "5": run_toy5,
    "6": run_toy6,
    "7": run_toy7,
    "8": run_toy8,
    "9": run_toy9,
}


def run_named_toy(
    name: str, config: dict[str, Any], seeds: list[int]
) -> list[dict[str, Any]]:
    normalized = name.lower().removeprefix("toy")
    if normalized not in TOY_RUNNERS:
        raise ValueError(f"unknown toy {name!r}; choose 1..9 or all")
    return TOY_RUNNERS[normalized](config, seeds)


def run_all_toys(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in sorted(TOY_RUNNERS, key=int):
        records.extend(TOY_RUNNERS[name](config, seeds))
    return records
