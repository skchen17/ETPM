"""Preregistered constants for the Stage 2D.1 diagnostic audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CHECKPOINT_STEPS = (0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500)
INIT_SEEDS = tuple(range(9101, 9109))
DATA_SEEDS = tuple(range(12101, 12105))
EVAL_SEED = 15101
ROUTING_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
GATE_M_VALUES = (0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0)
CURRICULUM_P = (0.90, 0.80, 0.75, 0.70, 0.65)


@dataclass(frozen=True)
class FrozenStage2D1:
    gamma: float = 0.50
    rho_fast: float = 0.97
    rho_slow: float = 0.9995
    baseline_train_p: float = 0.70
    steps: int = 1500
    batch: int = 16
    evaluator_steps: int = 1000
    lengths: tuple[int, ...] = (16, 32, 64)
    factorial_initializations: int = 8
    factorial_streams: int = 4
    health_tv_min: float = 0.10
    health_interaction_min: float = 0.10
    health_bs_min: float = 0.10
    gate_required: int = 6
    gate_total: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


FROZEN_D1 = FrozenStage2D1()


def evidence_schedule(name: str, step: int, total: int = 1500) -> float:
    """Exact, deterministic curricula. ``step`` is zero based."""
    if not 0 <= step < total:
        raise ValueError((step, total))
    if name in {"C0", "C2"}:
        return 0.70
    if name == "C1":
        return CURRICULUM_P[min(4, (5 * step) // total)]
    if name == "C3":
        return 0.90 if step < 300 else 0.70
    if name == "C4":
        return CURRICULUM_P[min(4, (5 * step) // total)]
    raise ValueError(name)


def gate_floor(name: str, step: int, *, floor: float = 0.65, window: int = 100) -> float | None:
    if name not in {"C2", "C4"}:
        return None
    return floor if step < window else None
