"""Preregistered Stage 2D protocol constants.

This module is deliberately data-free.  Formal evaluation reads these values;
the analyzer refuses to adjudicate a run whose manifest disagrees with them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


FORMATION_N = (0, 1, 2, 4, 8, 16, 32, 64)
P_LEVELS = (0.55, 0.60, 0.65, 0.70)
PRIMARY_P_LEVELS = (0.60, 0.70)
DELAYS = (0, 10, 50, 100, 250, 500, 1000, 2000, 5000)
REVERSAL_R = (0, 1, 2, 4, 8, 16, 32, 64, 128)
OPPOSE_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
NULL_TICKS = (0, 1, 2, 4, 8, 16, 32, 64)
CONTINUOUS_TICKS = (100, 500, 1000, 5000, 10000)
HANDOFF_TIMES = ("early_formation", "late_formation", "post_formation", "mid_delay", "late_delay", "probe")


@dataclass(frozen=True)
class FrozenProtocol:
    train_evidence_p: float = 0.70
    formation_length: int = 32
    primary_p_low: float = 0.60
    primary_p_high: float = 0.70
    persistence_delay: int = 500
    persistence_ratio_min: float = 0.20
    formed_threshold: float = 0.10
    acquisition_threshold: float = 0.10
    reversal_max: int = 128
    selectivity_margin: float = 0.05
    consolidation_retention_margin: float = 0.03
    # Frozen after development; noisy evidence has a lower theoretical ceiling
    # than deterministic Stage 2C.3, so its raw G57 numbers are not copied.
    health_action_tv_min: float = 0.10
    health_interaction_min: float = 0.10
    replicates: int = 32
    gate_seed_required: int = 6
    gate_seed_total: int = 8
    training_steps: int = 1500
    evaluator_pretrain_steps: int = 1000
    training_batch: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


FROZEN = FrozenProtocol()


COARSE_CONFIGS = (
    ("g000_f0970_m09995", 0.00, 0.970, 0.9995),
    ("g005_f0950_m09950", 0.05, 0.950, 0.9950),
    ("g005_f0970_m09995", 0.05, 0.970, 0.9995),
    ("g005_f0990_m10000", 0.05, 0.990, 1.0000),
    ("g012_f0900_m09900", 0.12, 0.900, 0.9900),
    ("g012_f0950_m09950", 0.12, 0.950, 0.9950),
    ("g012_f0970_m09995", 0.12, 0.970, 0.9995),
    ("g012_f0990_m10000", 0.12, 0.990, 1.0000),
    ("g025_f0950_m09950", 0.25, 0.950, 0.9950),
    ("g025_f0970_m09995", 0.25, 0.970, 0.9995),
    ("g025_f0990_m10000", 0.25, 0.990, 1.0000),
    ("g050_f0970_m09995", 0.50, 0.970, 0.9995),
)
