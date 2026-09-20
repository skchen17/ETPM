"""Exploratory storage-precision regression, not a primary gate."""

from __future__ import annotations

import torch


def evaluate_precision(*, run_id: str, updates: int = 1000,
                       gamma: float = 1e-5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    analytic = (1.0 - gamma) ** updates
    for mode in ("FP32", "BF16", "FP32_shadow_BF16_consumption"):
        dtype = torch.bfloat16 if mode == "BF16" else torch.float32
        fast = torch.tensor(1.0, dtype=dtype)
        slow = torch.tensor(0.0, dtype=dtype)
        for step in range(1, updates + 1):
            delta = gamma * fast
            fast = fast - delta
            slow = slow + delta
            consumed_fast = fast.to(torch.bfloat16) if mode.endswith("consumption") else fast
            if step in (1, 10, 100, updates):
                rows.append({
                    "run_id": run_id, "experiment": "precision_regression",
                    "precision_mode": mode, "step": step, "gamma": gamma,
                    "F_storage": float(fast), "M_storage": float(slow),
                    "F_consumption": float(consumed_fast),
                    "total_storage": float(fast + slow),
                    "analytic_F": (1.0 - gamma) ** step,
                    "analytic_F_error": abs(float(fast) - (1.0 - gamma) ** step),
                    "primary_gate": False,
                })
        assert analytic > 0
    return rows
