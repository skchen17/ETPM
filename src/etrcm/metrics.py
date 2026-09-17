"""Common behavioral and state metrics."""

from __future__ import annotations

from typing import Any

import torch

from etrcm.state import ETState


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left.flatten(), right.flatten()) / denominator)


def retention_projection(read: torch.Tensor, target: torch.Tensor) -> float:
    denominator = torch.dot(target, target).clamp_min(1e-12)
    return float(torch.dot(read, target) / denominator)


def state_metrics(state: ETState) -> dict[str, float]:
    return {
        "h_norm": float(torch.linalg.vector_norm(state.H)),
        "f_norm": float(torch.linalg.vector_norm(state.F)),
        "m_norm": float(torch.linalg.vector_norm(state.M)),
        "a_norm": float(torch.linalg.vector_norm(state.F + state.M)),
    }


def empty_record(**values: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "toy": None,
        "condition": None,
        "seed": None,
        "step": None,
        "phase": None,
        "parameter": None,
        "parameter_value": None,
        "retention": None,
        "accuracy": None,
        "loss": None,
        "confidence": None,
        "calibration": None,
        "h_norm": None,
        "f_norm": None,
        "m_norm": None,
        "a_norm": None,
        "delta_norm": None,
        "transfer_fraction": None,
        "q_json": None,
        "query_similarity": None,
        "reuse_count": None,
        "cumulative_access": None,
        "h_change": None,
        "state_convergence": None,
        "idle_trajectory_length": None,
        "readout_drift": None,
        "memory_interference": None,
        "latency_steps": None,
        "compute_budget": None,
        "answer": None,
        "target": None,
        "notes": None,
    }
    record.update(values)
    return record

