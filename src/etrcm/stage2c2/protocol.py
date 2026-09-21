"""Frozen Stage 2C.2 decision and curriculum invariants.

These helpers do not imply that gated downstream experiments were run.
"""

ASSISTANCE_SCHEDULE = (1.0, 0.75, 0.5, 0.25, 0.0)


def final_evaluation_is_endogenous(probability: float) -> bool:
    """A formal curriculum score is allowed only at exactly zero assistance."""
    return probability == 0.0


def next_gate_status(prior_pass: bool, current_pass: bool | None) -> str:
    if not prior_pass:
        return "NOT_RUN_BY_GATE"
    if current_pass is None:
        return "NOT_RUN"
    return "PASS" if current_pass else "FAIL"
