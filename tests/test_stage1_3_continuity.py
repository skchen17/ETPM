from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from etrcm.stage1_3.events import (
    ContinuousEvent,
    Stage13EventKind,
    evidence_event,
    self_output_event,
)
from etrcm.stage1_3.model import ContinuousETRCM, Stage13Config


ROOT = Path(__file__).resolve().parents[1]


def model(mode: str = "arbitration") -> ContinuousETRCM:
    torch.manual_seed(13)
    return ContinuousETRCM(Stage13Config(), read_mode=mode)


def test_null_and_self_output_never_external_write() -> None:
    current = model()
    state = current.initial_state(3)
    state, null_output = current.step(state, None)
    assert not null_output["external_write_flag"].any()
    before_f, before_m = state.F.clone(), state.M.clone()
    event = self_output_event(torch.tensor([1, 2, 3]))
    _, output = current.step(state, event)
    assert not output["external_write_flag"].any()
    assert torch.count_nonzero(output["external_update"]) == 0
    # A SELF_OUTPUT transition may consolidate/decay, but never calls evidence write.
    assert torch.equal(before_f, state.F) and torch.equal(before_m, state.M)


def test_self_output_feedback_changes_h_only_and_does_not_reset() -> None:
    current = model()
    state = current.initial_state(2)
    state.F.normal_(); state.M.normal_(); state.H.normal_()
    result = current.apply_self_output_feedback(state, torch.tensor([4, 5]))
    assert not torch.equal(result.H, state.H)
    assert torch.equal(result.F, state.F)
    assert torch.equal(result.M, state.M)
    assert result.tau == state.tau and result.external_time == state.external_time


def test_emit_does_not_reset_state_and_more_ticks_are_allowed() -> None:
    current = model()
    state = current.initial_state(2)
    state.F.normal_(); state.M.normal_(); state.H.normal_()
    emitted_state, output = current.step(state.clone(), None, threshold=0.0)
    silent_state, _ = current.step(state.clone(), None, threshold=1.1)
    assert output["emitted"].all()
    assert torch.allclose(emitted_state.F, silent_state.F)
    assert torch.allclose(emitted_state.M, silent_state.M)
    assert not torch.allclose(emitted_state.H, current.initial_state(2).H)
    advanced, _ = current.step(emitted_state, None, threshold=1.1)
    assert advanced.tau == emitted_state.tau + 1


def test_external_input_can_follow_self_output_at_arbitrary_tick() -> None:
    current = model()
    state = current.initial_state(2)
    for _ in range(5):
        state, _ = current.step(state, None)
    state = current.apply_self_output_feedback(state, torch.tensor([1, 2]))
    before = state.external_time
    state, output = current.step(state, evidence_event(torch.tensor([3, 4]), torch.tensor([5, 6])))
    assert output["external_write_flag"].all()
    assert torch.linalg.vector_norm(output["external_update"]) > 0
    assert state.external_time == before + 1


def test_conservation_before_decay_and_arbitration_range() -> None:
    current = model()
    state = current.initial_state(4)
    state.F.normal_(); state.M.normal_()
    _, output = current.step(state, None)
    assert torch.max(output["conservation_error"]).item() < 1e-5
    assert torch.all((output["arbitration_gate"] >= 0) & (output["arbitration_gate"] <= 1))


def test_f_only_and_m_only_are_exact_reads() -> None:
    base = model("f_only")
    other = model("m_only")
    other.load_state_dict(base.state_dict())
    state = base.initial_state(3)
    state.H.normal_(); state.F.normal_(); state.M.normal_()
    q = base.query(state.H)
    expected_f = torch.einsum("bvk,bk->bv", state.F, q)
    expected_m = torch.einsum("bvk,bk->bv", state.M, q)
    _, f_output = base.step(state.clone(), None)
    _, m_output = other.step(state.clone(), None)
    assert torch.allclose(f_output["read"], expected_f, atol=1e-6)
    assert torch.allclose(m_output["read"], expected_m, atol=1e-6)


def test_threshold_is_deterministic_for_fixed_state() -> None:
    current = model()
    state = current.initial_state(4)
    first = current.expression(state)
    second = current.expression(state)
    assert torch.equal(first["expression_score"], second["expression_score"])
    assert torch.equal(first["content_logits"], second["content_logits"])


def test_event_schema_has_no_query_answer_solved_or_halting_field() -> None:
    forbidden = {"answer", "target", "label", "solved", "halt", "stop", "history"}
    assert not (ContinuousEvent.ALLOWED_FIELDS & forbidden)
    current = model()
    names = {name.lower() for name, _ in current.named_modules()}
    assert not any("halt" in name or "solved" in name or "stop_thinking" in name for name in names)


def test_formal_threshold_protocol_is_hash_frozen() -> None:
    freeze = json.loads((ROOT / "artifacts/stage1_3_protocol.freeze.json").read_text())
    for relative, expected in freeze["files"].items():
        if relative == "reports/STAGE1_3_AMENDMENTS.md":
            continue
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
    amendment = json.loads((ROOT / "artifacts/stage1_3_amendment2.freeze.json").read_text())
    for relative, expected in amendment["files"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
    config = yaml.safe_load((ROOT / "configs/stage1_3.yaml").read_text())
    assert len(config["threshold_selection"]["candidates"]) == 16
    assert len(config["training"]["formal_seeds"]) == 8
