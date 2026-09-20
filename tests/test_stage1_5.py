"""Stage 1.5 intervention, provenance, evidence and precision invariants."""

from __future__ import annotations

from dataclasses import replace
import subprocess
from pathlib import Path

import pytest
import torch

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_3.events import self_output_event
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_4.world import generate_world
from etrcm.stage1_5.interventions import CHANNELS, finite_read_intervention, swap_components
from etrcm.stage1_5.capacity import evaluate_associative_capacity, evaluate_capacity_prediction
from etrcm.stage1_5.evaluation import (
    evaluate_anatomy, evaluate_oracle, evaluate_perturbations,
    evaluate_read_mediation, evaluate_stability,
)
from etrcm.stage1_5.model import AnatomicalETRCM
from etrcm.stage1_5.precision import evaluate_precision
from etrcm.stage1_5.probes import evaluate_observability, evaluate_timescales
from etrcm.stage1_5.routing import (
    build_historical_bank, closed_loop_oracle_read, static_oracle_read,
)


@pytest.fixture
def model() -> AnatomicalETRCM:
    torch.manual_seed(1515)
    return AnatomicalETRCM(Stage14Config(), "B5_separate").eval()


def _states(model: AnatomicalETRCM) -> tuple[LearnedState, LearnedState]:
    first = model.initial_state(2)
    return first, LearnedState(
        first.H + 1.0, first.F + 2.0, first.M + 3.0, 7, 11,
    )


@pytest.mark.parametrize("channels", CHANNELS)
def test_exact_swap_isolation(model: AnatomicalETRCM, channels: str) -> None:
    reference, donor = _states(model)
    swapped = swap_components(reference, donor, channels)
    for component in "HFM":
        expected = getattr(donor if component in channels else reference, component)
        assert torch.equal(getattr(swapped, component), expected)
    assert swapped.tau == reference.tau
    assert swapped.external_time == reference.external_time


def test_read_restore_changes_read_not_stored_memory(model: AnatomicalETRCM) -> None:
    base, donor = _states(model)
    swapped = swap_components(base, donor, "M")
    _, control_diag = model.step(base, None)
    next_state, output = model.step_with_read(
        swapped, None, slow_read_override=control_diag["r_M"],
    )
    assert torch.equal(output["r_M"], control_diag["r_M"])
    expected_stored_read = torch.einsum("bvk,bk->bv", swapped.M, output["q_M"])
    assert torch.equal(output["memory_r_M"], expected_stored_read)
    assert torch.allclose(next_state.M, model.config.rho_slow * (
        swapped.M + output["transfer"]
    ))
    assert not torch.equal(swapped.M, base.M)


def test_unclamped_read_interface_matches_frozen_transition(model: AnatomicalETRCM) -> None:
    world = generate_world("long_gap_relation", batch=4, length=16, seed=1520)
    state = model.initial_state(4)
    state, _ = model.step(state, world.events[0])
    qf, qm = model._queries(state.H)
    raw_m = torch.einsum("bvk,bk->bv", state.M, qm)
    standard, _ = model.step(state.clone(), None)
    clamped, _ = model.step_with_read(state.clone(), None, slow_read_override=raw_m)
    for component in "HFM":
        torch.testing.assert_close(getattr(standard, component), getattr(clamped, component),
                                   atol=1e-7, rtol=1e-6)


def test_oracle_is_historical_only_and_distinct_from_future_label(model: AnatomicalETRCM) -> None:
    world = generate_world("long_gap_relation", batch=8, length=16, seed=1515, gap=128)
    _, bank = build_historical_bank(model, world.events, history_length=4)
    read, source = static_oracle_read(bank, world.events[world.bridge_index])
    assert read.shape == (8, model.config.value_dim)
    assert source.max() < 4
    assert torch.equal(bank.key_ids[source, torch.arange(8)], world.events[world.bridge_index].value_id)
    # Mutating the held-out target tensor cannot change the read-bank API.
    fake = replace(world, targets=torch.zeros_like(world.targets), target_key=torch.full_like(world.target_key, -1))
    _, fake_bank = build_historical_bank(model, fake.events, history_length=4)
    fake_read, fake_source = static_oracle_read(fake_bank, fake.events[fake.bridge_index])
    assert torch.equal(read, fake_read)
    assert torch.equal(source, fake_source)


def test_closed_loop_oracle_recomputes_from_current_h(model: AnatomicalETRCM) -> None:
    world = generate_world("long_gap_relation", batch=8, length=16, seed=1516, gap=128)
    state, bank = build_historical_bank(model, world.events, history_length=4)
    bridge = world.events[world.bridge_index]
    first, _ = closed_loop_oracle_read(model, state.H, bank, bridge)
    # Re-evaluation consumes H, not a cached first-tick read.
    second, _ = closed_loop_oracle_read(model, state.H + 10, bank, bridge)
    assert first.shape == second.shape
    assert not torch.equal(first, second) or bool((bank.valid & bank.key_ids.eq(bridge.value_id[None])).sum(0).eq(1).all())


def test_retrieval_advantage_is_finite_loss_difference() -> None:
    target = torch.tensor([0])
    candidates = torch.tensor([[[0.0]], [[1.0]]])

    def rollout(read: torch.Tensor | None) -> torch.Tensor:
        strength = torch.zeros(1) if read is None else read[:, 0]
        return torch.stack([strength, torch.zeros_like(strength)], -1)

    advantage = finite_read_intervention(rollout, target, candidates)
    assert advantage.shape == (2, 1)
    assert advantage[0, 0] == pytest.approx(0.0)
    assert advantage[1, 0] == pytest.approx(
        float(torch.nn.functional.cross_entropy(rollout(None), target)
              - torch.nn.functional.cross_entropy(rollout(candidates[1]), target))
    )


def test_no_target_query_or_external_write_on_null_self_output(model: AnatomicalETRCM) -> None:
    world = generate_world("long_gap_relation", batch=4, length=16, seed=19)
    assert all(not hasattr(event, "target") for event in world.events)
    state = model.initial_state(4)
    null_state, null_diag = model.step_with_read(state, None, slow_read_override=torch.zeros(4, 16))
    assert not bool(null_diag["external_write_flag"].any())
    assert null_diag["external_update"].abs().sum() == 0
    assert null_state.external_time == state.external_time
    event = self_output_event(torch.ones(4, dtype=torch.long))
    self_state, self_diag = model.step_with_read(state, event, slow_read_override=torch.zeros(4, 16))
    assert not bool(self_diag["external_write_flag"].any())
    assert self_diag["external_update"].abs().sum() == 0
    assert self_state.external_time == state.external_time


def test_conservation_before_decay_and_fp32_analytic(model: AnatomicalETRCM) -> None:
    world = generate_world("latent_transition", batch=4, length=16, seed=20)
    state = model.initial_state(4)
    state, _ = model.step(state, world.events[0])
    next_state, diag = model.step_with_read(state, None, slow_read_override=torch.zeros(4, 16))
    assert torch.max(diag["conservation_error"]) < 1e-5
    torch.testing.assert_close(
        next_state.F / model.config.rho_fast + next_state.M / model.config.rho_slow,
        state.F + state.M, atol=1e-5, rtol=1e-5,
    )
    gamma, steps = 0.1, 20
    fast = torch.tensor(1.0, dtype=torch.float32)
    slow = torch.tensor(0.0, dtype=torch.float32)
    for _ in range(steps):
        delta = gamma * fast
        fast -= delta
        slow += delta
    assert float(fast) == pytest.approx((1 - gamma) ** steps, rel=1e-6)
    assert float(fast + slow) == pytest.approx(1.0, abs=1e-6)


def test_bf16_vs_shadow_small_update_diagnostic() -> None:
    steps, gamma = 1000, 1e-5
    bf_fast = torch.tensor(1.0, dtype=torch.bfloat16)
    bf_slow = torch.tensor(0.0, dtype=torch.bfloat16)
    sh_fast, sh_slow = torch.tensor(1.0), torch.tensor(0.0)
    for _ in range(steps):
        bf_delta = torch.tensor(gamma, dtype=torch.bfloat16) * bf_fast
        bf_fast = bf_fast - bf_delta
        bf_slow = bf_slow + bf_delta
        sh_delta = gamma * sh_fast
        sh_fast -= sh_delta
        sh_slow += sh_delta
        _ = sh_slow.to(torch.bfloat16)  # consumption is rounded; storage is not
    assert abs(float(sh_fast + sh_slow) - 1) < 1e-5
    assert float(sh_fast) == pytest.approx((1 - gamma) ** steps, rel=1e-4)
    assert float(bf_fast) == 1.0  # the tiny subtraction was rounded away
    assert float(bf_slow) < 0.5 * float(sh_slow)  # accumulation also stalled


def test_architecture_and_routing_smoke(model: AnatomicalETRCM) -> None:
    device = torch.device("cpu")
    assert evaluate_stability(model, seed=1, run_id="smoke", device=device, batch=2, ticks=2)
    assert evaluate_perturbations(
        model, seed=1, run_id="smoke", device=device, batch=2, checkpoints=(0, 1, 2)
    )
    assert evaluate_anatomy(model, seed=1, run_id="smoke", device=device, batch=4)
    assert evaluate_read_mediation(model, seed=1, run_id="smoke", device=device, batch=4)
    oracle_rows = evaluate_oracle(model, seed=1, run_id="smoke", device=device,
                                  gaps=(8,), batch=4)
    assert {row["intervention_condition"] for row in oracle_rows} == {
        "learned", "zero", "no_read", "random", "shuffled", "oracle_static", "oracle_closed_loop"
    }


def test_capacity_probe_precision_smoke(model: AnatomicalETRCM) -> None:
    device = torch.device("cpu")
    assert evaluate_associative_capacity(
        model, seed=1, run_id="smoke", device=device,
        counts=(1, 2), distractor_grid=(0, 2),
    )
    assert evaluate_capacity_prediction(
        model, seed=1, run_id="smoke", device=device,
        batch=2, max_gap=8, distractor_grid=(0, 8),
    )
    assert evaluate_timescales(
        model, seed=1, run_id="smoke", device=device,
        train_episodes=8, test_episodes=4, lags=(1, 2),
    )
    assert evaluate_observability(
        model, seed=1, run_id="smoke", device=device,
        train_per_family=8, test_per_family=4,
    )
    assert len(evaluate_precision(run_id="smoke")) == 12


def test_prior_frozen_artifacts_have_no_tracked_edits() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "diff", "--name-only", "13671106e0602326e9fd84de9493568e24ff003f", "--",
         "README.md", "configs/stage1_4_v1a1.yaml", "reports/STAGE1_4_FINAL_REPORT.md",
         "results/stage1_4", "artifacts/stage1_4_all_assets.sha256"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert not result.stdout.strip()
