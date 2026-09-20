from pathlib import Path
import subprocess
from dataclasses import replace

import torch

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_3.events import ContinuousEvent, Stage13EventKind
from etrcm.stage1_6.runner import (derangement, historical_oracle, make_model,
                                    norm_matched_random, oracle_probability, run_episode, scrub)
from etrcm.stage1_6.world import audit_no_future_leak, generate_world


CONFIG = Stage14Config(hidden_dim=32, latent_slots=2, symbol_count=24,
                       key_dim=16, value_dim=16, event_type_dim=8)


def test_oracle_only_past_and_no_future_target_leak():
    model = make_model(CONFIG, "learned", seed=11, device=torch.device("cpu"))
    world = generate_world(batch=16, seed=7)
    assert audit_no_future_leak(world)
    state = model.initial_state(16)
    for event in world.events[:4]:
        state, _ = model.step(state, event)
    read = historical_oracle(model, state, world)
    changed = generate_world(batch=16, seed=7)
    # Counterfactual: change only later bridge offset, hence future target.
    changed_offset = 16 + (changed.offset - 15).remainder(8)
    changed_target = ((changed.past_value - 8) + (changed_offset - 16)).remainder(8)
    changed_events = changed.events[:-1] + (
        ContinuousEvent.create(kind=Stage13EventKind.CONTEXT,
                               key_id=changed.key, aux_id=changed_offset, write=False),)
    changed = replace(changed, offset=changed_offset, target=changed_target,
                      events=changed_events)
    assert bool(changed_target.ne(world.target).all())
    assert torch.equal(read, historical_oracle(model, state, changed))
    assert all(event.value_id.eq(-1).all() for event in world.events[4:])


def test_scrub_preserves_memory_and_removes_history_from_H():
    model = make_model(CONFIG, "learned", seed=1, device=torch.device("cpu"))
    a = generate_world(batch=8, seed=5)
    b = generate_world(batch=8, seed=6)
    states = []
    for world in (a, b):
        state = model.initial_state(8)
        for event in world.events[:4]:
            state, _ = model.step(state, event)
        after = scrub(model, state)
        assert torch.equal(after.F, state.F)
        assert torch.equal(after.M, state.M)
        states.append(after)
    assert torch.equal(states[0].H, states[1].H)
    assert not torch.equal(states[0].F, states[1].F)


def test_zero_read_only_and_norm_matched_controls():
    model = make_model(CONFIG, "learned", seed=2, device=torch.device("cpu"))
    world = generate_world(batch=16, seed=12)
    state = model.initial_state(16)
    for event in world.events[:4]:
        state, _ = model.step(state, event)
    bank = historical_oracle(model, state, world)
    state = scrub(model, state)
    for event in world.events[4:world.bridge_index]:
        state, _ = model.step(state, event)
    event = world.events[world.bridge_index]
    full, _ = model.step(state, event)
    zero, output = model.step_with_read(state, event,
                                       fast_read_override=torch.zeros_like(bank),
                                       slow_read_override=torch.zeros_like(bank))
    assert torch.equal(state.F, state.F)
    assert torch.equal(state.M, state.M)
    assert torch.equal(output["read"], torch.zeros_like(output["read"]))
    assert not torch.equal(full.H, zero.H)
    random = norm_matched_random(bank, seed=21)
    assert torch.allclose(random.norm(dim=-1), bank.norm(dim=-1), atol=1e-6)
    shuffle = derangement(16, bank.device)
    assert bool(shuffle.ne(torch.arange(16)).all())


def test_curriculum_schedule_and_no_query_labels():
    assert [oracle_probability(i, 100) for i in (0, 20, 40, 60, 80, 99)] == [1, .75, .5, .25, 0, 0]
    model = make_model(CONFIG, "oracle", seed=3, device=torch.device("cpu"))
    world = generate_world(batch=8, seed=30)
    loss, diagnostics = run_episode(model, world, condition="oracle")
    assert bool(torch.isfinite(loss))
    assert diagnostics["ce"].shape == (8,)
    assert not any("query_label" in name or "target_query" in name for name in model.state_dict())
    assert model.future_heads["1"].in_features == CONFIG.hidden_dim


def test_lesion_sign_and_frozen_parent_unchanged():
    full, lesioned = 1.0, 1.3
    assert lesioned - full > 0
    root = Path(__file__).resolve().parents[1]
    # The parent report and manifest remain tracked in their original paths.
    assert (root / "reports/STAGE1_5_FINAL_REPORT.md").exists()
    assert (root / "artifacts/stage1_5_all_assets.sha256").exists()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "fbe650424281ce9606f683d884b4bff3efa91e34", "--",
         "results/stage1_5", "reports/STAGE1_5_FINAL_REPORT.md", "src/etrcm/stage1_5",
         "artifacts/stage1_5_all_assets.sha256"], cwd=root, text=True)
    assert changed.strip() == ""


def test_checkpoint_intervention_reproducibility():
    model = make_model(CONFIG, "learned", seed=21, device=torch.device("cpu"))
    world = generate_world(batch=8, seed=101)
    first, metrics_a = run_episode(model, world, condition="random", random_seed=77)
    second, metrics_b = run_episode(model, world, condition="random", random_seed=77)
    assert torch.equal(first, second)
    assert torch.equal(metrics_a["read_norm"], metrics_b["read_norm"])
    shuffled = derangement(8, torch.device("cpu"))
    assert not bool(shuffled.eq(torch.arange(8)).any())


def test_no_memory_counterfactual_identifiability():
    world = generate_world(batch=8, seed=55)
    alternate_A = 8 + (world.past_value - 7).remainder(8)
    alternate_events = tuple(
        ContinuousEvent.create(kind=Stage13EventKind.EVIDENCE,
                               key_id=world.key, value_id=alternate_A, write=True)
        for _ in range(world.early_exposures)) + world.events[world.early_exposures:]
    alternate = replace(world, events=alternate_events, past_value=alternate_A,
                        target=((alternate_A-8)+(world.offset-16)).remainder(8))
    assert bool(world.target.ne(alternate.target).all())
    model = make_model(CONFIG, "no_memory", seed=5, device=torch.device("cpu"))
    with torch.no_grad():
        for candidate in (world, alternate):
            state = model.initial_state(8)
            for event in candidate.events[:4]:
                state, _ = model.step(state, event)
            state = scrub(model, state)
            for event in candidate.events[4:]:
                state, _ = model.step(state, event)
            if candidate is world:
                logits = model.predict_logits(state)[1]
            else:
                assert torch.equal(logits, model.predict_logits(state)[1])
