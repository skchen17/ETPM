"""Stage 2C protocol invariants; no historical frozen result is rewritten."""

import random

import torch

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2c.model import BehavioralModel
from etrcm.stage2c.rollout import play_experience, probe_policy
from etrcm.stage2c.world import (
    ACTION, ABSTRACT, Experience, consequence, paired_histories, surface,
)


CFG = Stage14Config(hidden_dim=16, latent_slots=1, symbol_count=24,
                    key_dim=8, value_dim=8, event_type_dim=8)


def test_frozen_lifetime_parameters_bit_exact():
    torch.manual_seed(7)
    model = BehavioralModel(CFG)
    frozen = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
    digest = model.parameter_digest()
    rng = random.Random(17)
    state = model.initial_state(2)
    model.eval()
    with torch.no_grad():
        for t in range(8):
            experiences = [Experience(*surface(rng, "train"), t % 2,
                                      consequence(rng, z, t % 2)) for z in (0, 1)]
            state, _, _, _ = play_experience(model, state, experiences)
        for _ in range(12):
            state, _ = model.step(state, None)
        p, *_ = probe_policy(model, state, [surface(rng, "novel")] * 2)
    assert torch.isfinite(p).all()
    assert digest == model.parameter_digest()
    for name, tensor in model.named_parameters():
        assert torch.equal(frozen[name], tensor)


def test_paired_histories_have_only_consequence_difference():
    a, b = paired_histories(91, 32)
    assert len(a) == len(b) == 32
    assert [x.action for x in a].count(0) == 16
    for n in (2,4,8,16,32):
        assert [x.action for x in a[:n]].count(0) == n//2
    for x, y in zip(a, b):
        assert (x.color, x.shape, x.nuisance, x.action) == (
            y.color, y.shape, y.nuisance, y.action)
    assert ABSTRACT not in ACTION


def test_probe_surface_unseen_combination_and_no_latent_cue():
    rng = random.Random(2)
    train = {(surface(rng, "train")[:2]) for _ in range(200)}
    novel = {(surface(rng, "novel")[:2]) for _ in range(200)}
    assert train.isdisjoint(novel)
    assert {x[0] for x in train} == {x[0] for x in novel}
    assert {x[1] for x in train} == {x[1] for x in novel}


def test_read_clamp_preserves_memory_storage():
    torch.manual_seed(11)
    model = BehavioralModel(CFG)
    state = model.initial_state(2)
    rng = random.Random(3)
    events = [Experience(*surface(rng, "train"), 0, consequence(rng, z, 0))
              for z in (0, 1)]
    state, _, _, _ = play_experience(model, state, events)
    F_before, M_before=state.F.clone(),state.M.clone()
    with torch.no_grad():
        clamped, _ = model.step(state, None, read_clamp="FM")
    assert torch.equal(state.F, F_before)
    assert torch.equal(state.M, M_before)
    assert torch.count_nonzero(clamped.F) > 0
    assert torch.count_nonzero(clamped.M) > 0


def test_gamma_zero_transfer_is_zero():
    torch.manual_seed(4)
    model = BehavioralModel(CFG, "gamma_zero")
    state = model.initial_state(2)
    event = [Experience(8, 14, 20, 0, 4), Experience(9, 15, 21, 1, 5)]
    with torch.no_grad():
        state, _, _, diags = play_experience(model, state, event)
    assert all(torch.count_nonzero(d["transfer"]) == 0 for d in diags)


def test_full_state_swap_reverses_paired_probe_distribution():
    torch.manual_seed(33)
    model=BehavioralModel(CFG)
    model.eval()
    a,b=paired_histories(4,4)
    state=model.initial_state(2)
    with torch.no_grad():
        for x,y in zip(a,b):
            state,_,_,_=play_experience(model,state,[x,y])
        features=[surface(random.Random(99),"novel")]*2
        original,*_=probe_policy(model,state,features)
        swapped=LearnedState(state.H.flip(0),state.F.flip(0),state.M.flip(0),
                             state.tau,state.external_time)
        counterfactual,*_=probe_policy(model,swapped,features)
    assert torch.allclose(original.flip(0),counterfactual,atol=1e-6)
