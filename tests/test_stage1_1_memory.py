import torch

from etrcm.stage1_1.data import fact_event
from etrcm.stage1_1.model import LearnedETRCM, LearnedModelConfig


def tiny_config(**changes):
    values = dict(
        hidden_dim=16,
        latent_slots=2,
        symbol_count=16,
        key_dim=8,
        value_dim=8,
        event_type_dim=4,
        gamma=0.2,
        rho_fast=1.0,
        rho_slow=1.0,
        eta_external=0.5,
    )
    values.update(changes)
    return LearnedModelConfig(**values)


def test_batched_consolidation_conserves_total_before_decay():
    model = LearnedETRCM(tiny_config())
    state = model.initial_state(5)
    state.F.normal_()
    state.M.normal_()
    q = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    access = torch.rand(5)
    before = state.F + state.M
    fast, slow, _ = model._consolidate(state.F, state.M, q, access)
    assert torch.allclose(before, fast + slow, atol=1e-6)


def test_null_tick_never_invokes_external_write():
    model = LearnedETRCM(tiny_config())
    state = model.initial_state(3)
    keys = torch.tensor([1, 2, 3])
    values = torch.tensor([4, 5, 6])
    state, _ = model.step(state, fact_event(keys, values))
    external_time = state.external_time
    state, output = model.step(state, None)
    assert state.external_time == external_time
    assert not output["event_written"]
    assert torch.count_nonzero(output["external_update"]) == 0


def test_all_idle_ticks_share_parameters():
    model = LearnedETRCM(tiny_config())
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    state = model.initial_state(2)
    for _ in range(4):
        state, _ = model.step(state, None)
        assert {id(parameter) for parameter in model.parameters()} == parameter_ids

