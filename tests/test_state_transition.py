import torch

from etrcm.model import ETRCM, ETRCMConfig


def test_null_event_advances_internal_not_external_time() -> None:
    torch.manual_seed(10)
    config = ETRCMConfig(gamma=0.0, rho_fast=1.0, rho_slow=1.0)
    model = ETRCM(config)
    state = model.initial_state(dtype=torch.float32)
    before_h = state.H.clone()
    before_f = state.F.clone()
    before_m = state.M.clone()
    state, output = model.step(state, torch.zeros(config.event_dim))
    assert state.tau == 1
    assert state.external_time == 0
    assert output["event_written"] is False
    assert torch.equal(state.F, before_f)
    assert torch.equal(state.M, before_m)
    assert not torch.equal(state.H, before_h)


def test_external_event_writes_once() -> None:
    torch.manual_seed(11)
    config = ETRCMConfig(gamma=0.0, rho_fast=1.0, rho_slow=1.0)
    model = ETRCM(config)
    state = model.initial_state()
    event = torch.randn(config.event_dim)
    state, output = model.step(state, event)
    assert state.external_time == 1
    assert output["event_written"] is True
    written = state.F.clone()
    state, output = model.step(state, torch.zeros(config.event_dim))
    assert state.external_time == 1
    assert output["event_written"] is False
    assert torch.equal(state.F, written)


def test_dict_api_roundtrip() -> None:
    model = ETRCM()
    state = model.initial_state().as_dict()
    updated, output = model.step(state, None)
    assert isinstance(updated, dict)
    assert updated["tau"] == 1
    assert output["conservation_error"].item() < 1e-6

