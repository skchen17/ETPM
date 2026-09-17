import torch

from etrcm.stage1_1.baselines import TRAINED_BASELINES, build_baseline
from etrcm.stage1_1.data import fact_event, query_event
from etrcm.stage1_1.model import LearnedModelConfig
from etrcm.stage1_1.training import TrainSpec, train_memory_model


def config():
    return LearnedModelConfig(
        hidden_dim=16,
        latent_slots=2,
        symbol_count=16,
        key_dim=8,
        value_dim=8,
        event_type_dim=4,
    )


def test_every_preregistered_baseline_steps():
    keys, values = torch.tensor([1, 2]), torch.tensor([3, 4])
    for name in TRAINED_BASELINES:
        model = build_baseline(name, config())
        state = model.initial_state(2)
        state, _ = model.step(state, fact_event(keys, values))
        state = model.reset_active(state)
        state, output = model.step(state, query_event(keys))
        assert output["symbol_logits"].shape == (2, 16)
        assert torch.isfinite(output["symbol_logits"]).all()


def test_two_step_training_smoke():
    model = build_baseline("B6_full", config())
    logs = train_memory_model(
        model,
        TrainSpec(2, 4, 1e-3, 0.0, 1.0, 0.0, 16),
        seed=7,
        device=torch.device("cpu"),
        log_every=1,
    )
    assert len(logs) == 2
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in logs)
