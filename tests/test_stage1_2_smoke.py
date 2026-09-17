import torch

from etrcm.stage1_1.model import LearnedModelConfig
from etrcm.stage1_1.training import TrainSpec
from etrcm.stage1_2.evaluation import (
    autonomous_memory_selection,
    functional_query_intervention,
    no_self_evidence,
    sequential_computation,
    storage_vs_use,
)
from etrcm.stage1_2.model import Stage12ETRCM
from etrcm.stage1_2.training import (
    train_autonomous_model,
    train_no_evidence_model,
    train_sequential_model,
)


def model():
    config = LearnedModelConfig(
        hidden_dim=16, latent_slots=2, symbol_count=16, key_dim=8,
        value_dim=8, event_type_dim=4,
    )
    return Stage12ETRCM(config)


def spec():
    return TrainSpec(1, 4, 1e-3, 0.0, 1.0, 0.0, 16)


def test_specialized_training_and_evaluation_smoke():
    device = torch.device("cpu")
    autonomous = model()
    assert train_autonomous_model(autonomous, spec(), seed=1, device=device)
    assert autonomous_memory_selection(
        autonomous, seed=1, tick_counts=[0, 1], episodes=2,
        symbol_count=16, device=device,
    )
    sequential = model()
    assert train_sequential_model(
        sequential, spec(), seed=2, device=device, train_max_length=3,
    )
    assert sequential_computation(
        sequential, seed=2, path_lengths=[1, 3, 4], tick_counts=[0, 1],
        episodes=2, train_max_length=3, symbol_count=16, device=device,
    )
    unknown = model()
    assert train_no_evidence_model(unknown, spec(), seed=3, device=device)
    assert no_self_evidence(
        unknown, seed=3, tick_counts=[0, 1], episodes=2,
        symbol_count=16, device=device,
    )
    memory = model()
    assert functional_query_intervention(
        memory, seed=4, episodes=2, interference=2,
        symbol_count=16, device=device,
    )
    assert storage_vs_use(
        memory, seed=4, episodes=2, interference=2,
        symbol_count=16, device=device,
    )
