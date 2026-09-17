import torch

from etrcm.stage1_1.data import fact_event, query_event
from etrcm.stage1_1.model import LearnedModelConfig
from etrcm.stage1_2.evaluation import selective_scaling
from etrcm.stage1_2.model import Stage12ETRCM


def config():
    return LearnedModelConfig(
        hidden_dim=16, latent_slots=2, symbol_count=16, key_dim=8,
        value_dim=8, event_type_dim=4,
    )


def test_sequential_graph_history_is_removed_from_active_state():
    model = Stage12ETRCM(config())
    first = model.initial_state(2)
    second = model.initial_state(2)
    first, _ = model.step(first, fact_event(torch.tensor([1, 2]), torch.tensor([3, 4])))
    second, _ = model.step(second, fact_event(torch.tensor([5, 6]), torch.tensor([7, 8])))
    first, second = model.reset_active(first), model.reset_active(second)
    assert torch.allclose(first.H, second.H)
    assert not torch.allclose(first.F + first.M, second.F + second.M)
    event = query_event(torch.tensor([1, 2]), target=torch.tensor([9, 10]))
    first, output = model.step(first, event)
    assert output["query"].shape == (2, config().key_dim)
    assert output["read"].shape == (2, config().value_dim)


def test_null_step_has_no_external_evidence_write():
    model = Stage12ETRCM(config())
    state = model.initial_state(2)
    state, output = model.step(state, None)
    assert not output["event_written"]
    assert torch.count_nonzero(output["external_update"]) == 0


def test_lesion_records_use_explicit_pre_and_post_names():
    model = Stage12ETRCM(config())
    rows = selective_scaling(
        model, model_name="B6_full", seed=2, distractor_counts=[0],
        useful_count=2, episodes=2, symbol_count=16, device=torch.device("cpu"),
    )
    lesion = [row for row in rows if row["experiment"].endswith("lesion")]
    assert lesion
    required = {
        "pre_lesion_slow_retention", "post_lesion_slow_retention",
        "pre_lesion_accuracy", "post_lesion_accuracy",
    }
    assert required.issubset(lesion[0])
    assert lesion[0]["post_lesion_slow_retention"] == 0.0


def test_knowable_and_unknowable_can_use_identical_event_kinds():
    query_keys = torch.tensor([1, 1])
    write_keys = torch.tensor([1, 2])
    values = torch.tensor([3, 3])
    write = fact_event(write_keys, values)
    query = query_event(query_keys)
    assert write.kind[0] == write.kind[1]
    assert query.kind[0] == query.kind[1]
    assert write.scalars[0].equal(write.scalars[1])
    assert query.scalars[0].equal(query.scalars[1])

