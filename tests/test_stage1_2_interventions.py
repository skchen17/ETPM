import torch

from etrcm.stage1_1.data import fact_event, query_event
from etrcm.stage1_1.events import EventKind
from etrcm.stage1_1.model import LearnedModelConfig
from etrcm.stage1_2.baselines import FORMAL_MODELS, build_stage12_baseline
from etrcm.stage1_2.data import goal_event
from etrcm.stage1_2.interventions import decompose_query
from etrcm.stage1_2.model import Stage12ETRCM


def config():
    return LearnedModelConfig(
        hidden_dim=16, latent_slots=2, symbol_count=16, key_dim=8,
        value_dim=8, event_type_dim=4, rho_fast=1.0, rho_slow=1.0,
    )


def test_query_projection_decomposition_and_target_removal():
    torch.manual_seed(1)
    query = torch.randn(5, 8)
    target = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    non_target = torch.nn.functional.normalize(torch.randn(5, 3, 8), dim=-1)
    result = decompose_query(query, target, non_target)
    assert torch.allclose(result.parallel + result.perpendicular, query, atol=1e-6)
    assert torch.allclose((result.perpendicular * target).sum(-1), torch.zeros(5), atol=1e-6)
    assert torch.allclose(result.target_zero, result.perpendicular)
    assert torch.allclose(result.signflip, -query)
    assert torch.allclose(result.squared_projection, (query * target).sum(-1).pow(2))


def test_forced_query_preserves_conservation_and_signflip():
    model = Stage12ETRCM(config())
    state = model.initial_state(3)
    state, _ = model.step(state, fact_event(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])))
    event = query_event(torch.tensor([1, 2, 3]))
    query, _, _ = model.proposed_query(state, event)
    _, output = model.step_with_forced_query(state, event, -query)
    assert torch.allclose(output["query"], -query)
    assert output["conservation_error"].max() < 1e-6


def test_goal_and_query_events_contain_no_answer_or_retrieval_keys():
    goal = goal_event(4, torch.tensor([0, 1, 2, 0]), torch.device("cpu"))
    assert goal.kind.eq(int(EventKind.TASK_CUE)).all()
    assert goal.key_id.eq(-1).all() and goal.value_id.eq(-1).all() and goal.aux_id.eq(-1).all()
    query = query_event(torch.tensor([1, 2, 3, 4]))
    assert query.value_id.eq(-1).all() and not query.write_mask.any()


def test_state_bytes_and_parameter_accounting_are_explicit():
    models = {name: build_stage12_baseline(name, config()) for name in FORMAL_MODELS}
    assert models["B2_single_persistent"].persistent_state_bytes() == models["B6_full"].persistent_state_bytes()
    for model in models.values():
        assert model.persistent_state_bytes() > 0
        assert model.trainable_parameters() > 0

