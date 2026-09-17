from dataclasses import fields
import inspect

import torch

from etrcm.stage1_1.data import fact_event, query_event
from etrcm.stage1_1.events import StructuredEvent
from etrcm.stage1_1.model import LearnedETRCM, LearnedModelConfig


def config():
    return LearnedModelConfig(
        hidden_dim=16,
        latent_slots=2,
        symbol_count=16,
        key_dim=8,
        value_dim=8,
        event_type_dim=4,
    )


def test_event_schema_has_no_history_or_future_fields():
    names = {field.name for field in fields(StructuredEvent)}
    assert names == StructuredEvent.ALLOWED_FIELDS
    forbidden = {"history", "episode", "future", "importance", "reuse_count", "target_value"}
    assert not names.intersection(forbidden)


def test_step_accepts_only_state_and_current_event():
    names = list(inspect.signature(LearnedETRCM.step).parameters)
    assert names == ["self", "state", "event"]


def test_query_cue_cannot_contain_answer_value():
    keys = torch.tensor([1, 2, 3])
    event = query_event(keys)
    assert event.write_mask.eq(False).all()
    assert event.value_id.eq(-1).all()


def test_memory_lesion_changes_retrieval_path_after_active_scrub():
    torch.manual_seed(4)
    model = LearnedETRCM(config())
    state = model.initial_state(4)
    state, _ = model.step(state, fact_event(torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])))
    state = model.reset_active(state)
    intact, output_intact = model.step(state, query_event(torch.tensor([1, 2, 3, 4])))
    lesioned, output_lesioned = model.step(model.memory_lesion(state), query_event(torch.tensor([1, 2, 3, 4])))
    assert not torch.allclose(output_intact["read"], output_lesioned["read"])
    assert intact.H.shape == lesioned.H.shape

