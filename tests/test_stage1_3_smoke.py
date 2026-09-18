from __future__ import annotations

import torch

from etrcm.stage1_3.baselines import MODEL_NAMES, build_model
from etrcm.stage1_3.events import context_event, evidence_event, noise_event
from etrcm.stage1_3.model import Stage13Config


def test_all_stage1_3_baselines_execute_interleaved_stream() -> None:
    config = Stage13Config()
    keys = torch.tensor([1, 2])
    values = torch.tensor([3, 4])
    for name in MODEL_NAMES:
        current = build_model(name, config)
        state = current.initial_state(2)
        state, _ = current.step(state, evidence_event(keys, values))
        state, _ = current.step(state, None)
        state, output = current.step(state, noise_event(values, keys), threshold=0.5)
        state, _ = current.step(state, context_event(keys), threshold=0.5)
        assert state.H.shape == (2, config.latent_slots, config.hidden_dim)
        assert output["expression_score"].shape == (2,)
        assert current.trainable_parameters() > 0
