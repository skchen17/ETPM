from pathlib import Path

import torch
import yaml

from etrcm.baselines import BASELINE_REGISTRY, GRUBaseline, NoMemoryMLP
from etrcm.toys.suite import run_named_toy


def config():
    path = Path(__file__).parents[1] / "configs" / "toy_default.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_all_toy_generators_smoke() -> None:
    cfg = config()
    cfg["experiments"]["interference_steps"] = 2
    cfg["training"]["toy3_steps"] = 3
    for index in range(1, 10):
        records = run_named_toy(str(index), cfg, [0])
        assert records
        assert all(record["toy"] for record in records)


def test_baseline_registry_and_shapes() -> None:
    assert len(BASELINE_REGISTRY) == 7
    mlp = NoMemoryMLP(8, 16, 4)
    gru = GRUBaseline(8, 16, 4)
    event = torch.randn(2, 8)
    assert mlp(event).shape == (2, 4)
    hidden, output = gru.step(torch.zeros(2, 16), event)
    assert hidden.shape == (2, 16)
    assert output.shape == (2, 4)

