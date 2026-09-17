"""Stage-1.2 baseline factory; state accounting remains architecture specific."""

from etrcm.stage1_1.baselines import (
    NoMemoryRecurrentMLP,
    TrainedGRU,
    TwoHeadSinglePersistent,
)
from etrcm.stage1_1.model import LearnedModelConfig
from etrcm.stage1_2.model import Stage12ETRCM


FORMAL_MODELS = (
    "B1_gru",
    "B2_single_persistent",
    "B3_uniform",
    "B5_no_idle",
    "B6_full",
)


def build_stage12_baseline(name: str, config: LearnedModelConfig):
    if name == "B0_no_memory_mlp":
        return NoMemoryRecurrentMLP(config)
    if name == "B1_gru":
        return TrainedGRU(config)
    if name == "B2_single_persistent":
        return TwoHeadSinglePersistent(config)
    if name == "B3_uniform":
        return Stage12ETRCM(config, consolidation="uniform")
    if name == "B5_no_idle":
        return Stage12ETRCM(config, idle_updates=False)
    if name == "B6_full":
        return Stage12ETRCM(config)
    raise KeyError(name)

