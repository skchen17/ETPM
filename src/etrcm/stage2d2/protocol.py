"""Preregistered Stage 2D.2 analysis and causal thresholds."""

from __future__ import annotations
from dataclasses import asdict,dataclass

CHECKPOINTS=(0,25,50,100,200,300,500,750,1000,1500)
RANKS=(1,2,4,8)
EPS_MULTIPLIERS=(.05,.10,.25,.50)
DIRECTIONS=128
PHASES=("early","mid","post","delay","probe")
ROTATION_LAMBDAS=(.25,.50,.75,1.0)


@dataclass(frozen=True)
class FrozenStage2D2:
    directions:int=DIRECTIONS
    ranks:tuple[int,...]=RANKS
    epsilon_multipliers:tuple[float,...]=EPS_MULTIPLIERS
    primary_rank:int=4
    primary_epsilon:float=.10
    rescue_required:int=6
    rescue_total:int=8
    necessity_required:int=6
    controllability_auc_min:float=.75
    controllability_stream_auc_min:float=.70
    controllability_streams_required:int=2
    intervention_norm_cap:float=1.25
    meaningful_margin:float=.01

    def to_dict(self): return asdict(self)

FROZEN_D2=FrozenStage2D2()
