"""Restricted-input predictors for the marginal-shortcut audit."""

from __future__ import annotations

import torch
from torch import nn


class MarginalPredictor:
    """Only observed outcome identities are accepted, never action or state."""

    def __init__(self):
        self.probability=torch.full((4,),0.25,dtype=torch.float64)

    def fit(self,outcomes: torch.Tensor):
        count=torch.bincount(outcomes.cpu().long(),minlength=4).double()+1
        self.probability=count/count.sum()
        return self

    def predict(self,batch: int) -> torch.Tensor:
        return self.probability.expand(batch,4)


class ActionOnlyPredictor:
    """Accepts an action ID, never a history or recurrent state."""

    def __init__(self):
        self.probability=torch.full((2,4),0.25,dtype=torch.float64)

    def fit(self,actions: torch.Tensor,outcomes: torch.Tensor):
        a=actions.cpu().long();y=outcomes.cpu().long()
        count=torch.ones(2,4,dtype=torch.float64)
        for action in (0,1):count[action]+=torch.bincount(y[a==action],minlength=4).double()
        self.probability=count/count.sum(-1,keepdim=True)
        return self

    def predict(self,actions: torch.Tensor) -> torch.Tensor:
        return self.probability[actions.cpu().long()]


class HistoryOnlyPredictor(nn.Module):
    """Post-hoc H readout with no candidate-action argument."""

    def __init__(self,hidden_dim: int):
        super().__init__()
        self.head=nn.Linear(hidden_dim,4)

    def forward(self,history_state: torch.Tensor) -> torch.Tensor:
        return self.head(history_state)
