"""Label-free temporary alignment objective used only if diagnostics authorize it."""

from __future__ import annotations
import torch
from etrcm.stage2d2.geometry import project


def auxiliary_weight(step:int,window:int,weight:float)->float:
    return float(weight) if step<window else 0.0


def alignment_loss(delta_h:torch.Tensor,basis:torch.Tensor,mode:str="useful")->torch.Tensor:
    flat=delta_h.reshape(-1,delta_h.shape[-1]); useful=project(flat,basis)
    ratio=useful.square().sum()/flat.square().sum().clamp_min(1e-12)
    if mode=="useful": return -ratio
    if mode=="orthogonal": return ratio
    raise ValueError(mode)
