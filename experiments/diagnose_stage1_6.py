#!/usr/bin/env python3
"""Checkpoint Jacobian/JVP and gate diagnostics; never a causal gate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_6.runner import historical_oracle, make_model, scrub
from etrcm.stage1_6.world import generate_world


ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/"configs/stage1_6.yaml").read_text())
RUN=CONFIG["protocol"]["formal_run_id"]
STEPS=(0,160,500,1000,2000,3000)


def one(model, world, *, seed: int, step: int) -> dict[str,float]:
    state=model.initial_state(world.key.shape[0],device=world.key.device)
    with torch.no_grad():
        for event in world.events[:world.early_exposures]:
            state,_=model.step(state,event)
        oracle=historical_oracle(model,state,world).detach()
        state=scrub(model,state)
        for event in world.events[world.early_exposures:world.bridge_index]:
            state,_=model.step(state,event)
        event=world.events[world.bridge_index]
        encoded=model.event_encoder(event)
        h_pre=state.H+model.event_to_slots(encoded).view_as(state.H)
        q_fast,q_slow=model._queries(h_pre)
        gate=torch.sigmoid(model.core_gate(torch.cat([
            model.norm(h_pre),torch.zeros_like(oracle)[:,None,:].expand(-1,model.config.latent_slots,-1),
            encoded[:,None,:].expand(-1,model.config.latent_slots,-1)],dim=-1)))
    def h_after(read:torch.Tensor)->torch.Tensor:
        next_state,_=model.step_with_read(state,event,
                                           fast_read_override=torch.zeros_like(read),
                                           slow_read_override=read)
        return next_state.H
    # Four independent unit directions estimate RMS local sensitivity. This is
    # a JVP diagnostic, not evidence that memory was used beneficially.
    generator=torch.Generator(device="cpu").manual_seed(916_000+seed*31+step)
    magnitudes=[]
    for _ in range(4):
        direction=torch.randn(oracle.shape,generator=generator).to(oracle.device)
        direction= torch.nn.functional.normalize(direction,dim=-1)
        _,tangent=torch.autograd.functional.jvp(h_after,(oracle,),(direction,),create_graph=False)
        magnitudes.append(tangent.norm(dim=(-2,-1)).mean().item())
    return {"JVP_memory_to_H_norm":sum(magnitudes)/len(magnitudes),
            "recurrent_gate_mean":gate.mean().item(),
            "recurrent_gate_saturated_fraction":((gate<.05)|(gate>.95)).float().mean().item(),
            "fast_query_projection_norm":model.q_fast_projection.weight.norm().item(),
            "slow_query_projection_norm":model.q_slow_projection.weight.norm().item(),
            "q_fast_norm":q_fast.norm(dim=-1).mean().item(),
            "q_slow_norm":q_slow.norm(dim=-1).mean().item(),
            "oracle_read_norm":oracle.norm(dim=-1).mean().item()}


def main()->None:
    torch.set_num_threads(1)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows=[]
    raw=ROOT/"results/stage1_6/raw"/RUN
    for arm in ("learned","oracle","curriculum"):
        for seed in CONFIG["training"]["formal_seeds"]:
            cell=raw/f"{arm}-seed{seed}-lr0.001"
            for step in STEPS:
                model=make_model(Stage14Config.from_mapping(CONFIG),arm,seed=seed,device=device)
                checkpoint=torch.load(cell/f"checkpoint_{step:04d}.pt",map_location=device,weights_only=False)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                world=generate_world(batch=32,seed=17_000_000+seed,device=device)
                row={"run_id":RUN,"arm":arm,"seed":seed,"step":step,
                     **one(model,world,seed=seed,step=step)}
                rows.append(row)
    target=ROOT/"results/stage1_6/processed"/RUN
    target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_parquet(target/"jacobian_gate_diagnostics.parquet",index=False)
    print(json.dumps({"diagnostic_rows":len(rows)}))


if __name__=="__main__":
    main()
