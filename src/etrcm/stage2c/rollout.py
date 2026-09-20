"""Lifetime transitions, consequence prediction and entropy-based action policy."""

from __future__ import annotations

import random

import torch
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from .model import BehavioralModel
from .world import ABSTRACT, ACTION, NUISANCE, OUTCOME, Experience, context_token, tensor_ids, unrelated_token


def _event_ids(experiences: list[Experience], field: str, device) -> torch.Tensor:
    return tensor_ids([getattr(item, field) for item in experiences], device)


def play_experience(model: BehavioralModel, state: LearnedState,
                    experiences: list[Experience], *, read_clamp: str = "none",
                    write_block: bool = False, trace=None):
    """Forecast y after observing context and action, then observe actual y."""
    device=state.H.device
    tokens=(torch.full((len(experiences),), ABSTRACT, dtype=torch.long, device=device),
            _event_ids(experiences,"color",device),_event_ids(experiences,"shape",device),
            _event_ids(experiences,"nuisance",device),
            tensor_ids([ACTION[e.action] for e in experiences],device))
    diagnostics=[]
    for label,ids in zip(("abstract_context","color","shape","nuisance","action"),tokens):
        state,diag=model.step(state,context_token(ids),read_clamp=read_clamp)
        diagnostics.append(diag)
        if trace is not None:trace(label,state,diag,None,None)
    logits=model.consequence_logits(state)
    target=_event_ids(experiences,"outcome",device)-OUTCOME[0]
    loss=F.cross_entropy(logits,target,reduction="none")
    # Only the observable consequence is new external evidence. The raw token
    # remains the event key/value, exactly as in the inherited memory law.
    from .world import outcome_token
    state,diag=model.step(state,outcome_token(_event_ids(experiences,"outcome",device)),
                          read_clamp=read_clamp,write_block=write_block)
    diagnostics.append(diag)
    if trace is not None:trace("outcome",state,diag,logits,loss)
    return state,loss,logits,diagnostics


def probe_policy(model: BehavioralModel, state: LearnedState,
                 surfaces: list[tuple[int,int,int]], *, temperature: float = 0.35,
                 read_clamp: str = "none"):
    """Prefer actions whose learned future distribution is more predictable.

    No action or environment label is read by this policy. Both hypothetical
    action branches start from the same state and are discarded after scoring.
    """
    device=state.H.device
    features=(torch.full((len(surfaces),),ABSTRACT,dtype=torch.long,device=device),
              tensor_ids([x[0] for x in surfaces],device),
              tensor_ids([x[1] for x in surfaces],device),
              tensor_ids([x[2] for x in surfaces],device))
    probe_state=state.clone()
    context_diags=[]
    for ids in features:
        probe_state,diag=model.step(probe_state,context_token(ids),read_clamp=read_clamp)
        context_diags.append(diag)
    forecasts=[]
    action_states=[]
    action_diags=[]
    for action in range(2):
        branch,diag=model.step(probe_state.clone(),context_token(
            torch.full((len(surfaces),),ACTION[action],dtype=torch.long,device=device)),
            read_clamp=read_clamp)
        forecasts.append(model.consequence_logits(branch))
        action_states.append(branch)
        action_diags.append(diag)
    logits=torch.stack(forecasts,dim=1)
    log_p=logits.log_softmax(-1)
    entropy=-(log_p.exp()*log_p).sum(-1)
    action_logits=-entropy/temperature
    probs=action_logits.softmax(-1)
    return probs,action_logits,logits,probe_state,action_states,context_diags+action_diags


def unrelated_delay(model: BehavioralModel, state: LearnedState, ticks: int,
                    seed: int, *, read_clamp: str = "none", trace=None):
    rng=random.Random(seed)
    batch=state.H.shape[0]
    for t in range(ticks):
        # Identical nuisance tokens for each A/B pair, no latent cue.
        values=[rng.choice(NUISANCE) for _ in range((batch+1)//2)]
        ids=tensor_ids([values[i//2] for i in range(batch)],state.H.device)
        state,diag=model.step(state,unrelated_token(ids),read_clamp=read_clamp)
        if trace is not None:
            trace(t,state,diag,"delay")
    return state
