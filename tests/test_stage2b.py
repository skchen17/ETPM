"""Stage 2B math, ordering, provenance, split and frozen-history guards."""

import hashlib
import random
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2a.language import lesion
from etrcm.stage2b.data import COLORS, NAMES, PLACES, allowed, counterfactual_pair, make_association, make_corpus, sample_basic_safe
from etrcm.stage2b.interventions import derangement, random_norm_matched, shuffled_memory, swap_component
from etrcm.stage2b.model import ContextualLanguageETRCM
from etrcm.stage2a.language import WordTokenizer
from experiments.stage2b_train import make_batch


def net(arm="E1"):
    torch.manual_seed(5)
    return ContextualLanguageETRCM(Stage14Config(hidden_dim=32,latent_slots=1,symbol_count=32,
                                                 key_dim=8,value_dim=8),arm)


def test_contextual_key_value_shapes_and_normalization():
    for arm in ("E1","E2"):
        model=net(arm)
        state=model.initial_state(3)
        key,value=model.contextual_kv(state.H,torch.tensor([2,3,4]))
        assert key.shape==(3,8) and value.shape==(3,8)
        assert torch.allclose(key.norm(dim=-1),torch.ones(3),atol=1e-5)


def test_no_future_target_leakage_and_same_step_read_before_write():
    for arm in ("E1","E2"):
        model=net(arm)
        state=model.initial_state(1)
        before=state.clone()
        next_a,_,diag_a=model.step_token(state,torch.tensor([5]))
        # Divergent future target is not part of the current token interface.
        next_b,_,diag_b=model.step_token(before,torch.tensor([5]))
        assert torch.equal(diag_a["context_key"],diag_b["context_key"])
        assert torch.equal(diag_a["context_value"],diag_b["context_value"])
        assert torch.equal(next_a.F,next_b.F)
        assert float(diag_a["read_pre_write_F"].norm())==0
        assert float(diag_a["read_pre_write_M"].norm())==0
        assert float(diag_a["external_update"].norm())>0


def test_nonzero_old_memory_read_is_not_contaminated_by_current_write():
    model=net("E1")
    state=model.initial_state(1)
    state,_,_=model.step_token(state,torch.tensor([5]))
    old=state.clone()
    token=torch.tensor([6])
    from etrcm.stage1_3.events import evidence_event
    event=evidence_event(token,token)
    encoded=model.event_encoder(event).to(old.H.dtype)
    H_pre=old.H+model.event_to_slots(encoded).view_as(old.H)
    q_F,q_M=model._queries(H_pre)
    old_reads=model._read_stage14(old.F,old.M,q_F,q_M,H_pre,encoded)
    newer,_,diag=model.step_token(state,token)
    assert torch.allclose(diag["read_pre_write_F"],old_reads["r_F"],atol=1e-6)
    assert torch.allclose(diag["read_pre_write_M"],old_reads["r_M"],atol=1e-6)
    assert float(diag["external_update"].norm())>0
    assert not torch.equal(newer.F,old.F)


def test_answer_target_is_shifted_after_prefix_not_in_current_event():
    example=make_association(random.Random(99),"attribute",16,"train")
    tok=WordTokenizer.fit([example.text])
    tokens,position=make_batch([example],tok,"cpu")
    t=int(position[0])
    assert int(tokens[0,t+1])==tok.encode(example.answer)[0]
    assert int(tokens[0,t])==tok.to_id[":"]
    # Altering the future target cannot change the prefix/current event.
    altered=tokens.clone();altered[0,t+1]=tok.to_id["<unk>"]
    assert torch.equal(tokens[0,:t+1],altered[0,:t+1])


def test_contextual_write_is_identical_for_same_prefix_different_future_label():
    example=make_association(random.Random(101),"attribute",16,"train")
    fake_answer=next(color for color in COLORS if color!=example.answer)
    fake=replace(example,text=example.prefix+f" {fake_answer} .",answer=fake_answer)
    tok=WordTokenizer.fit([example.text,fake.text])
    model=ContextualLanguageETRCM(Stage14Config(hidden_dim=32,latent_slots=1,
                                                symbol_count=len(tok.vocabulary),key_dim=8,value_dim=8),"E1")
    captured=[]
    for episode in (example,fake):
        state=model.initial_state(1)
        last=None
        for token in tok.encode(episode.prefix,bos=True):
            state,_,last=model.step_token(state,torch.tensor([token]))
        captured.append((last["context_key"].clone(),last["context_value"].clone(),state.F.clone()))
    assert all(torch.equal(x,y) for x,y in zip(captured[0],captured[1]))


def test_null_self_no_external_write_and_finite():
    model=net(); state=model.initial_state(1)
    state,_,external=model.step_token(state,torch.tensor([5]))
    assert bool(external["external_write_flag"][0])
    previous_external_time=state.external_time
    state,_,null=model.null_tick(state)
    assert not bool(null["external_write_flag"][0]) and float(null["external_update"].norm())==0
    state,_,self_token=model.step_token(state,torch.tensor([6]),source="self_output")
    assert not bool(self_token["external_write_flag"][0])
    assert float(self_token["external_update"].norm())==0
    assert state.external_time==previous_external_time
    for _ in range(40):
        state,_,_=model.null_tick(state)
    assert all(torch.isfinite(x).all() for x in (state.H,state.F,state.M))


def test_H_F_M_lesions_and_swaps_exact():
    model=net(); a=model.initial_state(1); b=model.initial_state(1)
    a,_,_=model.step_token(a,torch.tensor([5])); b,_,_=model.step_token(b,torch.tensor([6]))
    reset=lesion(a,"H",model)
    assert torch.equal(reset.F,a.F) and torch.equal(reset.M,a.M)
    assert torch.equal(lesion(a,"F",model).F,torch.zeros_like(a.F))
    assert torch.equal(lesion(a,"M",model).M,torch.zeros_like(a.M))
    for component in ("F","M","FM"):
        swapped,_=swap_component(a,b,component)
        assert torch.equal(swapped.F,b.F if component in {"F","FM"} else a.F)
        assert torch.equal(swapped.M,b.M if component in {"M","FM"} else a.M)


def test_derangement_and_norm_matched():
    order=derangement(8,torch.Generator().manual_seed(2))
    assert sorted(order.tolist())==list(range(8))
    assert all(int(order[i])!=i for i in range(8))
    model=net(); state=model.initial_state(8)
    for token in (5,6,7):
        state,_,_=model.step_token(state,torch.full((8,),token))
    changed=shuffled_memory(state,order)
    assert torch.equal(changed.F,state.F[order])
    random_state=random_norm_matched(state,torch.Generator().manual_seed(3))
    assert torch.allclose(random_state.F.norm(dim=(-2,-1)),state.F.norm(dim=(-2,-1)),atol=1e-5)
    assert torch.allclose(random_state.M.norm(dim=(-2,-1)),state.M.norm(dim=(-2,-1)),atol=1e-5)


def test_lexical_ood_disjoint_and_counterfactual_inventory():
    for name in NAMES:
        for color in COLORS:
            assert allowed(name,color,COLORS,"train") != allowed(name,color,COLORS,"ood")
    pair=counterfactual_pair(random.Random(42),32)
    assert pair[0].answer!=pair[1].answer
    assert sorted(pair[0].prefix.split())==sorted(pair[1].prefix.split())
    assert pair[0].question==pair[1].question


def test_auxiliary_basic_examples_do_not_leak_name_value_pairs():
    rng=random.Random(12)
    for _ in range(300):
        text=sample_basic_safe(rng).text
        assert not (any(name in text.split() for name in NAMES) and
                    any(value in text.split() for value in COLORS+PLACES))


def test_primary_lexical_ood_final_associations_disjoint():
    families={"attribute","location","revision","interference"}
    train={(ex.family,ex.entity,ex.value) for ex in make_corpus(20,4800,"train") if ex.family in families}
    ood={(ex.family,ex.entity,ex.value) for ex in make_corpus(21,4800,"ood") if ex.family in families}
    assert train and ood and train.isdisjoint(ood)


def test_answer_label_balance_and_gaps():
    corpus=make_corpus(42,4800,"train")
    colors=Counter(ex.answer for ex in corpus if ex.family=="attribute")
    assert len(colors)==8
    assert max(colors.values())/min(colors.values())<1.6
    assert {16,32,64,128}.issubset({ex.gap for ex in corpus})
    interference=[ex for ex in corpus if ex.family=="interference"]
    counts={ex.prefix.split(" question :")[0].count(" has the ") for ex in interference}
    assert {8,16,32}.issubset(counts)


def test_historical_stage1_and_stage2a_files_unchanged():
    root=Path(__file__).resolve().parents[1]
    frozen=(root/"artifacts/stage1_5_all_assets.sha256").read_text().splitlines()
    for line in frozen:
        digest,relative=line.split(maxsplit=1)
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
    previous="d458abd8f5a674e71a57a7d129207278df07cb16"
    paths=subprocess.check_output(["git","ls-tree","-r","--name-only",previous],cwd=root,text=True).splitlines()
    stage2a=[p for p in paths if p.startswith(("src/etrcm/stage2a/","results/stage2a/","experiments/stage2a_"))
             or p in {"reports/STAGE2A_NATURAL_LANGUAGE_PROTOTYPE_RESULTS.md","STAGE2A_START_HERE.md","configs/stage2a.yaml","tests/test_stage2a.py"}]
    assert len(stage2a)>30
    for relative in stage2a:
        historical=subprocess.check_output(["git","show",f"{previous}:{relative}"],cwd=root)
        assert (root/relative).read_bytes()==historical,relative
