"""Adjudicate G69--G72 without changing Stage 2D frozen criteria."""

from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path


def main(args):
    rows=[json.loads(p.read_text()) for p in sorted(args.results.glob("*.json"))]
    g69=[]; g70=[]; g71=[]; intervention=Counter(); detail=[]
    for x in rows:
        h=x["training_health"]; ok69=h["action_TV"]>=.1 and abs(h["interaction_y0"])>=.1 and h["CFA"]>0; g69.append(ok69)
        c=x["formation"]["0.70"]["curve"]; vals=[abs(c[str(n)]["BS"]) for n in (4,16,64)]; ok70=ok69 and vals[0]<vals[1]<vals[2]; g70.append(ok70)
        p=x["persistence"]; bs0=abs(p["curve"]["0"]["behavioral_separation_entropy"]); bs500=abs(p["curve"]["500"]["behavioral_separation_entropy"])
        persist=bs0>=.1 and bs500/max(bs0,1e-12)>=.2
        revision=x["revision"]["T_change"] is not None
        s=x["selectivity"]["matched"]; pred=abs(s["predictive"]["delayed"]["behavioral_separation_entropy"]); noise=abs(s["noise"]["delayed"]["behavioral_separation_entropy"])
        selective=pred>=noise+.05; ok71=persist and revision and selective; g71.append(ok71)
        base=abs(x["handoff"]["baseline"]["behavioral_separation_entropy"])
        for window in ("early_formation","late_formation"):
            for which in ("F","M","FM"):
                item=x["handoff"]["read_mediation"][window][which]
                if base-abs(item["behavioral_separation_entropy"])>0: intervention[(window,which)]+=1
        detail.append({"seed":x["seed"],"G69":ok69,"formation_BS":vals,"G70":ok70,"persistence":persist,
                       "D500_ratio":bs500/max(bs0,1e-12),"revision":revision,"selectivity":selective,"G71":ok71,
                       "continuous":x["continuous"]})
    best,count=intervention.most_common(1)[0]; payload={"n":len(rows),"G69":{"count":sum(g69),"pass":sum(g69)>=6},
        "G70":{"count":sum(g70),"pass":sum(g70)>=6},"G71":{"count":sum(g71),"pass":sum(g71)>=6},
        "G72":{"best_intervention":{"window":best[0],"which":best[1]},"count":count,"pass":count>=6},"runs":detail}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2)); print(json.dumps({k:v for k,v in payload.items() if k.startswith('G')}))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); p.add_argument("--out",type=Path,required=True); main(p.parse_args())
