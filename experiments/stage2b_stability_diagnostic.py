"""Separate, post-training residual-scale diagnostic; never part of G38–G41."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from etrcm.stage2b.data import make_association
from stage2b_eval import load,score_examples,aggregate,stability


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    model,tok,ckpt=load(args.checkpoint,args.device)
    if ckpt["arm"] not in {"E1","E2"}:
        raise ValueError("stabilization diagnostic restricted to contextual arms")
    rng=random.Random(9114)
    examples=[make_association(rng,"attribute",128,"train") for _ in range(16)]
    results=[]
    for alpha in (1.0,0.5,0.25,0.1):
        hook=None
        if alpha!=1.0:
            hook=model.core_out.register_forward_hook(lambda _module,_inputs,output,scale=alpha:output*scale)
        try:
            rows=score_examples(model,tok,examples,args.device)
            stable=stability(model,tok,args.device)
            results.append({"alpha":alpha,"recall":aggregate(rows),"completed":stable["completed"],
                            "first_H_gt_100":stable["first_H_gt_100"],
                            "first_H_gt_1000":stable["first_H_gt_1000"],
                            "first_nonfinite":stable["first_nonfinite"],
                            "H_at_1000":stable["snapshots"]["1000"]["H"] if stable["snapshots"]["1000"] else None,
                            "H_at_5000":stable["snapshots"]["5000"]["H"] if stable["snapshots"]["5000"] else None})
        finally:
            if hook is not None:hook.remove()
    payload={"arm":ckpt["arm"],"seed":ckpt["seed"],"size":ckpt["config"]["hidden_dim"],
             "design":"post-training forward-hook scales core_out proposal; no retraining, not a formal contextual-KV arm",
             "results":results}
    path=args.out/"stabilization.json"
    path.write_text(json.dumps(payload,indent=2))
    (args.out/"hashes.json").write_text(json.dumps({path.name:hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))
    print(json.dumps(payload),flush=True)


if __name__=="__main__":main()
