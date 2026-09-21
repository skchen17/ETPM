"""Freeze one curriculum after the matched two-seed development screen."""

from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path


def main(args):
    groups=defaultdict(list)
    for p in args.audit.glob("*.json"):
        x=json.loads(p.read_text()); name=x["run"]
        m=re.match(r"(C[123])_f([0-9.]+)_w([0-9]+)_",name)
        if not m: continue
        key=f"{m.group(1)}_f{m.group(2)}_w{m.group(3)}"; groups[key].append(x["trajectory"][-1])
    rows=[]
    for key,vals in groups.items():
        hs=[v["health"] for v in vals]; count=sum(v["class"]=="healthy" for v in vals)
        quality=sum(h["action_TV"]+h["I_HA"]+abs(h["BS"])+max(0,h["CFA"]) for h in hs)/len(hs)
        rows.append({"candidate":key,"healthy":count,"n":len(vals),"quality":quality,
                     "means":{k:sum(h[k] for h in hs)/len(hs) for k in ("action_TV","I_HA","BS","CFA")}})
    # Primary criterion is replicated health; quality breaks ties. C1 wins an exact tie
    # because it is the least direct routing scaffold.
    priority={"C1":2,"C3":1,"C2":0}
    selected=max(rows,key=lambda r:(r["healthy"],r["quality"],priority[r["candidate"][:2]]))
    payload={"candidates":sorted(rows,key=lambda r:r["candidate"]),"selection_rule":"healthy count, then mean interface quality; no final-test access",
             "selected":selected,"C4_authorized":False,"C4_reason":"G68 failed; C1/C2 independent causal benefit not established"}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2)); print(json.dumps(selected))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--audit",type=Path,required=True); p.add_argument("--out",type=Path,required=True); main(p.parse_args())
