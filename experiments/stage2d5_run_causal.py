"""Execute the frozen selected causal interventions in independent processes."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from stage2d5_audit import one


def main(args):
    root=args.root
    selected=json.loads((root/"processed"/"selection.json").read_text())["selected"]
    norms=json.loads((root/"processed"/"healthy_norms.json").read_text())
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[]
        for item in selected:
            path=root/"path_restoration"/"individual"/item["cohort"]/f"{item['run']}.json"
            futures.append(pool.submit(one,Path(item["checkpoint"]),item["arm"],item["cohort"],
                                       path,norms,True,args.reps))
        for future in as_completed(futures):
            result=future.result();print("COMPLETE",result["cohort"],result["run"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--jobs",type=int,default=12);p.add_argument("--reps",type=int,default=16)
    main(p.parse_args())
