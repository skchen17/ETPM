"""Read-only condition-wise BS comparison of two evaluation artifacts."""

import json
import sys
from pathlib import Path


def rows(path):
    return {((r["section"],r["exposure_count"],r["delay"],r["probe_type"],
              r["revision_count"],r["useful_noise_condition"],r["intervention"])):r
            for r in (json.loads(x) for x in Path(path).read_text().splitlines())}


a,b=map(rows,sys.argv[1:3])
assert a.keys()==b.keys()
differences=sorted(((abs(a[k]["BS"]-b[k]["BS"]),k) for k in a),reverse=True)
print(json.dumps({"rows":len(a),"max_abs_BS_change":differences[0][0],
                  "largest_condition":differences[0][1],
                  "count_over_1e-6":sum(x[0]>1e-6 for x in differences)}))
