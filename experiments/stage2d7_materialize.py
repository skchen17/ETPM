"""Split run-level audit payloads into the preregistered result families."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/stage2d7")
COHORTS = ("historical_legacy", "stage2d6_confirmatory", "stage2d7_confirmatory")


def save(folder, cohort, run, payload):
    path = ROOT / folder / cohort / f"{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    count = 0
    for cohort in COHORTS:
        for path in sorted((ROOT / "projection_visibility" / cohort).glob("*.json")):
            data = json.loads(path.read_text())
            run, endpoint = data["run"], data["endpoint"]
            vis = endpoint["visibility"]
            save("state_projection_svd", cohort, run, {"run": run, "cohort": cohort,
                "rank": vis["matrix_rank"], "singular_values": vis["singular_values"],
                "reconstruction_error": vis["svd_reconstruction_error"],
                "energy": vis["energy"], "V_H": vis["V_H"]})
            save("f1_f2_visibility", cohort, run, {"run": run, "cohort": cohort,
                "class": endpoint["class"], "phenotype": endpoint["phenotype"],
                "D_H": vis["D_H"], "D_S": vis["D_S"], "G_proj": vis["G_proj"],
                "probe_H": endpoint["health"]["z_probe"]["H"],
                "probe_M": endpoint["health"]["z_probe"]["M"]})
            save("training_trajectories", cohort, run, {"run": run, "cohort": cohort,
                "trajectory": data.get("trajectory", [])})
            if "state_main_rescue" in endpoint:
                rescue = endpoint["state_main_rescue"]
                save("state_main_rescue", cohort, run, {"run": run, "cohort": cohort,
                    "class": endpoint["class"], "phenotype": endpoint["phenotype"],
                    "interventions": rescue})
                save("controls", cohort, run, {"run": run, "cohort": cohort,
                    "random_state": rescue["random_state"],
                    "low_sensitivity_state": rescue["low_sensitivity_state"],
                    "action_main": rescue["action_main"],
                    "common_shift": rescue["common_shift"],
                    "fusion_post_positive": rescue["fusion_post_positive"]})
            if "healthy_destruction" in endpoint:
                save("healthy_state_destruction", cohort, run, {"run": run,
                    "cohort": cohort, "interventions": endpoint["healthy_destruction"]})
            count += 1
    for folder in ("h_coordinate_rotation", "expanded_replication", "dynamics_recheck",
                   "memory_mediation", "continuous_runs", "processed", "manifests"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    print("MATERIALIZED", count)


if __name__ == "__main__":
    main()
