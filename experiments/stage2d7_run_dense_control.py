"""Pre-outcome amendment: 1056-parameter C4 matched to C1's W_H+b_H."""

from pathlib import Path

from stage2d7_run_development import SEEDS, one


if __name__ == "__main__":
    root = Path("results/stage2d7/training_rescue/development")
    for init, stream in SEEDS:
        row = one(root, ("C4", 0, .1, 300), init, stream)
        print("COMPLETED", row["config"], row["init_seed"], flush=True)
