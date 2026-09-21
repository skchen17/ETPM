# Stage 2C.2 protocol — frozen failed-L3 pathway audit

The object of this stage is the eight **failed** Stage 2C.1 L3 checkpoints
`results/stage2c1/raw/lifetime/7201..7208/checkpoint.pt`. Their weights are
frozen during P0/P1 and native-state probes. The inherited F write,
readout-conserving F→M transfer, decay, queries, core and action head are not
modified. Historical Stage 2C and 2C.1 assets are read-only and inventoried by
SHA-256 under `results/stage2c2/manifests/`.

## Decision ladder

P0 replays four matched N=16 paired histories per seed, recording forecasts,
TV_A, TV_H, history×action interaction, BS, H/F/M norms, q_F/q_M and r_F/r_M.
P1 then fits only two **L3-local** oracle H state variables. Both H norms are
projected to the checkpoint's own native N=16 median H norm. The frozen action
head receives balanced candidate actions and observed consequence CE; it gets
neither a correct-action label nor an outcome one-hot input. Four restarts of
1,000 Adam steps use disjoint fit and formal observed-outcome samples. P1
injects H immediately before the head, isolating the downstream ceiling.

G51 requires both latent-conditional action-TV branches ≥0.10, positive
interaction ≥0.15 and entropy-policy BS ≥0.10 in at least 6/8 formal frozen
checkpoints. Two independently trained failed-L3 development checkpoints
(7301–7302) were used before the threshold was frozen. Formal frozen
checkpoints are 7201–7208; details are in `configs/stage2c2_formal.yaml`.

Only if G51 passes may P2 fit norm-matched oracle read vectors using the
inherited read→H route. Only if G52 passes may P3 fit per-checkpoint oracle M
and test swap/read-clamp. Only if G53 passes may P4 train a legal-history→M
adapter. Only if G54 passes may curriculum C1–C3 be evaluated; C0 is the
already frozen Stage 2C.1 scratch baseline. A failed gate
marks all downstream gates `NOT_RUN_BY_GATE`; absent interventions are not
interpreted as failures. The prospective zero-assistance curriculum schedule
is `(1.0, 0.75, 0.5, 0.25, 0.0)` and a G55 score would require exactly zero
teacher probability at final evaluation. No curriculum score is inferred from
parallel scratch training.

## Parallel non-causal diagnostics

The native z-probe uses 128 paired training lifetimes and 128 disjoint held-out
lifetimes per frozen checkpoint. It fits a linear and a 16-unit MLP decoder on
H, F, M and [F;M]. Labels are visible only to probes. Training surfaces use
parity-even color/shape combinations, testing parity-odd held-out
combinations. Decodability alone is not causal mediation. A routing rescue
adapter requires both an intact downstream gate and replicated M linear
accuracy ≥0.70 in 6/8 seeds; otherwise it is not run.

The independent scratch joint-training curve is another diagnostic, **not** a
curriculum. It reproduces the Stage 2C.1 L3 objective and unchanged memory
law from new seeds (development 7351–7352; formal 7401–7408). Snapshots at
steps 0/25/50/100/200/300/500 store action-TV, history-TV, interaction,
native BS, M latent-z ridge-probe accuracy, r_M/H differences, H norm and
pre-clipping gradients of the action branch, consequence head, q_M, read gate,
event encoder and H core. Gradient magnitudes are diagnostic only. Details
were fixed in `configs/stage2c2_joint_curve.yaml` before curve experiments.

All outputs are under `results/stage2c2/`; the final report is
`reports/STAGE2C2_FROZEN_L3_PATHWAY_AUDIT_RESULTS.md`. Stage 2C and 2C.1
gates and reports remain frozen.
