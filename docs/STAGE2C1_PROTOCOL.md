# Stage 2C.1 protocol — action–outcome binding ladder

This diagnostic is independent of frozen Stage 2C. It does not change the
external F write, readout-conserving F→M transfer, decay, gamma, memory size, or
the Stage 2C historical thresholds/results. Its purpose is to localize a
behavioral-interface failure before attributing null behavior to memory.

## World and causal contrasts

The world is exactly the Stage 2C `z ∈ {A,B}` environment. An action matching
`z` yields observable outcome 0; a nonmatching action yields one of outcomes
1–3 uniformly. The training target is only observed consequence cross-entropy.
There is no reward, importance, correct-action label, or action policy loss.

For a fixed state, the finite action intervention changes only candidate
action A↔B. Record the entire four-way forecast, TV, KL in both directions,
JS, entropy difference, p0 difference, and correct-action p0 margin. Also
compare A/B states at fixed action and the history×action interaction.

## Ladder and stop rule

1. **L0 oracle latent:** train a two-index latent embedding plus two small
   candidate-action heads, late concat `[H;a]` and additive hidden modulation
   `H+W_a a`. Balanced four-cell batches. Two development seeds 7101–7102;
   pre-registered eight formal seeds 7201–7208, with TV and expected CE at
   0/100/500/1000/3000 steps. G48 requires both latent-specific action-TV
   ≥0.50 and both ΔQ>0 in at least 6/8 late-concat seeds.
2. **L1 oracle persistent M:** only after G48. Fixed, random QR-derived,
   orthogonal equal-norm 8×8 M_A/M_B (seed 271828). Same H0, F0=0, no present
   event; one inherited `AnatomicalETRCM.step` with no external write.
   Candidate action enters only the consequence head. M never goes directly to
   output. G49 requires both action-TV≥0.50, entropy-policy BS≥0.30, and exact
   M-swap preference reversal in 6/8 seeds.
3. **L2 legal-history encoder:** only after G49. Eight past records of
   `(color, shape, nuisance, actual action, observed outcome)` feed a small
   GRU and reshape into M, which then goes through the same inherited read→H
   path. No latent, correct-action label, or future outcome enters the model.
   Novel evaluation uses held-out parity combinations in the past history,
   64 fresh histories per seed. G50 requires both action-TV≥0.50, entropy BS
   ≥0.30 and positive history×action interaction in 6/8 seeds.
4. **L3 full lifetime diagnostic:** only after all gates pass. Train the Stage
   2C B5 separate-read core and unchanged F/M law for 500 steps, batch 16,
   4/8/16 experiences per lifetime, AdamW 0.001, H-square penalty 0.001,
   gradient clip 1. The only behavioral change is a direct late-concat
   candidate-action head. Evaluate four paired lifetimes per seed for
   formation, unrelated-delay persistence, seen/novel/hard-OOD surface
   generalization, opposing-evidence revision, F/M/FM swaps and direct
   action-branch forecasts. This is not a re-adjudication of G42–G47.

Formal parameters and thresholds are in `configs/stage2c1_formal.yaml` and
`configs/stage2c1_lifetime.yaml`; these were frozen before the respective
formal runs. All checkpoints, logs, interventions and processed aggregates are
under `results/stage2c1/`. The final report is
`reports/STAGE2C1_ACTION_OUTCOME_BINDING_RESULTS.md`.

## Limits on interpretation

L0 proves only a supervised oracle-state/action forecast can be learned. L1
proves the inherited M-read→H route can transmit an imposed state over one
tick. L2 proves legal past observations can be encoded into a useful imposed
state. None proves endogenous F/M formation, persistence or human-like memory.
L2's novel split varies past-history surface combinations, not the current
probe surface. An L3 failure with weak L3 action-TV is a mixed joint-training
failure, not a uniquely identified memory-law defect.

Historical Stage 2C reference: `reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md`.
The Stage 2C source, config, report and all 876 result files are inventoried by
SHA-256 under `results/stage2c1/manifests/` and never written by Stage 2C.1.
