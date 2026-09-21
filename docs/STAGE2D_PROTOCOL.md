# ET-RCM Stage 2D frozen protocol

> **Once ET-RCM can form endogenous behavioral memory, can repeated uncertain experience gradually produce a persistent but revisable behavioral disposition, and does causal control shift from fast memory F toward slow memory M over time?**

> **在 ET-RCM 已经能够形成 endogenous behavioral memory 的基础上，重复而不确定的经验能否逐渐形成持久但可修正的行为倾向，并且这种行为的因果控制是否会随时间从 fast memory F 转移到 slow memory M？**

Stage 2D is a new experiment. It does not revise Stage 2C–2C.3 reports or
artifacts. The core `AnatomicalETRCM` transition, external delta write,
readout-conserving F→M transfer, decay, NULL semantics and SELF_OUTPUT semantics
are unchanged.

## World and information boundary

Each hidden state is `z∈{0,1}`. A legal experience contains only surface tokens,
an action and an outcome. A simulator-only support bit equals `z` with
probability `p_evidence`; the observed action/outcome pair reveals that noisy
support bit but never z. Thus one event updates the Bayesian posterior to `p`,
not to 0 or 1. Balanced actions give the same marginal outcome distribution as
Stage 2C: `[1/2,1/6,1/6,1/6]`. Matched noise uses the same event count, surfaces,
actions, outcome marginals and compute, but a support bit independent of z.

The model is never passed z, a correct-action label, reward, importance, store,
retrieve or memory supervision. Bayesian posteriors are evaluation references
only.

## Development and frozen formal schedule

The inherited short curriculum (4/8/16 events, p=.65, 1,000 steps) was tried on
two development seeds and produced near-zero behavioral separation. This failed
development is retained. Development then tested the same architecture with a
longer noisy-evidence curriculum (16/32/64 events, p=.70, 1,500 steps). All
formal thresholds and schedules were frozen before formal seed 7801 began.

Formal primary unit: eight independently initialized/trained A2 protected-head
models. A0 and the gamma-zero, F-only, no-memory recurrent and GRU controls use
five independent seeds unless promoted to the eight-seed shortlist. All
comparisons use observed-consequence CE and equal endogenous training steps;
the A2 family additionally receives the disclosed 1,000-step privileged
evaluator pretraining used by Stage 2C.3.

The noisy-world interface-health prerequisite is action-TV ≥ .10 and positive
action×history interaction ≥ .10 in the same seed. These lower numerical
ceilings were frozen after development because irreducible p=.65 outcome noise
makes the deterministic Stage 2C.3 raw ceilings unattainable. CFA is still
reported but its old .10 threshold is not reused. The primary gate thresholds
are |BS|=.10 for acquisition/formed state, D500 retention ratio=.20,
predictive-over-noise margin=.05 and full-over-gamma0 retention margin=.03.

## Read and state interventions

Read clamps set finite F, M or both read vectors to zero while preserving stored
state. They are applied in six predeclared windows: early formation (events
1–8), late formation (25–32), four immediately post-formation NULL ticks,
mid-delay (ticks 249–252), late-delay (497–500), or the final probe. Strong
interventions zero or paired-swap F, M or FM once at the corresponding point.
Behavioral separation—not state norm—is the causal endpoint.

## Statistical discipline

The seed is the formal replication unit. Within-seed paired replicas reduce
Monte Carlo noise but do not increase n. G62–G67 require 6/8 primary-seed
replication where applicable. Full curves, failed seeds, null findings and all
controls remain machine-readable. Curve fits are descriptive only. No exact
recall gate is used.

The coarse phase diagram is development-only. A configuration is promoted only
when both sweep seeds satisfy the noisy-world interface-health prerequisite and
acquire by N64. This rule selected `γ=.50, ρF=.97, ρM=.9995`; it is retrained
from independent initialization on formal seeds 7801–7808. The default γ=.12
eight-seed run remains a separately reported negative result.
