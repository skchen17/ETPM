# Stage 1.4 predictive continuous-state architecture

Stage 1.4 retains exactly the persistent state `(H,F,M)` and the historical
external-write and conserving-consolidation laws. Only the present validated
external event can write new evidence into F. NULL and SELF_OUTPUT never use
that code path. The correct model has no planner, halting flag, thought trace,
episodic database or decoder LM.

The primary B5 read makes two unit queries from the same active H:
`q_F=normalize(Q_F(H))`, `q_M=normalize(Q_M(H))`. Fast and slow matrix reads
are separately RMS-normalized, then a two-way softmax gate conditioned on
`(H,current event)` mixes them. The fast query drives the unchanged
`Delta=gamma*a*(F q_F)q_F^T` transfer; F loses exactly what M gains before
decay. The recurrent gated core writes the mixed read back to H. Four
prediction heads forecast future event tokens at external horizons 1/2/4/8;
they read H only. Thus any peripheral effect on prediction after a NULL tick
must first affect H, while persistent F/M can re-enter H on later ticks.

R0 joint and R1 shared-query scalar arbitration reuse the same transition
law. B0/B1 have no persistent matrix; B2 has one slow matrix; B6 disables
consolidation; B7 has a fixed random slow query. They share the same
world-prediction training/selection budget, not necessarily the same active
parameter count. R3 joint fusion is not needed for the frozen comparison and
is not implemented as a formal arm.

The latent-transition world has a hidden z available only to its generator.
Current event key/value fields contain noisy observations, not z. In the
long-gap relation world, memory stores B→A; after unrelated interference,
a C→B bridge is observed and an A-dependent future token is forecast.
No memory-query supervision or target key enters the model. Future targets
are shifted strictly beyond the external prefix.

Same-H peripheral swaps, H restoration, rank-one M-direction lesions and a
direction-specific consolidation block are intervention interfaces. The
block is conditional on development evidence; its implementation does not
authorize formal Experiment G. Learned keys are nonorthogonal, so a direction
lesion can affect correlated keys; this limits item-level interpretation.

Prediction usefulness, descriptive memory access, intervention-based causal
influence and delayed M-only retention are four different measurements.
