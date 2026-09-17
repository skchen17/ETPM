# Stage-2 language-model plan (gated)

Stage 2 begins only if the Stage-1 report supports G1–G6 without a major
unresolved negative criterion. The target is a 30M–100M parameter decoder:

```text
Token embedding
      ↓
Local causal attention
      ↓
Shared recurrent latent core
      ↓
ET-RCM F/M state
      ↓
FFN
      ↓
LM head
```

Language input and output are interfaces between external events and persistent
state. The central object remains `S_(tau+1) = Phi(S_tau, e_tau)`, including
`Phi(S_tau, 0)`.

Required gates before scaling:

1. conservation and analytic tests pass in float32 and float64;
2. use-dependent retention beats uniform transfer under a matched budget;
3. single persistent memory does not explain the same results without a worse
   stability/plasticity tradeoff;
4. idle compute shows a benefit beyond an unreported compute shift, or the
   claim is narrowed to readiness/latency;
5. unknowable-bit confidence remains calibrated;
6. revision succeeds across old-strength/new-evidence sweeps.

Stage 2 would add multi-head memory, more than one latent slot, a shared
Transformer core, checkpointable persistent state, and causal interventions on
`F/M/H`. It would not add remember-token heuristics, a vector database, RAG, or
an importance classifier to rescue a failed Stage-1 mechanism.

