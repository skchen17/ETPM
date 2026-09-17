# Learned No-Self-Evidence Control

The prediction head, recurrent core, memory query and access strength are learned. For unknowable episodes, the Bernoulli target is sampled independently and is absent from every event; the knowable parity arm is a positive training/control task.

| control | internal_tick | accuracy | confidence | entropy | ece | brier |
|---|---|---|---|---|---|---|
| knowable | 0 | 1.0000 | 0.9992 | 0.0061 | 0.0008 | 0.0000 |
| knowable | 1 | 1.0000 | 0.9997 | 0.0031 | 0.0003 | 0.0000 |
| knowable | 2 | 1.0000 | 0.9997 | 0.0029 | 0.0003 | 0.0000 |
| knowable | 4 | 1.0000 | 0.9996 | 0.0036 | 0.0004 | 0.0000 |
| knowable | 8 | 1.0000 | 0.9978 | 0.0144 | 0.0022 | 0.0000 |
| knowable | 16 | 1.0000 | 0.9642 | 0.1355 | 0.0358 | 0.0028 |
| knowable | 32 | 0.9990 | 0.8575 | 0.3678 | 0.1416 | 0.0302 |
| unknowable | 0 | 0.5025 | 0.5446 | 0.6871 | 0.0421 | 0.2529 |
| unknowable | 1 | 0.5025 | 0.5387 | 0.6896 | 0.0362 | 0.2517 |
| unknowable | 2 | 0.5025 | 0.5348 | 0.6906 | 0.0323 | 0.2511 |
| unknowable | 4 | 0.5025 | 0.5296 | 0.6913 | 0.0271 | 0.2508 |
| unknowable | 8 | 0.5025 | 0.5236 | 0.6913 | 0.0210 | 0.2508 |
| unknowable | 16 | 0.5028 | 0.5273 | 0.6907 | 0.0244 | 0.2511 |
| unknowable | 32 | 0.5032 | 0.5325 | 0.6897 | 0.0332 | 0.2516 |

## Finding

G12 is **PASS**. At K=32 unknowable accuracy was 0.5032; confidence changed by -0.0121 and ECE by -0.0089 relative to K=0. N9 was not triggered.
