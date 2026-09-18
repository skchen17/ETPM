# Self-Output Evidence Audit — Stage 1.3

## Experimental details

- Formal run: `stage1_3-formal-v1`; 8 unseen seeds `[5301, 5302, 5303, 5304, 5305, 5306, 5307, 5308]`.
- Architectures: B0 no memory, B1 GRU, B2 single persistent matrix, B3 joint F+M, B4 M-only, B5 F-only, B6 learned scalar arbitration, B7 non-conserving self-replay control.
- Every architecture received 1,000 training steps, batch size 64, identical task cycle and model-independent development budget. Learning rates and expression thresholds were selected separately per architecture using only development seeds 4301/4302.
- State: H=2×64, F/M=16×16; gamma=0.12, rho_fast=0.97, rho_slow=0.9995, eta_external=0.6.
- Formal evaluation sizes per seed are A=512 episodes, B=256 per condition, C=128, E=512, F=128, G=128, H=32 per distractor/condition, I=256×8 facts; D uses 16 parallel streams and B6/B7 additionally run 10,000 ticks.
- B6 development threshold primary-feasible: `False`. Its frozen diagnostic threshold is `0.45`.

## Post-expression trajectories

| model                        |   internal_tick |   external_write_count |   memory_relative_change |   expression_score_change |   self_memory_update |   emitted_rate |   duplicate_target_emission_rate |
|:-----------------------------|----------------:|-----------------------:|-------------------------:|--------------------------:|---------------------:|---------------:|---------------------------------:|
| B6_arbitration               |               0 |                      0 |                   0      |                    0      |               0      |         0.998  |                           0.998  |
| B6_arbitration               |               1 |                      0 |                  -0.0294 |                   -0.5177 |               0      |         0.4961 |                           0.3289 |
| B6_arbitration               |               2 |                      0 |                  -0.0579 |                   -0.4768 |               0      |         0.5403 |                           0.0549 |
| B6_arbitration               |               4 |                      0 |                  -0.1123 |                   -0.1863 |               0      |         0.8418 |                           0.0376 |
| B6_arbitration               |               8 |                      0 |                  -0.2112 |                   -0.2546 |               0      |         0.7793 |                           0.0264 |
| B6_arbitration               |              16 |                      0 |                  -0.3756 |                   -0.1835 |               0      |         0.8684 |                           0.0381 |
| B6_arbitration               |              32 |                      0 |                  -0.6041 |                   -0.1738 |               0      |         0.8767 |                           0.0332 |
| B6_arbitration               |              64 |                      0 |                  -0.8283 |                   -0.1594 |               0      |         0.9033 |                           0.0271 |
| B7_nonconserving_self_replay |               0 |                      0 |                   0      |                    0      |               0.2495 |         0.998  |                           0.998  |
| B7_nonconserving_self_replay |               1 |                      0 |                  -0.0373 |                   -0.5155 |               0.1246 |         0.4983 |                           0.3306 |
| B7_nonconserving_self_replay |               2 |                      0 |                  -0.0653 |                   -0.4751 |               0.1354 |         0.5417 |                           0.0586 |
| B7_nonconserving_self_replay |               4 |                      0 |                  -0.1184 |                   -0.1884 |               0.2095 |         0.8381 |                           0.042  |
| B7_nonconserving_self_replay |               8 |                      0 |                  -0.2172 |                   -0.2533 |               0.1932 |         0.7729 |                           0.0295 |
| B7_nonconserving_self_replay |              16 |                      0 |                  -0.3764 |                   -0.1785 |               0.2157 |         0.8628 |                           0.0342 |
| B7_nonconserving_self_replay |              32 |                      0 |                  -0.5625 |                   -0.1617 |               0.2213 |         0.8853 |                           0.0391 |
| B7_nonconserving_self_replay |              64 |                      0 |                  -0.6094 |                   -0.1486 |               0.2285 |         0.9138 |                           0.0256 |

G22: **FAIL**. B6 safety subcriteria pass=True; external-write maximum=0; maximum mean memory change=0.0000; maximum mean score change=0.0000. B7 executed nonzero self-memory updates, but net target-memory strength did not reach the registered +0.10 control increase (observed 0.0000), so the compound gate fails because the pathological-control check was not validated—not because B6 amplified itself.
