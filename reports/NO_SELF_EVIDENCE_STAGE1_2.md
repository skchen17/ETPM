# No Self-Evidence Without an Explicit Cue — Stage 1.2

Both strata use the same FACT, QUERY and NULL event kinds, shapes and learned head. Only evidence availability differs; the unknowable target is independent.

| condition | internal_tick | accuracy | confidence | entropy | ece | brier |
|---|---|---|---|---|---|---|
| knowable | 0 | 0.9993 | 0.8794 | 0.3470 | 0.1199 | 0.0194 |
| knowable | 1 | 1.0000 | 0.9021 | 0.3028 | 0.0979 | 0.0129 |
| knowable | 2 | 1.0000 | 0.9065 | 0.2924 | 0.0935 | 0.0119 |
| knowable | 4 | 1.0000 | 0.9032 | 0.2971 | 0.0968 | 0.0130 |
| knowable | 8 | 1.0000 | 0.8867 | 0.3265 | 0.1133 | 0.0181 |
| knowable | 16 | 1.0000 | 0.8565 | 0.3758 | 0.1435 | 0.0289 |
| knowable | 32 | 0.9953 | 0.8209 | 0.4257 | 0.1744 | 0.0446 |
| knowable | 64 | 0.9838 | 0.7896 | 0.4641 | 0.1942 | 0.0611 |
| unknowable | 0 | 0.4997 | 0.6591 | 0.6154 | 0.1594 | 0.2867 |
| unknowable | 1 | 0.5002 | 0.6572 | 0.6170 | 0.1570 | 0.2856 |
| unknowable | 2 | 0.5000 | 0.6548 | 0.6185 | 0.1548 | 0.2846 |
| unknowable | 4 | 0.5027 | 0.6507 | 0.6213 | 0.1481 | 0.2829 |
| unknowable | 8 | 0.5021 | 0.6459 | 0.6251 | 0.1438 | 0.2810 |
| unknowable | 16 | 0.5016 | 0.6409 | 0.6293 | 0.1393 | 0.2790 |
| unknowable | 32 | 0.5002 | 0.6364 | 0.6330 | 0.1361 | 0.2774 |
| unknowable | 64 | 0.5005 | 0.6328 | 0.6359 | 0.1323 | 0.2761 |

G17: **FAIL**.
