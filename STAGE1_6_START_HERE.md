# ET-RCM Stage 1.6 — results and reproduction

The repository README is intentionally unchanged: an earlier frozen-stage integrity test protects it. Stage 1.6 is an add-only study under `src/etrcm/stage1_6/`, `configs/stage1_6.yaml`, `experiments/*stage1_6*`, `results/stage1_6/` and `reports/*STAGE1_6*`.

> **Is ET-RCM failing to use persistent memory because its current integration architecture is incapable of doing so, or because the training process never forces the recurrent core to learn memory-dependent computation?**

> **ET-RCM 当前无法有效利用持久记忆，究竟是因为现有 memory-to-H integration 架构本身做不到，还是因为训练过程从未真正迫使 recurrent core 学会依赖 memory 进行计算？**

正式结果：G34 FAIL、G35 PASS、G36 FAIL、G37 FAIL。现有 gated-residual integration 在 oracle 训练下能利用正确历史读出（8/8 种子），但普通训练的 learned-read 与 slow-M 必要性仅 2/8 种子复制。B3 curriculum 的 learned-read 收益达 6/8 种子，不过主要是 fast-memory 效应；触发的辅助 F 课程未建立稳定 slow-M 依赖。旧目标延长训练显著降低 CE，却没有复制的 oracle-read 收益。不得把这些 toy 结果解释为语言模型或自主长期记忆已成功。

从 [完整报告](reports/STAGE1_6_FINAL_REPORT.md) 开始；[冻结协议](reports/STAGE1_6_PROTOCOL.md)、[辅助 F](reports/AUXILIARY_MEMORY_USE_CURRICULUM_STAGE1_6.md)、[旧目标延长训练](reports/TRAINING_LENGTH_SCALING_ORIGINAL_OBJECTIVE_STAGE1_6.md) 和 [机器可读 gate/种子效应](results/stage1_6/processed/stage1_6-formal-v1/gate_summary.json) 提供细节。982 项文件的哈希在 `results/stage1_6/processed/stage1_6-formal-v1/integrity.json`。

从服务器项目根目录运行：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python experiments/verify_stage1_6.py
```

复现实验入口为 `experiments/run_stage1_6_grid.py`（development/formal）、`run_stage1_6_legacy_grid.py`、`run_stage1_6_aux_F_grid.py`、`diagnose_stage1_6.py` 和 `analyze_stage1_6.py`。完成过的单元会被跳过；真正独立的重跑应使用隔离 checkout 和新的 run ID，不覆盖冻结数据。
