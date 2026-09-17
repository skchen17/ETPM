#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m pytest
"${PYTHON_BIN}" experiments/run_toy.py --toy all --config configs/toy_default.yaml
"${PYTHON_BIN}" experiments/analyze_toys.py --tests-passed

