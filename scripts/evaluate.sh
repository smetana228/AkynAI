#!/usr/bin/env bash
# Phase 5: eval a checkpoint on the fixed prompt set (+ --judge for LLM track).
set -euo pipefail
python -m kyrpoet.eval.evaluate "$@"
