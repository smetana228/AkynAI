#!/usr/bin/env bash
# Phase 0: tokenizer fertility probe.
set -euo pipefail
python -m kyrpoet.tokenizer_probe --corpus "${1:-data/sample_ky.txt}"
