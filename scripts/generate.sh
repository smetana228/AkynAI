#!/usr/bin/env bash
# Phase 5: generate a poem (best-of-N rejection sampling recommended).
set -euo pipefail
python -m kyrpoet.generate.generate "$@"
