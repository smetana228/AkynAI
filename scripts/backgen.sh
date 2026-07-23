#!/usr/bin/env bash
# Phase 3: back-generate instructions (needs ANTHROPIC_API_KEY + .[llm]).
set -euo pipefail
python -m kyrpoet.data.backgen "$@"
python -m kyrpoet.data.build_datasets sft
