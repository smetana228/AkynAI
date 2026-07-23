#!/usr/bin/env bash
# Phase 2: normalize + prosody-tag + filter -> CleanPoem JSONL.
set -euo pipefail
python -m kyrpoet.data.clean "$@"
