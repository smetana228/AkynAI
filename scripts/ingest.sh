#!/usr/bin/env bash
# Phase 2: raw .txt poems -> RawPoem JSONL.  usage: ingest.sh <dir> <source> <license>
set -euo pipefail
python -m kyrpoet.data.ingest --input-dir "$1" --source "$2" --license "$3"
