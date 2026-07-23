#!/usr/bin/env bash
# Phase 4: QLoRA CPT -> SFT -> DPO (needs .[train] + GPU).
set -euo pipefail
python -m kyrpoet.train.cpt --config configs/cpt.yaml
# chain SFT off the CPT checkpoint (sft.yaml defaults to base for SFT-only runs)
python -m kyrpoet.train.sft --config configs/sft.yaml --init-from checkpoints/cpt
python -m kyrpoet.train.dpo --config configs/dpo.yaml
