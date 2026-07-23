"""Phase 4.3 — prosody-aware preference tuning (QLoRA DPO). See configs/dpo.yaml.

Where rhyme/meter quality is won. Preference pairs are built offline (see
data/build_datasets.make_preference_pair) by scoring SFT-model samples with
score_poem: higher `overall` = chosen. Requires the ``train`` extra + GPU.
"""

from __future__ import annotations

import argparse

from .common import load_config


def run(config_path: str) -> None:  # pragma: no cover - GPU + heavy deps
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    from .common import build_qlora_model

    cfg = load_config(config_path)
    model, tok = build_qlora_model(cfg)
    ds = load_dataset("json", data_files=cfg.data["train"], split="train")

    args = DPOConfig(
        output_dir=cfg.output_dir,
        beta=cfg.dpo.get("beta", 0.1),
        learning_rate=cfg.dpo.get("lr", 5e-6),
        lr_scheduler_type=cfg.dpo.get("scheduler", "cosine"),
        num_train_epochs=cfg.dpo.get("epochs", 1),
        max_length=cfg.dpo.get("seq_len", 1024),
        per_device_train_batch_size=cfg.dpo.get("batch_size", 2),
        gradient_accumulation_steps=cfg.dpo.get("grad_accum", 16),
        save_steps=cfg.dpo.get("save_steps", 200),
        logging_steps=10,
        seed=cfg.seed,
    )
    trainer = DPOTrainer(model=model, tokenizer=tok, train_dataset=ds, args=args)
    trainer.train()
    trainer.save_model(cfg.output_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preference tuning (QLoRA DPO)")
    ap.add_argument("--config", default="configs/dpo.yaml")
    run(ap.parse_args(argv).config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
