"""Phase 4.1 — continued pretraining (QLoRA, plain LM). See configs/cpt.yaml.

Raises base-language fluency and absorbs poetic register BEFORE instruction
tuning, since fluency is the quality ceiling. Requires the ``train`` extra + GPU.
"""

from __future__ import annotations

import argparse

from .common import load_config


def run(config_path: str) -> None:  # pragma: no cover - GPU + heavy deps
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    from .common import build_qlora_model

    cfg = load_config(config_path)
    model, tok = build_qlora_model(cfg)
    ds = load_dataset("json", data_files=cfg.data["train"], split="train")
    text_field = cfg.data.get("text_field", "text")

    from .common import pick_kwarg, supported_kwargs

    desired = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.train.get("lr", 2e-4),
        "lr_scheduler_type": cfg.train.get("scheduler", "cosine"),
        "warmup_ratio": cfg.train.get("warmup_ratio", 0.03),
        "num_train_epochs": cfg.train.get("epochs", 1),
        "per_device_train_batch_size": cfg.train.get("batch_size", 2),
        "gradient_accumulation_steps": cfg.train.get("grad_accum", 16),
        "save_steps": cfg.train.get("save_steps", 500),
        "logging_steps": 10,
        "seed": cfg.seed,
        "dataset_text_field": text_field,
    }
    desired.update(pick_kwarg(SFTConfig, ["max_length", "max_seq_length"],
                              cfg.train.get("seq_len", 2048)))
    args = SFTConfig(**supported_kwargs(SFTConfig, desired))

    tok_kw = pick_kwarg(SFTTrainer, ["processing_class", "tokenizer"], tok)
    trainer = SFTTrainer(model=model, train_dataset=ds, args=args, **tok_kw)
    trainer.train()
    trainer.save_model(cfg.output_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Continued pretraining (QLoRA)")
    ap.add_argument("--config", default="configs/cpt.yaml")
    run(ap.parse_args(argv).config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
