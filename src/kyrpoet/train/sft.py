"""Phase 4.2 — instruction tuning (QLoRA). See configs/sft.yaml.

Starts from the CPT checkpoint; trains instruction -> real poem using the base
model's chat template. Requires the ``train`` extra + GPU.
"""

from __future__ import annotations

import argparse

from .common import load_config


def _format_chat(tok, instruction: str, output: str) -> str:  # pragma: no cover
    return tok.apply_chat_template(
        [{"role": "user", "content": instruction},
         {"role": "assistant", "content": output}],
        tokenize=False,
    )


def resolve_init_from(config_value: str | None, override: str | None) -> str | None:
    """Which checkpoint to continue from.

    ``override`` of None keeps the config value; 'base'/'none'/'' means train from
    the base model (an SFT-only run); anything else is a checkpoint path.
    """
    if override is None:
        return config_value
    return None if override.lower() in ("base", "none", "") else override


def run(cfg) -> None:  # pragma: no cover - GPU + heavy deps
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    from .common import build_qlora_model

    model, tok = build_qlora_model(cfg)

    instr_f = cfg.data.get("instruction_field", "instruction")
    out_f = cfg.data.get("output_field", "output")
    ds = load_dataset("json", data_files={"train": cfg.data["train"],
                                          "val": cfg.data["val"]})
    ds = ds.map(lambda r: {"text": _format_chat(tok, r[instr_f], r[out_f])})

    args = SFTConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.train.get("lr", 1e-4),
        lr_scheduler_type=cfg.train.get("scheduler", "cosine"),
        num_train_epochs=cfg.train.get("epochs", 3),
        max_seq_length=cfg.train.get("seq_len", 1024),
        per_device_train_batch_size=cfg.train.get("batch_size", 4),
        gradient_accumulation_steps=cfg.train.get("grad_accum", 8),
        eval_steps=cfg.train.get("eval_steps", 100),
        save_steps=cfg.train.get("save_steps", 200),
        logging_steps=10,
        seed=cfg.seed,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, tokenizer=tok, args=args,
                         train_dataset=ds["train"], eval_dataset=ds["val"])
    trainer.train()
    trainer.save_model(cfg.output_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Instruction tuning (QLoRA)")
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to continue from (e.g. checkpoints/cpt); "
                         "use 'base' to train from the base model. Overrides the config.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.init_from = resolve_init_from(cfg.init_from, args.init_from)
    print(f"base_model={cfg.base_model}  init_from={cfg.init_from or '(base model)'}")
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
