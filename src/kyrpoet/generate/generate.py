"""Phase 5 single-shot generation (§8.1): topic (+ optional form) -> poem.

``build_prompt`` is pure and testable; ``HFGenerator`` loads the checkpoint and
runs the chat template (needs the ``train`` extra + a GPU checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os

from ..prosody.scorer import PoemForm


def base_model_from_adapter(checkpoint: str) -> str:
    """Base model id recorded in a PEFT checkpoint's adapter_config.json.

    A LoRA checkpoint holds only adapter weights, so the base model has to come
    from somewhere; PEFT writes it into adapter_config.json at save time. Falls
    back to the checkpoint path itself (a full, non-adapter model directory).
    """
    cfg = os.path.join(checkpoint, "adapter_config.json")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as fh:
            base = json.load(fh).get("base_model_name_or_path")
        if base:
            return base
    return checkpoint


def tokenizer_source(checkpoint: str, base: str) -> str:
    """Prefer the checkpoint's tokenizer — it carries the chat template the
    model was actually trained with."""
    if os.path.exists(os.path.join(checkpoint, "tokenizer_config.json")):
        return checkpoint
    return base


def build_prompt(topic: str, form: PoemForm | None = None) -> str:
    """Kyrgyz user instruction for a topic, optionally with explicit form."""
    parts = [f"'{topic}' темасында кыргызча ыр жаз."]
    if form is not None:
        if form.n_lines is not None:
            parts.append(f"{form.n_lines} сап болсун.")
        lo, hi = form.syllable_range()
        parts.append(f"Ар бир сап {lo}-{hi} муундан турсун." if lo != hi
                     else f"Ар бир сап {lo} муундан турсун.")
        if form.rhyme_scheme:
            parts.append(f"Уйкаштык схемасы '{form.rhyme_scheme}' болсун.")
    return " ".join(parts)


class HFGenerator:
    """Loads a (QLoRA) checkpoint and generates via the base model's chat template."""

    def __init__(self, checkpoint: str, base_model: str | None = None,
                 max_new_tokens: int = 256):
        self.checkpoint = checkpoint
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tok = None

    def _load(self):  # pragma: no cover - requires GPU + weights
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        base = self.base_model or base_model_from_adapter(self.checkpoint)
        self._tok = AutoTokenizer.from_pretrained(tokenizer_source(self.checkpoint, base))
        # must match the quantization used in training (see train/common.py)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base, device_map="auto", quantization_config=bnb
        )
        self._model = PeftModel.from_pretrained(model, self.checkpoint)

    def __call__(self, topic: str, form: PoemForm | None = None) -> str:  # pragma: no cover
        self._load()
        prompt = build_prompt(topic, form)
        messages = [{"role": "user", "content": prompt}]
        # Render to text then tokenize: apply_chat_template's tensor return type
        # changed across transformers releases (tensor vs BatchEncoding).
        text = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        enc = self._tok(text, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
        )
        generated = out[0][enc["input_ids"].shape[1]:]
        return self._tok.decode(generated, skip_special_tokens=True).strip()


def check_checkpoint(path: str) -> str | None:
    """Error message if ``path`` isn't a local checkpoint directory, else None.

    Without this, transformers treats the path as a Hub repo id and fails with
    an opaque 401 instead of saying the directory is missing.
    """
    if os.path.isdir(path):
        return None
    import glob

    found = sorted(d for d in glob.glob("checkpoints/*") if os.path.isdir(d))
    msg = f"checkpoint not found: {path}"
    return msg + (f"\navailable: {', '.join(found)}" if found
                  else "\nno checkpoints/ directory — train a model first")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI/infra
    ap = argparse.ArgumentParser(description="Generate a Kyrgyz poem")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/dpo")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--n-lines", type=int, default=None)
    ap.add_argument("--syllables", default=None, help="e.g. 7-8 or 7")
    ap.add_argument("--rhyme-scheme", default=None)
    ap.add_argument("--best-of", type=int, default=1)
    args = ap.parse_args(argv)

    problem = check_checkpoint(args.checkpoint)
    if problem:
        import sys
        print(problem, file=sys.stderr)
        return 1

    syllables = None
    if args.syllables:
        syllables = (tuple(int(x) for x in args.syllables.split("-"))
                     if "-" in args.syllables else int(args.syllables))
    form = None
    if args.n_lines or syllables or args.rhyme_scheme:
        form = PoemForm(n_lines=args.n_lines, syllables=syllables,
                        rhyme_scheme=args.rhyme_scheme)

    gen = HFGenerator(args.checkpoint, args.base_model)
    if args.best_of > 1:
        from .rejection_sample import best_of_n
        result = best_of_n(gen, args.topic, form, n=args.best_of)
        print(result.text)
        print(f"\n[overall={result.score.overall:.3f}]")
    else:
        print(gen(args.topic, form))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
