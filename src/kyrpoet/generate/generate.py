"""Phase 5 single-shot generation (§8.1): topic (+ optional form) -> poem.

``build_prompt`` is pure and testable; ``HFGenerator`` loads the checkpoint and
runs the chat template (needs the ``train`` extra + a GPU checkpoint).
"""

from __future__ import annotations

import argparse

from ..prosody.scorer import PoemForm


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
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base = self.base_model or self.checkpoint
        self._tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, device_map="auto",
                                                     load_in_4bit=True)
        self._model = PeftModel.from_pretrained(model, self.checkpoint)

    def __call__(self, topic: str, form: PoemForm | None = None) -> str:  # pragma: no cover
        self._load()
        prompt = build_prompt(topic, form)
        messages = [{"role": "user", "content": prompt}]
        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        out = self._model.generate(inputs, max_new_tokens=self.max_new_tokens,
                                   do_sample=True, temperature=0.9, top_p=0.95)
        return self._tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


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
