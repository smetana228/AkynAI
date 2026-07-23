"""Phase 0 — tokenizer fertility probe.

For each candidate tokenizer, measure how efficiently it encodes Kyrgyz text.
Fertility (tokens / whitespace-word) is the headline metric — lower is better.
Poor subword tokenization disproportionately hurts rhyme, because rhyme lives in
the final characters that greedy merges tend to bury.

Usage:
    python -m kyrpoet.tokenizer_probe --corpus data/sample_ky.txt

Requires the optional `probe` extra:  pip install -e ".[probe]"
"""

from __future__ import annotations

import argparse
import sys

# Candidate base models (README §1). Default is Qwen2.5-7B unless its fertility
# is clearly worse than an alternative (a common failure mode on Cyrillic).
CANDIDATES = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Gemma-2-9B": "google/gemma-2-9b",
    "Aya-23-8B": "CohereForAI/aya-23-8B",
}
DEFAULT_MODEL = "Qwen2.5-7B"


def _load_words(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return text.split()


def probe_tokenizer(model_id: str, words: list[str], n_examples: int = 5):
    """Return fertility stats for one tokenizer, or None if it can't be loaded."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is required: pip install -e '.[probe]'", file=sys.stderr)
        raise

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # network / gated model / missing auth
        print(f"  ! could not load {model_id}: {exc}", file=sys.stderr)
        return None

    total_tokens = 0
    single_token_words = 0
    examples = []
    for i, w in enumerate(words):
        ids = tok.encode(w, add_special_tokens=False)
        total_tokens += len(ids)
        if len(ids) == 1:
            single_token_words += 1
        if i < n_examples:
            pieces = tok.convert_ids_to_tokens(ids)
            examples.append((w, pieces))

    n_words = len(words)
    return {
        "fertility": total_tokens / n_words if n_words else 0.0,
        "single_token_pct": 100.0 * single_token_words / n_words if n_words else 0.0,
        "total_tokens": total_tokens,
        "n_words": n_words,
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kyrgyz tokenizer fertility probe")
    ap.add_argument("--corpus", required=True, help="path to a Kyrgyz text sample")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args(argv)

    try:
        import transformers  # noqa: F401
    except ImportError:
        print("transformers is required: pip install -e '.[probe]'", file=sys.stderr)
        return 1

    words = _load_words(args.corpus)
    if not words:
        print("corpus is empty", file=sys.stderr)
        return 1
    print(f"Corpus: {args.corpus}  ({len(words)} whitespace-words)\n")

    results = {}
    for name, model_id in CANDIDATES.items():
        print(f"Probing {name} ({model_id}) ...")
        stats = probe_tokenizer(model_id, words, args.examples)
        if stats is not None:
            results[name] = stats

    if not results:
        print("No tokenizer could be loaded. Check network / HF auth.", file=sys.stderr)
        return 1

    # Ranked table (best fertility first).
    ranked = sorted(results.items(), key=lambda kv: kv[1]["fertility"])
    print("\n" + "=" * 60)
    print(f"{'model':<14}{'fertility':>12}{'1-token %':>14}")
    print("-" * 60)
    for name, s in ranked:
        print(f"{name:<14}{s['fertility']:>12.3f}{s['single_token_pct']:>13.1f}%")
    print("=" * 60)

    for name, s in ranked:
        print(f"\n{name} example splits:")
        for w, pieces in s["examples"]:
            print(f"  {w!r:<16} -> {pieces}")

    # Recommendation: default to Qwen unless it is clearly worse than the best.
    best_name, best = ranked[0]
    choice = DEFAULT_MODEL
    if DEFAULT_MODEL in results:
        qwen_fert = results[DEFAULT_MODEL]["fertility"]
        # "clearly worse" = >15% higher fertility than the best alternative.
        if best_name != DEFAULT_MODEL and qwen_fert > 1.15 * best["fertility"]:
            choice = best_name
    else:
        choice = best_name

    print(f"\nRECOMMENDED BASE MODEL: {choice} ({CANDIDATES[choice]})")
    print("Record this choice in the README (§3 acceptance).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
