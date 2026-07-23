"""Phase 2 cleaning + tagging (§5.3): RawPoem JSONL -> CleanPoem JSONL.

Normalizes text, attaches prosody tags from the Phase-1 scorer, flags quality
issues, and drops garbled / irregular / too-short / duplicate poems.
"""

from __future__ import annotations

import argparse
import hashlib
import unicodedata
from collections import Counter

from ..jsonl import read_jsonl, write_jsonl
from ..prosody.scorer import score_poem
from ..prosody.syllables import count_line_syllables

# Latin -> Cyrillic homoglyph map for OCR confusions (§5.3). Applied only inside
# tokens that already contain Cyrillic, so genuine Latin text is left alone.
_HOMOGLYPHS = {
    "a": "а", "o": "о", "e": "е", "c": "с", "p": "р", "x": "х", "y": "у",
    "A": "А", "O": "О", "E": "Е", "C": "С", "P": "Р", "X": "Х", "H": "Н",
    "B": "В", "K": "К", "M": "М", "T": "Т",
}
_CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяөүңАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯӨҮҢ")

MIN_LINES = 2
MIN_CYRILLIC_RATIO = 0.6
# "wildly irregular": spread of syllable counts across content lines.
MAX_SYLLABLE_SPREAD = 6
# A "line" longer than this is prose (a collapsed paragraph or run-together
# verse), not a 7–8 syllable verse line, and is stripped out.
MAX_LINE_SYLLABLES = 16


def _fix_homoglyphs_token(tok: str) -> str:
    if not any(ch in _CYRILLIC for ch in tok):
        return tok  # pure-Latin token: leave as-is
    return "".join(_HOMOGLYPHS.get(ch, ch) for ch in tok)


def normalize(text: str) -> str:
    """NFC, homoglyph repair, whitespace + line-break standardization."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out_lines = []
    for line in text.split("\n"):
        toks = line.split()
        toks = [_fix_homoglyphs_token(t) for t in toks]
        out_lines.append(" ".join(toks))
    # collapse 3+ consecutive blank lines to one, strip leading/trailing blanks
    collapsed: list[str] = []
    blank_run = 0
    for ln in out_lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append(ln)
        else:
            blank_run = 0
            collapsed.append(ln)
    return "\n".join(collapsed).strip("\n")


def _cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch in _CYRILLIC) / len(letters)


def text_hash(text: str) -> str:
    """Hash of normalized, case-folded, whitespace-collapsed text (dup key)."""
    key = " ".join(text.lower().split())
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def quality_flags(text: str, syllables_per_line: list[int]) -> list[str]:
    flags = []
    content = [ln for ln in text.splitlines() if ln.strip()]
    if len(content) < MIN_LINES:
        flags.append("too_short")
    if _cyrillic_ratio(text) < MIN_CYRILLIC_RATIO:
        flags.append("low_cyrillic")
    if syllables_per_line:
        spread = max(syllables_per_line) - min(syllables_per_line)
        if spread > MAX_SYLLABLE_SPREAD:
            flags.append("irregular_meter")
        if any(c == 0 for c in syllables_per_line):
            flags.append("empty_or_garbled_line")
    return flags


def strip_prose_lines(text: str, max_syllables: int = MAX_LINE_SYLLABLES) -> tuple[str, int]:
    """Drop prose lines (syllable count above ``max_syllables``).

    Blank lines are kept (they carry stanza structure). Returns the filtered
    text and the number of prose lines removed.
    """
    kept, removed = [], 0
    for ln in text.split("\n"):
        if ln.strip() and count_line_syllables(ln) > max_syllables:
            removed += 1
        else:
            kept.append(ln)
    return "\n".join(kept).strip("\n"), removed


def clean_poem(raw: dict) -> dict:
    """Return a CleanPoem dict (RawPoem fields + prosody tags + flags)."""
    text = normalize(raw["text"])
    text, prose_removed = strip_prose_lines(text)
    score = score_poem(text)
    flags = quality_flags(text, score.syllables_per_line)
    return {
        **raw,
        "text": text,
        "n_lines": score.n_lines,
        "syllables_per_line": score.syllables_per_line,
        "detected_rhyme_scheme": score.detected_rhyme_scheme,
        "prosody": {
            "syllable_conformity": score.syllable_conformity,
            "rhyme_rate": score.rhyme_rate,
            "alliteration_rate": score.alliteration_rate,
        },
        "quality_flags": flags,
        "prose_lines_removed": prose_removed,
    }


def clean_stream(raws, drop_flags=("too_short", "low_cyrillic", "empty_or_garbled_line")):
    """Clean + de-duplicate + filter. Returns (kept, stats)."""
    seen: set[str] = set()
    kept: list[dict] = []
    stats = Counter()
    for raw in raws:
        stats["seen"] += 1
        cp = clean_poem(raw)
        h = text_hash(cp["text"])
        if h in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(h)
        bad = [f for f in cp["quality_flags"] if f in drop_flags]
        if bad:
            stats[f"dropped_{bad[0]}"] += 1
            continue
        kept.append(cp)
        stats["kept"] += 1
    return kept, stats


def _print_summary(kept: list[dict], stats: Counter) -> None:
    print("\n=== clean summary ===")
    for k in sorted(stats):
        print(f"  {k:<28} {stats[k]}")
    if kept:
        line_counts = Counter(p["n_lines"] for p in kept)
        syl = Counter(c for p in kept for c in p["syllables_per_line"])
        print("  lines-per-poem distribution:",
              dict(sorted(line_counts.items())))
        print("  syllables-per-line distribution:",
              dict(sorted(syl.items())))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clean + tag RawPoem -> CleanPoem")
    ap.add_argument("--in", dest="inp", default="data/raw/poems.jsonl")
    ap.add_argument("--out", default="data/clean/poems.jsonl")
    args = ap.parse_args(argv)

    kept, stats = clean_stream(read_jsonl(args.inp))
    n = write_jsonl(args.out, kept)
    _print_summary(kept, stats)
    print(f"\nWrote {n} CleanPoem records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
