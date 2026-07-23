"""Rhyme keys and rhyme-scheme detection.

Two modes:

* ``phonetic``   — final rime: substring from the last vowel to the end of the
  word (last vowel + trailing consonants). This is the standard end-rhyme test.
* ``suffix``/``grammatical`` — match on a shared trailing character sequence of
  configurable length. Captures Kyrgyz *grammatical rhyme*, where words sharing
  an agglutinative suffix rhyme by construction.

IMPORTANT (see README §4.3): vowel harmony means the "same" suffix surfaces with
different vowels (front vs back). Whether harmonic variants should count as
rhyming is **not** hard-coded here — it is a calibration decision that must be
validated against native-speaker judgments. Both modes are exposed so a chosen
threshold can be set empirically.
"""

from .vowels import is_vowel

# Characters stripped from the edges of a word before extracting a rhyme key.
_PUNCT = " \t\r\n.,;:!?—–-…\"'«»()[]{}"


def last_word(line: str) -> str:
    """Return the final word of a line, stripped of surrounding punctuation."""
    parts = line.strip().split()
    if not parts:
        return ""
    return parts[-1].strip(_PUNCT)


def _last_vowel_index(word: str) -> int:
    for i in range(len(word) - 1, -1, -1):
        if is_vowel(word[i]):
            return i
    return -1


def rhyme_key(word: str, mode: str = "phonetic", tail: int = 3) -> str:
    """Rhyme key for a single word (case-insensitive).

    ``phonetic``: from the last vowel to the end of the word.
    ``suffix``/``grammatical``: the last ``tail`` characters.
    A word with no vowels (phonetic) falls back to its full lowercased form.
    """
    word = word.lower()
    if mode == "phonetic":
        idx = _last_vowel_index(word)
        return word if idx == -1 else word[idx:]
    if mode in ("suffix", "grammatical"):
        return word[-tail:] if word else ""
    raise ValueError(f"unknown rhyme mode: {mode!r}")


def lines_rhyme(a: str, b: str, mode: str = "phonetic", tail: int = 3) -> bool:
    """Whether the last words of two lines rhyme under ``mode``."""
    wa, wb = last_word(a), last_word(b)
    if not wa or not wb:
        return False
    return rhyme_key(wa, mode, tail) == rhyme_key(wb, mode, tail)


def detect_rhyme_scheme(lines: list[str], mode: str = "phonetic", tail: int = 3) -> str:
    """Assign scheme letters (a, b, c, …) to distinct rhyme classes.

    Lines whose last word is empty get ``-`` (no rhyme participation).
    Letters recycle through the alphabet if there are more than 26 classes.
    """
    keys: list[str | None] = []
    for line in lines:
        w = last_word(line)
        keys.append(rhyme_key(w, mode, tail) if w else None)

    scheme: list[str] = []
    assigned: dict[str, str] = {}
    next_ord = 0
    for k in keys:
        if k is None:
            scheme.append("-")
            continue
        if k not in assigned:
            assigned[k] = chr(ord("a") + (next_ord % 26))
            next_ord += 1
        scheme.append(assigned[k])
    return "".join(scheme)
