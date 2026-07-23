"""Syllable counting.

Documented heuristic: syllable count == number of vowel *letters*. This is
exact enough for Kyrgyz because the orthography is near-phonemic.
"""

from .vowels import is_vowel


def count_syllables(word: str) -> int:
    """Number of vowel letters in ``word`` (case-insensitive)."""
    return sum(1 for ch in word if is_vowel(ch))


def _tokenize(line: str) -> list[str]:
    """Split a line into word-tokens, dropping punctuation and dashes.

    Keeps digits attached to any surrounding letters but they contribute no
    syllables (they contain no vowel letters); a standalone number simply
    counts as zero syllables.
    """
    tokens: list[str] = []
    current: list[str] = []
    for ch in line:
        if ch.isalnum():
            current.append(ch)
        else:
            # any non-alphanumeric (space, punctuation, dash) is a boundary
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def count_line_syllables(line: str) -> int:
    """Sum of syllables over the words in a line. Empty lines return 0."""
    return sum(count_syllables(tok) for tok in _tokenize(line))
