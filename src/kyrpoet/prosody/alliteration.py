"""Line-initial (vertical) alliteration.

Traditional Kyrgyz verse links lines by the initial sound of the first word.
Convention: all vowels alliterate with one another; consonants must match
exactly.
"""

from .vowels import is_vowel

_PUNCT = " \t\r\n.,;:!?—–-…\"'«»()[]{}"

_VOWEL_CLASS = "◇"  # sentinel: any word starting with a vowel shares this class


def initial_sound(word: str) -> str:
    """Line-initial sound class of a word.

    Returns a single vowel-class sentinel if the word starts with a vowel,
    otherwise the lowercased first consonant. Empty for an empty word.
    """
    w = word.strip(_PUNCT).lower()
    if not w:
        return ""
    return _VOWEL_CLASS if is_vowel(w[0]) else w[0]


def _first_word(line: str) -> str:
    parts = line.strip().split()
    return parts[0] if parts else ""


def alliteration_rate(lines: list[str]) -> float:
    """Fraction of adjacent line pairs whose first words share an initial sound.

    Pairs where either line has no initial sound are skipped. With fewer than
    two scorable lines the rate is 0.0.
    """
    sounds = [initial_sound(_first_word(ln)) for ln in lines]
    pairs = 0
    hits = 0
    for a, b in zip(sounds, sounds[1:]):
        if not a or not b:
            continue
        pairs += 1
        if a == b:
            hits += 1
    return hits / pairs if pairs else 0.0
