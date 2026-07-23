"""Kyrgyz Cyrillic vowel sets and harmony classification.

Kyrgyz orthography is near-phonemic: each vowel *letter* is exactly one
syllable nucleus. This is the fact that makes meter and rhyme checkable from
spelling alone, without a pronunciation dictionary.
"""

# The twelve Kyrgyz Cyrillic vowel letters. Each counts as exactly one nucleus.
# Note: `й` is a consonant (glide), not a vowel; `ъ`/`ь` (Russian loanwords)
# are not vowels. `я ю ё е` are iotated but still one nucleus each.
VOWELS = frozenset("аеёиоөуүыэюя")

# Harmony classes (used by the optional harmony metric and rhyme calibration).
# Iotated/Russian-loan vowels (е э ю я ё и) are assigned by their nuclear vowel
# sound for harmony purposes; `и` is treated as front per Kyrgyz convention.
BACK_VOWELS = frozenset("аоуы")
FRONT_VOWELS = frozenset("еэөүи")

ROUNDED_VOWELS = frozenset("оөуү")
UNROUNDED_VOWELS = frozenset("аеэыи")


def is_vowel(ch: str) -> bool:
    """True if ``ch`` is a Kyrgyz vowel letter (case-insensitive)."""
    return ch.lower() in VOWELS


def is_back(ch: str) -> bool:
    return ch.lower() in BACK_VOWELS


def is_front(ch: str) -> bool:
    return ch.lower() in FRONT_VOWELS


def is_rounded(ch: str) -> bool:
    return ch.lower() in ROUNDED_VOWELS
