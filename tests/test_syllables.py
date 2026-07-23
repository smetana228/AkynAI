import pytest

from kyrpoet.prosody.syllables import count_line_syllables, count_syllables

# Required fixtures from README §4.2.
@pytest.mark.parametrize(
    "word,expected",
    [
        ("ата", 2),
        ("алма", 2),
        ("мектеп", 2),
        ("Кыргызстан", 3),
        ("аю", 2),
    ],
)
def test_required_fixtures(word, expected):
    assert count_syllables(word) == expected


def test_case_insensitive():
    assert count_syllables("АТА") == count_syllables("ата") == 2


def test_i_kratkoye_is_consonant():
    # `й` is a glide/consonant, not a vowel nucleus.
    assert count_syllables("май") == 1  # vowels: а
    assert count_syllables("айыл") == 2  # vowels: а, ы  (й does not count)


def test_hard_soft_signs_not_vowels():
    # ъ/ь appear only in Russian loanwords and are not vowels.
    assert count_syllables("объект") == count_syllables("обект")


def test_line_strips_punctuation_and_dashes():
    assert count_line_syllables("Ата — алма!") == 4  # ата(2) + алма(2)


def test_empty_line_is_zero():
    assert count_line_syllables("") == 0
    assert count_line_syllables("   \t ") == 0


def test_digits_contribute_no_syllables():
    assert count_line_syllables("2025 ата") == 2


def test_iotated_vowels_one_nucleus_each():
    # я, ю, ё, е each count as exactly one.
    assert count_syllables("аю") == 2
    assert count_syllables("яя") == 2
