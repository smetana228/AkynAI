import pytest

from kyrpoet.prosody.scorer import PoemForm, score_poem

# A hand-crafted 7–8 syllable quatrain, aabb rhyme.
POEM = """\
Асман ачык жайнады
Куштар ырдап сайрады
Гүлдөр өсөт талаада
Дарак турат кырманда
"""

# 12 hand-checked real-style lines: (line, syllable count by vowel-letter rule).
HAND_CHECKED_LINES = [
    ("Асман ачык жайнады", 7),
    ("Куштар ырдап сайрады", 7),
    ("Гүлдөр өсөт талаада", 8),
    ("Дарак турат кырманда", 7),
    ("Мен сени сүйөм", 5),
    ("Жаз келди бизге", 5),
    ("Ак кар жаайт тынымсыз", 7),
    ("Тоолор бийик, кар баскан", 8),
    ("Суу агып жатат ылдамдап", 9),
    ("Күн батты тоонун артына", 9),
    ("Балдар ойноп жүрүшөт", 7),
    ("Эне тилим — алтын кенчим", 8),
]


@pytest.mark.parametrize("line,expected", HAND_CHECKED_LINES)
def test_hand_checked_line_syllables(line, expected):
    from kyrpoet.prosody.syllables import count_line_syllables

    assert count_line_syllables(line) == expected


def test_score_poem_populated():
    score = score_poem(POEM)
    assert score.n_lines == 4
    assert score.syllables_per_line == [7, 7, 8, 7]
    assert score.syllable_conformity == 1.0  # all within default 7–8
    assert score.detected_rhyme_scheme == "aabb"
    assert score.rhyme_rate == 1.0
    assert score.alliteration_rate == 0.0
    assert 0.0 <= score.overall <= 1.0


def test_overall_weighting():
    score = score_poem(POEM)
    # syllable 0.5*1 + rhyme 0.4*1 + allit 0.1*0 = 0.9
    assert abs(score.overall - 0.9) < 1e-9


def test_target_form_n_lines_and_range():
    target = PoemForm(n_lines=4, syllables=(7, 8), rhyme_scheme="aabb")
    score = score_poem(POEM, target)
    assert score.n_lines_ok is True
    assert score.syllable_conformity == 1.0
    assert score.rhyme_rate == 1.0  # intended aabb pairs all rhyme


def test_target_rhyme_scheme_mismatch_lowers_rate():
    # Ask for abab, but the poem is aabb: intended pairs (0,2) and (1,3) don't rhyme.
    target = PoemForm(rhyme_scheme="abab")
    score = score_poem(POEM, target)
    assert score.rhyme_rate == 0.0


def test_syllable_conformity_partial():
    text = "Мен сени сүйөм\nАсман ачык жайнады"  # 5, 7
    score = score_poem(text)  # default 7–8
    assert score.syllables_per_line == [5, 7]
    assert score.syllable_conformity == 0.5


def test_exact_syllable_target():
    score = score_poem(POEM, PoemForm(syllables=7))
    # line 3 has 8 syllables -> 3/4 conform to exactly 7
    assert score.syllable_conformity == 0.75


def test_empty_poem():
    score = score_poem("")
    assert score.n_lines == 0
    assert score.syllable_conformity == 0.0
    assert score.rhyme_rate == 0.0
    assert score.overall == 0.0


def test_custom_weights():
    w = {"syllable": 1.0, "rhyme": 0.0, "alliteration": 0.0}
    score = score_poem(POEM, weights=w)
    assert abs(score.overall - 1.0) < 1e-9  # pure syllable conformity
