from kyrpoet.prosody.alliteration import alliteration_rate, initial_sound


def test_initial_consonant_exact():
    assert initial_sound("кара") == "к"
    assert initial_sound("Кел,") == "к"  # case + punctuation


def test_all_vowels_share_a_class():
    assert initial_sound("ата") == initial_sound("өмүр")
    assert initial_sound("ата") != initial_sound("кара")


def test_empty_word():
    assert initial_sound("") == ""
    assert initial_sound("!!!") == ""


def test_full_alliteration():
    lines = ["кара кийген", "куш конду", "кел бери", "кыз бала"]
    # all start with к -> every adjacent pair matches
    assert alliteration_rate(lines) == 1.0


def test_no_alliteration():
    lines = ["ата", "бала", "чоң", "дос"]
    assert alliteration_rate(lines) == 0.0


def test_vowel_lines_alliterate():
    lines = ["ата эне", "өмүр бою", "улуу тоо"]
    # all start with vowels -> alliterate under the convention
    assert alliteration_rate(lines) == 1.0


def test_partial_rate():
    lines = ["кара", "куш", "бала", "берг"]
    # pairs: (к,к)=hit, (к,б)=miss, (б,б)=hit -> 2/3
    assert abs(alliteration_rate(lines) - 2 / 3) < 1e-9


def test_single_line_is_zero():
    assert alliteration_rate(["кара"]) == 0.0
    assert alliteration_rate([]) == 0.0
