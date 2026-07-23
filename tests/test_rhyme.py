from kyrpoet.prosody.rhyme import (
    detect_rhyme_scheme,
    lines_rhyme,
    last_word,
    rhyme_key,
)


def test_last_word_strips_punctuation():
    assert last_word("Мен сени сүйөм.") == "сүйөм"
    assert last_word("Кел, достум!") == "достум"
    assert last_word("") == ""


def test_phonetic_key_is_final_rime():
    # from last vowel to end of word
    assert rhyme_key("достум", "phonetic") == "ум"
    assert rhyme_key("сүйөм", "phonetic") == "өм"
    assert rhyme_key("бала", "phonetic") == "а"


def test_phonetic_rhyme_matches():
    assert lines_rhyme("кара кийген достум", "жакын турган кустум") is True
    # different final rime -> no rhyme
    assert lines_rhyme("ак бала", "кара таш") is False


def test_suffix_mode_grammatical_rhyme():
    # shared grammatical suffix -ди rhymes in suffix mode with tail=2
    assert lines_rhyme("сен келди", "мен берди", mode="suffix", tail=2) is True
    # harmony variant: rounded -дө vs unrounded -де do NOT match by raw suffix.
    # This is intentionally left to calibration, not hard-coded (README §4.3).
    assert lines_rhyme("сен келгенде", "мен көргөндө", mode="suffix", tail=2) is False


def test_suffix_tail_configurable():
    assert rhyme_key("келгенде", "suffix", tail=2) == "де"
    assert rhyme_key("келгенде", "suffix", tail=4) == "енде"


def test_detect_scheme_aabb():
    lines = [
        "жаз келди бизге",       # ...е (-> group a via last word "бизге" key "е")
        "гүлдөр ачылды тизге",   # "тизге" key "е" -> a
        "куштар сайрайт талда",  # "талда" key "а" -> b
        "жаңырат бүт талаа",     # "талаа" key "аа" -> c
    ]
    scheme = detect_rhyme_scheme(lines, mode="phonetic")
    assert scheme[0] == scheme[1]        # first two share rime
    assert scheme[2] != scheme[0]        # third distinct
    assert len(scheme) == 4


def test_detect_scheme_abab_shape():
    lines = ["бала", "таш", "жала", "кош"]
    # бала/жала share final rime "а"; таш/кош differ -> a b a c
    scheme = detect_rhyme_scheme(lines, mode="phonetic")
    assert scheme[0] == scheme[2]
    assert scheme[1] != scheme[3]


def test_empty_last_word_gets_free_marker():
    scheme = detect_rhyme_scheme(["бала", "!!!", "жала"], mode="phonetic")
    assert scheme[1] == "-"
