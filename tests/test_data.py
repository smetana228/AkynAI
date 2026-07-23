from kyrpoet.data.clean import (
    clean_poem,
    clean_stream,
    normalize,
    quality_flags,
    strip_prose_lines,
    text_hash,
)
from kyrpoet.data.ingest import make_id, raw_poem
from kyrpoet.llm.backend import FakeBackend, LLMBackend, OllamaBackend


# ---- backend ----
def test_fake_backend_records_and_replies():
    be = FakeBackend(reply="салам")
    assert isinstance(be, LLMBackend)  # runtime_checkable protocol
    assert be.complete("sys", "user") == "салам"
    assert be.calls == [("sys", "user")]


def test_fake_backend_fn():
    be = FakeBackend(fn=lambda s, u: u.upper())
    assert be.complete("s", "hi") == "HI"


def test_ollama_backend_builds_request_and_parses(monkeypatch):
    import io
    import json
    import urllib.request

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return io.BytesIO(json.dumps({"message": {"content": "салам"}}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    be = OllamaBackend(model="aya-expanse")
    assert isinstance(be, LLMBackend)  # satisfies the protocol
    assert be.complete("sys", "user") == "салам"
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["model"] == "aya-expanse"
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}


# ---- ingest ----
def test_raw_poem_schema_and_stable_id():
    p = raw_poem("ата\nэне", source="wiki", license="CC-BY-SA", title="t")
    assert set(p) == {"id", "source", "title", "author", "text", "license", "meta"}
    assert p["text"] == "ата\nэне"  # line breaks preserved
    assert p["id"] == make_id("wiki", "ата\nэне")  # deterministic


# ---- normalize ----
def test_normalize_crlf_and_whitespace():
    assert normalize("ата\r\n  эне  \tбала") == "ата\nэне бала"


def test_normalize_collapses_blank_runs():
    assert normalize("а\n\n\n\nб") == "а\n\nб"


def test_homoglyph_fix_only_in_cyrillic_tokens():
    # "аpа" mixes Latin p into a Cyrillic word -> repaired to Cyrillic р.
    assert normalize("аpа") == "ара"
    # a pure-Latin token is left alone
    assert normalize("cat") == "cat"


# ---- quality flags + dedup ----
def test_quality_flags_too_short_and_irregular():
    assert "too_short" in quality_flags("ата", [2])
    assert "irregular_meter" in quality_flags("a\nb", [2, 12])


def test_text_hash_ignores_case_and_spacing():
    assert text_hash("Ата  Эне") == text_hash("ата эне")


def test_strip_prose_lines_removes_long_lines_keeps_verse():
    verse = "Асман ачык жайнады"  # 7 syllables
    prose = "Илгерки өткөн заманда кыргыз эли катуу ачарчылыкка учурап калган " * 2
    text = f"{verse}\n{prose}\n{verse}"
    out, removed = strip_prose_lines(text, max_syllables=16)
    assert removed == 1
    assert prose.strip() not in out
    assert out == f"{verse}\n{verse}"


def test_strip_prose_lines_keeps_blank_stanza_breaks():
    text = "Асман ачык жайнады\n\nКуштар ырдап сайрады"
    out, removed = strip_prose_lines(text)
    assert removed == 0
    assert out == text


def test_clean_poem_reports_prose_removed():
    prose = "Илгерки өткөн заманда кыргыз эли катуу ачарчылыкка учурап калган эли " * 2
    raw = raw_poem(f"Асман ачык жайнады\n{prose}\nКуштар ырдап сайрады", "s", "l")
    cp = clean_poem(raw)
    assert cp["prose_lines_removed"] == 1
    assert cp["n_lines"] == 2  # prose line gone, two verse lines remain


def test_clean_poem_attaches_prosody_tags():
    raw = raw_poem("Асман ачык жайнады\nКуштар ырдап сайрады",
                   source="s", license="l")
    cp = clean_poem(raw)
    assert cp["n_lines"] == 2
    assert cp["syllables_per_line"] == [7, 7]
    assert "prosody" in cp and "rhyme_rate" in cp["prosody"]
    assert "quality_flags" in cp


def test_clean_stream_dedups_and_filters():
    good = raw_poem("Асман ачык жайнады\nКуштар ырдап сайрады", "s", "l")
    dup = raw_poem("асман  ачык   жайнады\nкуштар ырдап сайрады", "s2", "l")
    short = raw_poem("ата", "s", "l")
    kept, stats = clean_stream([good, dup, short])
    assert stats["kept"] == 1
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_too_short"] == 1
