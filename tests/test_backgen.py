from kyrpoet.data.backgen import (
    build_prompt,
    filter_buckets,
    filter_covered,
    form_spec,
    make_examples,
    _assemble_from_texts,
    _echoes_poem,
    _looks_kyrgyz,
)
from kyrpoet.data.build_datasets import (
    filter_by_lines,
    make_preference_pair,
    poetry_repeats,
    split_examples,
)
from kyrpoet.llm.backend import FakeBackend

CP = {
    "id": "p1",
    "text": "Асман ачык жайнады\nКуштар ырдап сайрады",
    "n_lines": 2,
    "syllables_per_line": [7, 7],
    "detected_rhyme_scheme": "aa",
}


def test_form_spec_from_tags():
    fs = form_spec(CP)
    assert fs == {"n_lines": 2, "syllables": "7-7", "rhyme_scheme": "aa"}


def test_prompt_includes_form_only_when_explicit():
    _, plain = build_prompt(CP, form_explicit=False)
    _, explicit = build_prompt(CP, form_explicit=True)
    assert "муун" not in plain
    assert "муун" in explicit
    assert CP["text"] in plain  # poem provided to the model


def test_make_examples_two_styles_output_identical():
    be = FakeBackend(reply="Жаз жөнүндө кыска ыр жаз")
    exs = make_examples(be, CP)
    assert len(exs) == 2  # plain + form-explicit
    for ex in exs:
        assert ex["output"] == CP["text"]  # byte-identical
        assert ex["source_poem_id"] == "p1"


def test_guardrail_rejects_non_kyrgyz():
    be = FakeBackend(reply="write a poem about spring")
    assert make_examples(be, CP) == []


def test_guardrail_rejects_poem_echo():
    be = FakeBackend(reply="Мына ыр: Асман ачык жайнады")
    assert _echoes_poem("Мына ыр: Асман ачык жайнады", CP["text"]) is True
    assert make_examples(be, CP) == []


def test_looks_kyrgyz():
    assert _looks_kyrgyz("Көктөм жөнүндө жаз") is True
    assert _looks_kyrgyz("hello world") is False
    # a mixed English leak (common on local models) is rejected
    assert _looks_kyrgyz("Мекен жөнүндө аbout жаз") is False


def test_assemble_from_texts_maps_by_index_and_guards():
    poems = [CP, {"id": "p2", "text": "Жаз келди\nгүл ачты"}]
    texts = {
        "0-0": "Мекен жөнүндө ыр жаз",       # poem 0, plain -> kept
        "0-1": "hello world",                # poem 0, form -> rejected (not Kyrgyz)
        "1-0": "Жаратылыш жөнүндө ыр жаз",    # poem 1, plain -> kept
        # "1-1" missing (e.g. that request errored) -> skipped
    }
    out = _assemble_from_texts(poems, texts)
    assert [e["source_poem_id"] for e in out] == ["p1", "p2"]
    assert all(e["output"] in (CP["text"], poems[1]["text"]) for e in out)


def test_filter_covered_skips_already_generated():
    poems = [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}]
    kept = filter_covered(poems, {"p1", "p3"})
    assert [p["id"] for p in kept] == ["p2"]
    # empty covered set keeps everything
    assert len(filter_covered(poems, set())) == 3


def test_filter_buckets_excludes_manas():
    poems = [
        {"id": "a", "meta": {"bucket": "classic"}},
        {"id": "b", "meta": {"bucket": "manas"}},
        {"id": "c", "meta": {"bucket": "modern"}},
    ]
    kept = filter_buckets(poems, {"manas"})
    assert [p["id"] for p in kept] == ["a", "c"]
    # empty exclude set keeps everything
    assert len(filter_buckets(poems, set())) == 3


def test_split_examples_deterministic_and_disjoint():
    exs = [{"source_poem_id": f"id{i}", "instruction": "x"} for i in range(200)]
    t1, v1 = split_examples(exs, val_frac=0.1)
    t2, v2 = split_examples(exs, val_frac=0.1)
    assert (len(t1), len(v1)) == (len(t2), len(v2))  # deterministic
    ids_t = {e["source_poem_id"] for e in t1}
    ids_v = {e["source_poem_id"] for e in v1}
    assert ids_t.isdisjoint(ids_v)
    assert len(v1) > 0  # roughly 10%


def test_filter_by_lines():
    exs = [{"form": {"n_lines": 8}}, {"form": {"n_lines": 48}}, {"form": {"n_lines": 16}}]
    assert len(filter_by_lines(exs, 16)) == 2      # drops the 48-line one
    assert len(filter_by_lines(exs, None)) == 3    # None keeps all


def test_poetry_repeats_hits_target_share():
    # 2M poetry vs 40M general: ~15% target needs ~4x
    assert poetry_repeats(2_000_000, 40_000_000, 0.15) == 4
    # ~10% needs ~2x
    assert poetry_repeats(2_000_000, 40_000_000, 0.10) == 2
    # capped to guard against memorization
    assert poetry_repeats(1_000_000, 100_000_000, 0.50, max_repeats=10) == 10
    # no target / no general text -> no upsampling
    assert poetry_repeats(2_000_000, 40_000_000, None) == 1
    assert poetry_repeats(2_000_000, 0, 0.15) == 1


def test_preference_pair_respects_min_gap():
    a = {"text": "A", "overall": 0.9}
    b = {"text": "B", "overall": 0.5}
    pair = make_preference_pair("prompt", a, b, min_gap=0.15)
    assert pair == {"prompt": "prompt", "chosen": "A", "rejected": "B"}
    assert make_preference_pair("prompt", a, {"text": "C", "overall": 0.85}) is None
