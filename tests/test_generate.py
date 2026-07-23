import pytest

from kyrpoet.generate.generate import (
    base_model_from_adapter,
    build_prompt,
    check_checkpoint,
    tokenizer_source,
)
from kyrpoet.generate.rejection_sample import best_of_n
from kyrpoet.prosody.scorer import PoemForm

GOOD = "Асман ачык жайнады\nКуштар ырдап сайрады"  # 7,7 conform
BAD = "Мен\nсен"  # 1,1 syllables, no rhyme -> low overall


def test_build_prompt_plain():
    p = build_prompt("көктөм")
    assert "көктөм" in p
    assert "сап" not in p  # no form clauses


def test_build_prompt_with_form():
    p = build_prompt("көктөм", PoemForm(n_lines=4, syllables=(7, 8), rhyme_scheme="aabb"))
    assert "4 сап" in p
    assert "7-8 муун" in p
    assert "aabb" in p


def test_base_model_from_adapter(tmp_path):
    import json

    ckpt = tmp_path / "sft"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B"}), encoding="utf-8")
    assert base_model_from_adapter(str(ckpt)) == "Qwen/Qwen2.5-7B"
    # no adapter_config -> treat the path as a full model dir
    plain = tmp_path / "plain"
    plain.mkdir()
    assert base_model_from_adapter(str(plain)) == str(plain)


def test_tokenizer_source_prefers_checkpoint(tmp_path):
    ckpt = tmp_path / "sft"
    ckpt.mkdir()
    # no tokenizer saved -> fall back to the base model
    assert tokenizer_source(str(ckpt), "Qwen/Qwen2.5-7B") == "Qwen/Qwen2.5-7B"
    # tokenizer saved with the checkpoint -> use it (keeps the trained template)
    (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert tokenizer_source(str(ckpt), "Qwen/Qwen2.5-7B") == str(ckpt)


def test_check_checkpoint(tmp_path):
    assert check_checkpoint(str(tmp_path)) is None       # exists -> no problem
    msg = check_checkpoint(str(tmp_path / "missing"))
    assert msg and "checkpoint not found" in msg


def test_best_of_n_picks_highest_overall():
    seq = iter([BAD, GOOD, BAD])  # second candidate is best
    result = best_of_n(lambda t, f: next(seq), "тема", n=3)
    assert result.text == GOOD
    assert result.score.overall > 0.5


def test_best_of_n_validates_n():
    with pytest.raises(ValueError):
        best_of_n(lambda t, f: GOOD, "тема", n=0)


def test_best_of_n_passes_form_to_generator():
    seen = []
    best_of_n(lambda t, f: (seen.append(f) or GOOD), "тема",
              PoemForm(n_lines=2), n=1)
    assert seen[0].n_lines == 2
