from kyrpoet.eval.evaluate import (
    aggregate_scores,
    evaluate,
    parse_judge,
    render_report,
)
from kyrpoet.llm.backend import FakeBackend
from kyrpoet.prosody.scorer import score_poem

GOOD = "Асман ачык жайнады\nКуштар ырдап сайрады"
BAD = "Мен\nсен"


def test_aggregate_scores_means_and_range():
    scores = [score_poem(GOOD), score_poem(BAD)]
    agg = aggregate_scores(scores)
    assert agg["n"] == 2
    assert 0.0 <= agg["overall_min"] <= agg["overall_mean"] <= agg["overall_max"] <= 1.0
    assert agg["overall_max"] > agg["overall_min"]


def test_aggregate_empty():
    agg = aggregate_scores([])
    assert agg["n"] == 0 and agg["overall_mean"] == 0.0


def test_parse_judge_json():
    assert parse_judge('{"fluency":4,"relevance":5,"aesthetic":3}') == {
        "fluency": 4, "relevance": 5, "aesthetic": 3}


def test_parse_judge_json_with_prose_wrapper():
    txt = 'Here: {"fluency":2,"relevance":3,"aesthetic":4} done'
    assert parse_judge(txt) == {"fluency": 2, "relevance": 3, "aesthetic": 4}


def test_parse_judge_fallback_lines():
    txt = "fluency: 5\nrelevance - 4\naesthetic = 2"
    assert parse_judge(txt) == {"fluency": 5, "relevance": 4, "aesthetic": 2}


def test_evaluate_end_to_end_with_fake_judge():
    prompts = [
        {"topic": "көктөм", "form": {"n_lines": 2, "syllables": "7-8", "rhyme_scheme": "aa"}},
        {"topic": "күз", "form": None},
    ]
    backend = FakeBackend(reply='{"fluency":4,"relevance":4,"aesthetic":4}')
    result = evaluate(prompts, generator=lambda t, f: GOOD, backend=backend)
    assert result["automatic"]["n"] == 2
    assert result["judge"]["fluency"] == 4.0
    assert len(result["poems"]) == 2


def test_render_report_contains_metrics():
    result = evaluate([{"topic": "күз", "form": None}], lambda t, f: GOOD)
    report = render_report(result, "checkpoints/dpo")
    assert "Eval report" in report
    assert "syllable_conformity" in report
    assert "judge" not in result  # no backend -> no judge track
