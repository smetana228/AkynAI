"""Phase 5 evaluation (§8.3).

Two tracks over a *fixed* eval prompt set:
  * Automatic — mean syllable_conformity / rhyme_rate / alliteration_rate and the
    distribution of ``overall`` (from the deterministic verifier).
  * LLM-as-judge — fluency / relevance / aesthetic on a fixed rubric. A proxy,
    NOT a substitute for native-speaker review.

Emits a markdown report and can sample poems into a CSV rating sheet for humans.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics

from ..jsonl import read_jsonl
from ..prosody.scorer import PoemForm, PoemScore, score_poem

_JUDGE_SYSTEM = (
    "You are a strict native-Kyrgyz poetry judge. Rate the poem 1-5 on three "
    "axes and reply with ONLY JSON: "
    '{"fluency":int,"relevance":int,"aesthetic":int}. '
    "fluency=natural Kyrgyz, relevance=matches the topic, aesthetic=imagery/quality."
)


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def aggregate_scores(scores: list[PoemScore]) -> dict:
    """Mean automatic metrics + `overall` distribution over a checkpoint's poems."""
    overalls = [s.overall for s in scores]
    return {
        "n": len(scores),
        "syllable_conformity": _mean([s.syllable_conformity for s in scores]),
        "rhyme_rate": _mean([s.rhyme_rate for s in scores]),
        "alliteration_rate": _mean([s.alliteration_rate for s in scores]),
        "overall_mean": _mean(overalls),
        "overall_min": min(overalls) if overalls else 0.0,
        "overall_max": max(overalls) if overalls else 0.0,
    }


def parse_judge(text: str) -> dict:
    """Parse the judge reply leniently: JSON first, then `axis: N` fallback."""
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return {k: int(obj[k]) for k in ("fluency", "relevance", "aesthetic")}
    except (ValueError, KeyError, TypeError):
        out = {}
        for axis in ("fluency", "relevance", "aesthetic"):
            m = re.search(rf"{axis}\D*(\d)", text, re.IGNORECASE)
            if m:
                out[axis] = int(m.group(1))
        return out


def judge_poem(backend, topic: str, poem: str) -> dict:
    user = f"Topic: {topic}\n\nPoem:\n{poem}"
    return parse_judge(backend.complete(_JUDGE_SYSTEM, user))


def _form_from_prompt(rec: dict) -> PoemForm | None:
    f = rec.get("form")
    if not f:
        return None
    syl = f.get("syllables")
    if isinstance(syl, str) and "-" in syl:
        syl = tuple(int(x) for x in syl.split("-"))
    return PoemForm(n_lines=f.get("n_lines"), syllables=syl,
                    rhyme_scheme=f.get("rhyme_scheme"))


def evaluate(prompts: list[dict], generator, backend=None) -> dict:
    """Generate + score each fixed prompt; optionally add LLM-judge ratings."""
    scores, judged, poems = [], [], []
    for rec in prompts:
        form = _form_from_prompt(rec)
        poem = generator(rec["topic"], form)
        scores.append(score_poem(poem, form))
        poems.append({"topic": rec["topic"], "poem": poem})
        if backend is not None:
            judged.append(judge_poem(backend, rec["topic"], poem))

    result = {"automatic": aggregate_scores(scores), "poems": poems}
    if backend is not None:
        result["judge"] = {
            axis: _mean([j[axis] for j in judged if axis in j])
            for axis in ("fluency", "relevance", "aesthetic")
        }
    return result


def render_report(result: dict, checkpoint: str) -> str:
    a = result["automatic"]
    lines = [
        f"# Eval report — `{checkpoint}`",
        "",
        f"Poems: **{a['n']}**",
        "",
        "## Automatic (verifier)",
        f"- syllable_conformity: **{a['syllable_conformity']:.3f}**",
        f"- rhyme_rate: **{a['rhyme_rate']:.3f}**",
        f"- alliteration_rate: **{a['alliteration_rate']:.3f}**",
        f"- overall: mean **{a['overall_mean']:.3f}** "
        f"(min {a['overall_min']:.3f}, max {a['overall_max']:.3f})",
    ]
    if "judge" in result:
        j = result["judge"]
        lines += [
            "",
            "## LLM-as-judge (proxy — not a substitute for native review)",
            f"- fluency: **{j['fluency']:.2f}** / 5",
            f"- relevance: **{j['relevance']:.2f}** / 5",
            f"- aesthetic: **{j['aesthetic']:.2f}** / 5",
        ]
    return "\n".join(lines) + "\n"


def write_rating_sheet(poems: list[dict], path: str) -> None:
    """Sample poems into a CSV for a native speaker to fill in (§8.3)."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["topic", "poem", "fluency_1_5", "rhyme_ok_0_1", "notes"])
        for p in poems:
            w.writerow([p["topic"], p["poem"], "", "", ""])


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI/infra
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint")
    ap.add_argument("--checkpoint", default="checkpoints/dpo")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--prompts", default="eval_prompts/prompts.jsonl")
    ap.add_argument("--report", default="runs/eval_report.md")
    ap.add_argument("--rating-sheet", default=None)
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge track")
    args = ap.parse_args(argv)

    from ..generate.generate import HFGenerator, check_checkpoint
    problem = check_checkpoint(args.checkpoint)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    generator = HFGenerator(args.checkpoint, args.base_model)
    backend = None
    if args.judge:
        from ..llm.backend import AnthropicBackend
        backend = AnthropicBackend()

    prompts = list(read_jsonl(args.prompts))
    result = evaluate(prompts, generator, backend)

    import os
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(render_report(result, args.checkpoint))
    print(render_report(result, args.checkpoint))
    if args.rating_sheet:
        write_rating_sheet(result["poems"], args.rating_sheet)
        print(f"Rating sheet -> {args.rating_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
