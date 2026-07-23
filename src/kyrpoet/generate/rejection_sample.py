"""Phase 5 rejection sampling (§8.2): generate N, score, keep the best.

The selection logic is decoupled from the model: pass any callable that returns
a poem string given (topic, form). The recommended default inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..prosody.scorer import PoemForm, PoemScore, score_poem

# A generator maps (topic, form) -> a poem string.
Generator = Callable[[str, "PoemForm | None"], str]


@dataclass
class Candidate:
    text: str
    score: PoemScore


def best_of_n(
    generator: Generator,
    topic: str,
    form: PoemForm | None = None,
    n: int = 8,
    **score_kw,
) -> Candidate:
    """Generate ``n`` candidates, score each, return the highest ``overall``.

    Ties keep the first generated. Raises ValueError for n < 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    best: Candidate | None = None
    for _ in range(n):
        text = generator(topic, form)
        cand = Candidate(text=text, score=score_poem(text, form, **score_kw))
        if best is None or cand.score.overall > best.score.overall:
            best = cand
    assert best is not None
    return best
