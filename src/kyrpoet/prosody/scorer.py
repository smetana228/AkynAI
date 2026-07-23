"""Top-level prosody verifier: ``score_poem``.

Deterministic, pure-Python. Everything downstream (data filtering, DPO
preference pairs, rejection sampling, eval) calls this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .alliteration import alliteration_rate
from .rhyme import detect_rhyme_scheme, lines_rhyme
from .syllables import count_line_syllables

# Default Kyrgyz syllabic-meter target (7–8 syllable line).
DEFAULT_SYLLABLE_RANGE = (7, 8)

# Default weights for the `overall` combination.
DEFAULT_WEIGHTS = {"syllable": 0.5, "rhyme": 0.4, "alliteration": 0.1}


@dataclass
class PoemForm:
    """Optional target form for a poem.

    ``syllables`` is either an int (exact) or an inclusive ``(min, max)`` range.
    ``rhyme_scheme`` is a string like ``"aabb"`` / ``"abab"`` (``-`` = free line).
    Any field left None is simply not constrained.
    """

    n_lines: int | None = None
    syllables: int | tuple[int, int] | None = None
    rhyme_scheme: str | None = None

    def syllable_range(self) -> tuple[int, int]:
        if self.syllables is None:
            return DEFAULT_SYLLABLE_RANGE
        if isinstance(self.syllables, int):
            return (self.syllables, self.syllables)
        return self.syllables


@dataclass
class PoemScore:
    n_lines: int
    syllables_per_line: list[int]
    syllable_conformity: float
    detected_rhyme_scheme: str
    rhyme_rate: float
    alliteration_rate: float
    overall: float
    n_lines_ok: bool | None = None
    target: PoemForm | None = None
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def _split_lines(text: str) -> list[str]:
    """Non-empty, stripped content lines."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _syllable_conformity(counts: list[int], lo: int, hi: int) -> float:
    if not counts:
        return 0.0
    return sum(1 for c in counts if lo <= c <= hi) / len(counts)


def _rhyme_rate(lines: list[str], scheme: str, mode: str, tail: int) -> float:
    """Fraction of intended rhyme pairs (same scheme letter) that actually rhyme.

    Intended pairs come from ``scheme`` (a target scheme, or the detected one
    when no target is given). With no intended pairs the rate is 0.0.
    """
    # Group line indices by scheme letter, ignoring free-line markers.
    groups: dict[str, list[int]] = {}
    n = min(len(lines), len(scheme))
    for i in range(n):
        letter = scheme[i]
        if letter == "-":
            continue
        groups.setdefault(letter, []).append(i)

    intended = 0
    actual = 0
    for idxs in groups.values():
        for i, j in combinations(idxs, 2):
            intended += 1
            if lines_rhyme(lines[i], lines[j], mode, tail):
                actual += 1
    return actual / intended if intended else 0.0


def score_poem(
    text: str,
    target: PoemForm | None = None,
    *,
    rhyme_mode: str = "phonetic",
    rhyme_tail: int = 3,
    weights: dict[str, float] | None = None,
) -> PoemScore:
    """Score a poem's prosody against an optional target form."""
    weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
    lines = _split_lines(text)

    counts = [count_line_syllables(ln) for ln in lines]
    lo, hi = target.syllable_range() if target else DEFAULT_SYLLABLE_RANGE
    syllable_conformity = _syllable_conformity(counts, lo, hi)

    detected = detect_rhyme_scheme(lines, rhyme_mode, rhyme_tail)
    scheme_for_rate = target.rhyme_scheme if (target and target.rhyme_scheme) else detected
    rhyme_rate = _rhyme_rate(lines, scheme_for_rate, rhyme_mode, rhyme_tail)

    allit = alliteration_rate(lines)

    total_w = sum(weights.values()) or 1.0
    overall = (
        weights["syllable"] * syllable_conformity
        + weights["rhyme"] * rhyme_rate
        + weights["alliteration"] * allit
    ) / total_w

    n_lines_ok: bool | None = None
    if target and target.n_lines is not None:
        n_lines_ok = len(lines) == target.n_lines

    return PoemScore(
        n_lines=len(lines),
        syllables_per_line=counts,
        syllable_conformity=syllable_conformity,
        detected_rhyme_scheme=detected,
        rhyme_rate=rhyme_rate,
        alliteration_rate=allit,
        overall=overall,
        n_lines_ok=n_lines_ok,
        target=target,
        weights=weights,
    )
