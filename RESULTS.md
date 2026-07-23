# Results

Measured with the Phase-1 prosody verifier (`score_poem`) — default weights:
syllable 0.5, rhyme 0.4, alliteration 0.1; default target 7–8 syllables/line.

## Reference: what human Kyrgyz poetry scores

Scoring the cleaned corpus itself (1,624 poems, Manas excluded) gives the target
to aim at. **1.0 is not the goal — 0.616 is.**

| Metric | Human corpus |
|---|---:|
| syllable_conformity | 0.394 |
| rhyme_rate | 0.951 |
| alliteration_rate | 0.389 |
| **overall** | **0.616** |

Note the shape: real Kyrgyz verse rhymes very consistently (0.951) but meets a
strict 7–8 syllable window only ~39% of the time — much of the corpus is 11–13
syllable verse. Rhyme, not meter, is the reliable signal in this tradition.

## Run 1 — SFT-only baseline (2026-07-23)

Cheapest possible validation of the pipeline: no continued pretraining.

| | |
|---|---|
| Base model | Qwen2.5-7B (4-bit QLoRA, LoRA r=32) |
| Init from | base model (no CPT) |
| Train / val | 1,312 / 59 examples (poems capped at 48 lines) |
| Steps | 123 (3 epochs) |
| Hardware | 1× RTX 4090 24GB, ~30 min, ~$0.15 |

Evaluated on the committed 10-prompt set (`eval_prompts/prompts.jsonl`),
single-shot generation:

| Metric | Human | SFT | % of human |
|---|---:|---:|---:|
| syllable_conformity | 0.394 | 0.332 | 84% |
| rhyme_rate | 0.951 | 0.183 | **19%** |
| alliteration_rate | 0.389 | 0.147 | 38% |
| **overall** | **0.616** | **0.254** | **41%** |

`overall` spread: min 0.058, max 0.666.

### Findings

1. **Rhyme is the dominant gap.** Meter is already at 84% of human; rhyme is at
   19%. At 0.4 weight it accounts for almost the whole deficit.
2. **Rejection sampling helps a lot.** Single-shot averages 0.254, but a
   best-of-8 sample reached 0.630 — roughly human-level. The model *can*
   produce good candidates; it just doesn't do so reliably. That is exactly the
   gap DPO is meant to close.
3. **Form control is not learned.** Asked for 4 lines, the model produced ~18 —
   the median length of the SFT training set. It followed the data distribution
   rather than the instruction.
4. **Fluency is thin.** Output is recognisably Kyrgyz but semantically weak and
   often off-topic; it also looped whole lines (since mitigated with a
   repetition penalty).

### Next

DPO looks like the higher-value spend before CPT: it optimises directly against
this verifier, targets rhyme (the weakest metric), and finding 2 shows the
headroom is already there. CPT remains worthwhile for fluency but costs 3–4×
more GPU time and will not move rhyme much.

**Caveat:** these are automatic metrics only. Per the spec (§4.3, §8.3) the
rhyme thresholds still need calibration against native-speaker judgments before
the numbers can be trusted as a quality measure.
