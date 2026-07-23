# AkynAI — a Kyrgyz poetry LLM

An open-source LLM fine-tuned to write original Kyrgyz poems on a given topic,
with controllable form (line count, syllables per line, rhyme scheme) and
respect for traditional Kyrgyz prosody.

The strategic idea: Kyrgyz orthography is near-phonemic, so **meter and rhyme
can be checked deterministically in code**. That verifier then sits in the loop
at every stage — data cleaning, evaluation, inference-time rejection sampling,
and preference tuning — which is what makes prosody good rather than accidental.

**Status:** pipeline validated end to end. An SFT-only baseline is trained and
measured; CPT and DPO are built but not yet run. See [RESULTS.md](RESULTS.md).

## Where it stands

| Stage | State |
|---|---|
| Prosody verifier (§4) | done, fully unit-tested — **99 tests** |
| Corpus ingest + clean (§5) | done — 1,625 poems, 82K lines |
| Instruction back-generation (§6) | done — 1,825 SFT pairs |
| Continued pretraining (§7.1) | built, **not yet run** |
| Instruction tuning (§7.2) | **run** — baseline measured |
| Preference tuning / DPO (§7.3) | built, **not yet run** |
| Generation + eval (§8) | done, verified on a real checkpoint |

**Corpus** (gitignored — scraped text is not redistributed):

| Bucket | Poems | Lines | Share |
|---|---:|---:|---:|
| Classic poetry | 1,002 | 55,798 | 67.7% |
| Manas (epic) | 1 | 12,237 | 14.8% |
| Modern songs | 622 | 14,419 | 17.5% |

Manas is deliberately kept a minority so the model does not collapse into epic
register. General Kyrgyz prose (Wikipedia, 83M chars) backs the CPT stage.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # verifier + data layer + pytest
pytest -q                      # 99 tests, no GPU or API key needed
```

The prosody verifier has **zero runtime dependencies**; ML and LLM deps live in
optional extras (`dev`, `llm`, `probe`, `train`).

```python
from kyrpoet.prosody import score_poem, PoemForm
score = score_poem(poem_text, PoemForm(n_lines=4, syllables=(7, 8), rhyme_scheme="aabb"))
print(score.syllables_per_line, score.detected_rhyme_scheme, score.overall)
```

## Pipeline

```bash
# 1. corpus -> prosody-tagged JSONL
python scripts/build_corpus.py
python -m kyrpoet.data.clean

# 2. instruction pairs (needs ANTHROPIC_API_KEY, or --backend ollama for a local model)
pip install -e ".[llm]"
python -m kyrpoet.data.backgen --backend anthropic --batch --skip-existing
python -m kyrpoet.data.build_datasets sft --max-lines 48

# 3. CPT corpus (poetry upsampled so it isn't swamped by general prose)
python scripts/fetch_general_ky.py
python -m kyrpoet.data.build_datasets cpt --extra data/cpt/general_ky.jsonl --poetry-frac 0.10

# 4. train on a 24GB GPU
pip install -e ".[train]"
./scripts/train.sh                                    # CPT -> SFT -> DPO
python -m kyrpoet.train.sft --config configs/sft.yaml # or SFT-only from base

# 5. generate + evaluate
python -m kyrpoet.generate.generate --checkpoint checkpoints/sft \
    --topic "көктөм" --n-lines 4 --syllables 7-8 --best-of 8
python -m kyrpoet.eval.evaluate --checkpoint checkpoints/sft
```

Base model is **Qwen2.5-7B** (all training is 4-bit QLoRA); `kyrpoet.tokenizer_probe`
ranks alternatives by tokenizer fertility on Kyrgyz, which matters because rhyme
lives in the word-final characters subword merges tend to bury.

---

The original build spec follows, unchanged.

<!-- ─────────────────────────────────────────────────────────────────────── -->

# Project Spec: Kyrgyz Poetry LLM (`kyr-poet`)

A fine-tuned open-source LLM that writes original Kyrgyz poems on arbitrary topics, with controllable form (line count, syllables per line, rhyme scheme) and respect for traditional Kyrgyz prosody (7–8 syllable syllabic meter, end rhyme, line-initial alliteration).

This document is the build spec. It is written to be executed by Claude Code incrementally, phase by phase, with each phase producing testable artifacts. **Build and test each module before moving on. Do not skip the verifier (Phase 1) — it is the backbone of the whole pipeline.**

---

## 0. Background the implementer must know

Kyrgyz poetry is **syllabic**, not stress-based. Traditional epic/folk verse (including Manas) uses a 7–8 syllable line, with lines linked by **end rhyme** and **line-initial (vertical) alliteration**; both are used loosely and irregularly. Our target is lyric/topical poetry in this tradition, not archaic epic pastiche.

Two properties of Kyrgyz make automatic prosody checking feasible, and this is the strategic core of the project:

1. **Near-phonemic orthography.** Syllable count ≈ vowel-letter count. We can count meter and check rhyme from spelling alone, without a pronunciation dictionary.
2. **Agglutinative + vowel harmony → grammatical rhyme.** Words sharing a grammatical suffix naturally rhyme. This is common and easy for a model to learn.

Because prosody is checkable in code, we put a deterministic verifier in the loop at every stage: data cleaning, evaluation, inference-time rejection sampling, and preference-tuning (DPO). This is what makes rhyme quality good rather than accidental.

---

## 1. Goals, non-goals, assumptions

### Goals
- Given a topic (and optional form constraints), generate an original Kyrgyz poem that reads as fluent, natural, and aesthetically pleasing to native speakers.
- Support explicit form control: number of lines, syllables per line, rhyme scheme.
- Provide automatic prosody metrics (syllable conformity, rhyme rate, alliteration rate).

### Non-goals
- Not a translation system, not a chat assistant, not ASR.
- Not aiming to reproduce copyrighted published editions of Manas verbatim.
- Not (initially) a character-level/byte-level model. Keep that as a documented fallback (§10).

### Assumptions / environment
- Python 3.10+.
- Single GPU with ~24 GB VRAM (RTX 3090/4090) as the baseline target; all training uses **QLoRA (4-bit)**. Note in README how to scale to an A100 rental for faster continued pretraining.
- Anthropic API is available (via the standard `ANTHROPIC_API_KEY` env var) and is the default backend for two LLM-assisted steps: instruction back-generation (§6) and LLM-as-judge eval (§8). Make this backend pluggable behind an interface so a local model can be swapped in.
- Base model: **Qwen2.5-7B** as the default, pending the tokenizer-fertility probe in Phase 0, which may recommend Gemma or Aya instead.

---

## 2. Repository layout

```
kyr-poet/
├── README.md
├── pyproject.toml                # deps + pinned versions
├── configs/
│   ├── cpt.yaml                  # continued-pretraining run config
│   ├── sft.yaml                  # supervised fine-tune config
│   └── dpo.yaml                  # preference-tuning config
├── src/kyrpoet/
│   ├── prosody/
│   │   ├── vowels.py             # vowel sets + classification
│   │   ├── syllables.py          # syllable counting
│   │   ├── rhyme.py              # rhyme keys + scheme detection
│   │   ├── alliteration.py       # initial-sound alliteration
│   │   └── scorer.py             # score_poem(): the top-level verifier
│   ├── data/
│   │   ├── ingest.py             # parse raw sources -> RawPoem JSONL
│   │   ├── clean.py              # normalize + tag + filter -> CleanPoem JSONL
│   │   ├── backgen.py            # poem -> synthetic instruction (LLM-assisted)
│   │   └── build_datasets.py     # assemble CPT / SFT / DPO splits
│   ├── llm/
│   │   └── backend.py            # LLMBackend interface + AnthropicBackend
│   ├── train/
│   │   ├── cpt.py                # continued pretraining (QLoRA)
│   │   ├── sft.py                # instruction tuning (QLoRA)
│   │   └── dpo.py                # preference tuning (QLoRA)
│   ├── generate/
│   │   ├── generate.py           # single-shot generation
│   │   └── rejection_sample.py   # generate N, score, keep best
│   ├── eval/
│   │   └── evaluate.py           # prosody metrics + LLM-judge harness
│   └── tokenizer_probe.py        # Phase 0 fertility probe
├── tests/
│   ├── test_syllables.py
│   ├── test_rhyme.py
│   ├── test_alliteration.py
│   └── test_scorer.py
├── data/                         # gitignored; raw/, clean/, sft/, dpo/
└── scripts/                      # thin CLI wrappers per phase
```

Use JSONL for all datasets. One record per line. Keep every intermediate artifact on disk so phases are independently re-runnable.

---

## 3. Phase 0 — Scaffold + tokenizer fertility probe

**Build:** `src/kyrpoet/tokenizer_probe.py`.

Take a sample Kyrgyz text file (a few thousand words; Wikipedia dump text is fine). For each candidate tokenizer — Qwen2.5-7B, Gemma-2-9B, Aya-23-8B — compute:
- total tokens / total whitespace-words (**fertility**; lower is better),
- % of words that survive as a single token,
- a few example word→token breakdowns for eyeballing.

Print a ranked table and a recommendation. **Default to Qwen2.5-7B unless its fertility is clearly worse than an alternative** (a common failure mode for Cyrillic). Poor tokenization disproportionately hurts rhyme, because rhyme lives in the final characters that subword merges tend to bury.

**Acceptance:** running `python -m kyrpoet.tokenizer_probe --corpus data/sample_ky.txt` prints the table and a chosen base model. Record the choice in README.

---

## 4. Phase 1 — The prosody verifier (build this first, test it hard)

This is the most important module. Everything downstream depends on it. It must be pure-Python, deterministic, dependency-light, and unit-tested.

### 4.1 Vowels — `prosody/vowels.py`
Kyrgyz Cyrillic vowel letters, each counting as exactly one syllable nucleus:

```
а, е, ё, и, о, ө, у, ү, ы, э, ю, я
```

Notes to encode:
- `й` is a **consonant** (glide) — never counts as a vowel.
- `ъ`, `ь` (appear only in Russian loanwords) are not vowels.
- `я`, `ю`, `ё`, `е` are iotated but still exactly **one** nucleus each for counting.
- Provide harmony classification for later use: back vowels {а, о, у, ы}, front vowels {е, э, ө, ү, и}; rounded {о, ө, у, ү}, unrounded {а, е, э, ы, и}. (Used by optional harmony metric and by rhyme calibration.)

### 4.2 Syllables — `prosody/syllables.py`
```
count_syllables(word: str) -> int          # = number of vowel letters in the word
count_line_syllables(line: str) -> int      # sum over words; strip punctuation first
```
Rule: syllable count equals the number of vowel letters. This is a deliberate, documented heuristic. Handle: mixed-case, punctuation/dashes, digits (skip or flag), and empty lines.

**Unit-test fixtures (must pass):**
| word | syllables |
|------|-----------|
| ата | 2 |
| алма | 2 |
| мектеп | 2 |
| Кыргызстан | 3 |
| аю | 2 |

### 4.3 Rhyme — `prosody/rhyme.py`
```
rhyme_key(word: str, mode="phonetic") -> str
lines_rhyme(a: str, b: str, mode) -> bool
detect_rhyme_scheme(lines: list[str], mode) -> str   # e.g. "aabb", "abab"
```
- Extract the **last word** of a line (strip punctuation).
- `phonetic` mode: rhyme key = substring from the last vowel to end of word (final rime: last vowel + trailing consonants). Two words rhyme if keys match.
- `suffix`/`grammatical` mode: also match on shared trailing morpheme-like sequences (e.g. last 2–3 characters), to capture grammatical rhyme. Make the tail length configurable.
- `detect_rhyme_scheme` assigns letters a, b, c… to distinct rhyme classes over the lines.

**Important:** Kyrgyz vowel harmony means the "same" suffix can surface with different vowels (e.g. front vs back variants). Do **not** hard-code a single correct answer for whether harmonic variants rhyme. Make it configurable and flag in README that rhyme thresholds must be **calibrated against native-speaker judgments** on a small labeled set before trusting the metric.

### 4.4 Alliteration — `prosody/alliteration.py`
```
initial_sound(word: str) -> str
alliteration_rate(lines: list[str]) -> float   # fraction of adjacent line pairs sharing line-initial sound
```
Compare the first letter of the first word of each line. Convention: **all vowels alliterate with each other** (standard for this tradition); consonants must match exactly.

### 4.5 Top-level scorer — `prosody/scorer.py`
```
score_poem(text: str, target: PoemForm | None = None) -> PoemScore
```
`PoemForm` (optional target): `{n_lines, syllables: int | (min,max), rhyme_scheme}`.
`PoemScore` returns at least:
- `n_lines`
- `syllables_per_line: list[int]`
- `syllable_conformity: float` — fraction of lines within target range (default target 7–8 if none given)
- `detected_rhyme_scheme: str`
- `rhyme_rate: float` — fraction of intended rhyme pairs that actually rhyme (vs `target.rhyme_scheme` if given, else vs detected)
- `alliteration_rate: float`
- `overall: float` — weighted combination (default weights: syllable 0.5, rhyme 0.4, alliteration 0.1; make configurable)

**Acceptance for Phase 1:** `pytest tests/` passes, including the fixtures above and at least 10 hand-checked real poem lines. `score_poem` runs on a full poem and returns a populated `PoemScore`.

---

## 5. Phase 2 — Data ingestion, cleaning, tagging

### 5.1 Sources
- General Kyrgyz text for fluency (Wikipedia `ky`, web crawl / OSCAR `ky`) — for continued pretraining.
- Poetry corpus: lyric/akyn verse and modern Kyrgyz poets (bulk of the style signal), plus **selective** Manas excerpts for meter/alliteration/parallelism only. Keep Manas a minority of the poetry data so the model doesn't collapse into epic register. Start from the community index `github.com/alexeyev/awesome-kyrgyz-nlp` to locate corpora.
- **Licensing:** the epic is ancient/public-domain, but specific published transcriptions (Sagymbay, Sayakbay Karalaev editions) may carry editorial copyright. Track a `license` field per source; keep anything non-redistributable out of any shareable release.

### 5.2 `data/ingest.py` → `RawPoem` JSONL
```json
{"id":"str","source":"str","title":"str|null","author":"str|null",
 "text":"str","license":"str","meta":{}}
```
One record per poem. Preserve line breaks in `text`.

### 5.3 `data/clean.py` → `CleanPoem` JSONL
- Normalize: unicode NFC, fix common OCR confusions (Latin/Cyrillic homoglyphs a/а, o/о, e/е, c/с, p/р, x/х), collapse whitespace, standardize line breaks.
- Run the Phase 1 scorer to attach form tags and quality flags:
```json
{"...RawPoem fields...",
 "n_lines":int,"syllables_per_line":[int],"detected_rhyme_scheme":"str",
 "prosody":{"syllable_conformity":float,"rhyme_rate":float,"alliteration_rate":float},
 "quality_flags":["str"]}
```
- Filter out: garbled/OCR-broken poems, wildly irregular syllable counts, too-short fragments, and duplicates (near-dup detection via normalized text hash + optional MinHash).

**Acceptance:** `clean/poems.jsonl` exists, every record has prosody tags, and a printed summary shows counts kept/dropped and distributions of lines and syllables-per-line.

---

## 6. Phase 3 — Instruction dataset via back-generation

Raw poems teach continuation, not on-demand topical writing. Build `(instruction → poem)` pairs where **the poem is authentic human text and only the instruction is synthetic**.

### 6.1 `llm/backend.py`
```
class LLMBackend(Protocol):
    def complete(self, system: str, user: str, **kw) -> str: ...
class AnthropicBackend(LLMBackend): ...   # uses ANTHROPIC_API_KEY
```
Keep it pluggable so a local model can replace the API later.

### 6.2 `data/backgen.py`
For each `CleanPoem`, prompt the backend to produce a natural Kyrgyz instruction that the poem answers — a topic/title plus, drawn from the poem's already-computed tags, the form spec (n_lines, syllables/line, rhyme scheme). Produce both a plain-topic instruction and a form-explicit instruction so the model learns both styles.

Output `SFTExample` JSONL:
```json
{"instruction":"str","output":"<the real poem>","form":{"n_lines":int,"syllables":"7-8","rhyme_scheme":"str"},"source_poem_id":"str"}
```
Guardrails: instruction must be in Kyrgyz; do not let the backend alter/echo the poem into the instruction; keep `output` byte-identical to the cleaned poem.

**Acceptance:** `sft/train.jsonl` + `sft/val.jsonl`, with a spot-check script printing 10 random pairs for manual review.

---

## 7. Phase 4 — Training

All runs are QLoRA (4-bit base, LoRA adapters). Prefer **Unsloth** for speed/memory on a single 24 GB GPU; fall back to TRL + PEFT + bitsandbytes if Unsloth doesn't support the chosen base. Configs live in `configs/*.yaml` (base model id, LoRA r/alpha/dropout, target modules, lr, scheduler, epochs, seq len, batch size, grad-accum, eval steps, save steps).

### 7.1 `train/cpt.py` — continued pretraining
Objective: raise base-language fluency and absorb poetic register **before** instruction tuning, since fluency is the quality ceiling. Train on general Kyrgyz + poetry corpus (plain LM objective). Save adapter/merged weights as `checkpoints/cpt`.

### 7.2 `train/sft.py` — instruction tuning
Start from the CPT checkpoint. Train on `sft/train.jsonl` using the base model's chat template. Save `checkpoints/sft`.

### 7.3 `train/dpo.py` — prosody-aware preference tuning
This is where rhyme/meter quality is won. Pipeline:
1. Sample a set of prompts (topic + form).
2. With the SFT model, generate several candidates per prompt.
3. Score every candidate with `score_poem`.
4. Build preference pairs: higher `overall` = chosen, lower = rejected (with a minimum score gap to avoid noisy pairs; optionally require chosen to also pass a fluency check via the LLM judge so we don't reward metrically-perfect nonsense).
5. Run DPO from the SFT checkpoint → `checkpoints/dpo`.

**Acceptance per sub-phase:** training completes, loss curves logged, and a fixed eval prompt set (see §8) shows non-regression vs the previous checkpoint on prosody metrics.

---

## 8. Phase 5 — Generation + evaluation

### 8.1 `generate/generate.py`
CLI: topic + optional form → poem. Uses the latest checkpoint and the chat template.

### 8.2 `generate/rejection_sample.py`
Generate N candidates, score with `score_poem`, return the best. This is the recommended default inference path — it noticeably lifts prosody conformity at the cost of extra sampling.

### 8.3 `eval/evaluate.py`
Maintain a **fixed** eval prompt set (varied topics × form specs) committed to the repo so every checkpoint is comparable. Report two tracks:
- **Automatic (from the verifier):** mean syllable_conformity, rhyme_rate, alliteration_rate, and a distribution of `overall`.
- **LLM-as-judge (fluency/coherence/imagery):** the `AnthropicBackend` rates each poem on fluency, topical relevance, and aesthetic quality with a fixed rubric. This is a proxy, **not** a substitute for native-speaker review.
- Emit a small HTML/markdown report per run.

**Native-speaker eval is required before trusting results.** Provide a tiny script that samples poems into a rating sheet (CSV) for a human to fill in, and calibrate the verifier's rhyme thresholds against those human labels (see §4.3).

**Acceptance:** `python -m kyrpoet.eval.evaluate --checkpoint checkpoints/dpo` produces a report; DPO beats SFT beats CPT-only on automatic prosody metrics.

---

## 9. Milestones (suggested execution order for Claude Code)

1. **M0** Scaffold repo, `pyproject.toml`, tokenizer probe → base-model decision.
2. **M1** Prosody verifier + full unit tests (the crux). Ship before anything else.
3. **M2** Ingest + clean + tag corpus; print data stats.
4. **M3** Back-generation → SFT dataset; manual spot-check.
5. **M4** CPT run; sanity-check Kyrgyz fluency of the CPT model.
6. **M5** SFT run; first end-to-end poems.
7. **M6** DPO loop + rejection sampling; eval report showing gains.
8. **M7** Native-speaker rating pass + threshold calibration; iterate.

Each milestone is independently runnable and leaves artifacts on disk. Commit after each.

---

## 10. Risks and fallbacks

- **Rhyme stays inconsistent after DPO.** Expected ceiling for subword models. Mitigations already built in: verifier-in-the-loop (data filtering, DPO, rejection sampling). Documented harder fallback: a byte/character-level base model (better at rhyme, bigger build) — do not start here.
- **Manas register bleed** (model writes archaic epic pastiche). Mitigation: keep Manas a minority of poetry data; monitor via the LLM judge on register.
- **Tokenizer fertility poor on Qwen.** Mitigation: Phase 0 probe can switch base to Gemma/Aya.
- **Grammatical-rhyme false positives/negatives** from vowel harmony. Mitigation: configurable rhyme modes + native-speaker calibration set.
- **MT'd/scraped junk in training data.** Mitigation: aggressive cleaning + prosody-based filtering; keep only well-formed verse.

---

## 11. Deliverables checklist

- [ ] Tokenizer fertility report + recorded base-model choice
- [ ] Prosody verifier package with passing test suite
- [ ] Cleaned, prosody-tagged poem corpus (JSONL)
- [ ] SFT dataset (topic/form → real poem) + DPO preference set
- [ ] CPT / SFT / DPO checkpoints
- [ ] Generation CLI + rejection-sampling CLI
- [ ] Eval harness (automatic + LLM-judge) with a committed fixed prompt set
- [ ] Native-speaker rating script + calibration notes
- [ ] README documenting decisions, how to run each phase, and hardware notes
