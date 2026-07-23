"""Phase 3 back-generation (§6.2): CleanPoem -> synthetic instruction.

The poem stays authentic human text; only the *instruction* is synthetic. For
each poem we emit two SFTExamples — a plain-topic instruction and a
form-explicit one — so the model learns both styles.

The prompt construction, form derivation, and guardrails are pure and testable;
the actual instruction text comes from an ``LLMBackend`` (Anthropic by default).
"""

from __future__ import annotations

import argparse
import os

from ..jsonl import read_jsonl, write_jsonl
from ..llm.backend import AnthropicBackend, LLMBackend, OllamaBackend

# The task is described in English (weaker/local models follow it far more
# reliably than a Kyrgyz-only prompt — a 7B model tends to just echo the poem
# otherwise), but the generated instruction itself must be in Kyrgyz.
_SYSTEM = (
    "You write a SHORT instruction (a prompt) that the given Kyrgyz poem is the "
    "answer to.\n"
    "Rules:\n"
    "- Output ONLY the instruction itself — no preamble, no quotes, no explanation.\n"
    "- Write the instruction in KYRGYZ.\n"
    "- Do NOT repeat, quote, or paraphrase any line of the poem.\n"
    "- Keep it to one or two short sentences.\n"
    "Example of a good instruction: Мекен жөнүндө ыр жаз."
)


def form_spec(cp: dict) -> dict:
    """Form spec drawn from the poem's already-computed prosody tags."""
    spl = cp.get("syllables_per_line") or []
    syl = f"{min(spl)}-{max(spl)}" if spl else "7-8"
    return {
        "n_lines": cp.get("n_lines", len(spl)),
        "syllables": syl,
        "rhyme_scheme": cp.get("detected_rhyme_scheme", ""),
    }


def build_prompt(cp: dict, form_explicit: bool) -> tuple[str, str]:
    """Return (system, user) messages for the backend.

    The user turn is English guidance; the ``form_explicit`` variant tells the
    model to state the form constraints *inside* the Kyrgyz instruction using the
    Kyrgyz words сап (lines), муун (syllables), and уйкаштык (rhyme).
    """
    fs = form_spec(cp)
    if form_explicit:
        instruction_hint = (
            "Write an instruction that ALSO states the form, in Kyrgyz, using the "
            f"words сап and муун: {fs['n_lines']} сап (lines), {fs['syllables']} "
            f"муун per line (syllables), уйкаштык (rhyme) '{fs['rhyme_scheme']}'."
        )
    else:
        instruction_hint = "Give a topic-only instruction. Do not mention the form."
    user = f"{instruction_hint}\n\nPoem:\n{cp['text']}"
    return _SYSTEM, user


def _looks_kyrgyz(text: str) -> bool:
    # Any Latin letter is an English leak (common on local models) — reject it;
    # a Kyrgyz instruction has no reason to contain a/b/c…
    if any(ch.isascii() and ch.isalpha() for ch in text):
        return False
    cyr = sum(1 for ch in text if ch.isalpha() and ch.lower() in
              "абвгдеёжзийклмнопрстуфхцчшщъыьэюяөүң")
    letters = sum(1 for ch in text if ch.isalpha())
    return letters > 0 and cyr / letters >= 0.6


def _echoes_poem(instruction: str, poem: str) -> bool:
    """True if the instruction leaked a poem line (guardrail)."""
    poem_lines = [ln.strip() for ln in poem.splitlines() if ln.strip()]
    return any(ln and ln in instruction for ln in poem_lines)


def _assemble(cp: dict, instruction: str) -> dict | None:
    """Apply guardrails and build one SFTExample, or None if rejected.

    Rejects an instruction that isn't Kyrgyz or that echoes the poem. ``output``
    is kept byte-identical to the cleaned poem.
    """
    instruction = (instruction or "").strip()
    if not _looks_kyrgyz(instruction) or _echoes_poem(instruction, cp["text"]):
        return None
    return {
        "instruction": instruction,
        "output": cp["text"],  # byte-identical, unchanged
        "form": form_spec(cp),
        "source_poem_id": cp.get("id", ""),
    }


def covered_poem_ids(examples_path: str) -> set[str]:
    """source_poem_ids already present in an existing SFTExamples file."""
    if not os.path.exists(examples_path):
        return set()
    return {e.get("source_poem_id") for e in read_jsonl(examples_path)}


def filter_covered(poems: list[dict], covered: set[str]) -> list[dict]:
    """Drop poems that already have back-generated instructions."""
    if not covered:
        return list(poems)
    return [p for p in poems if p.get("id") not in covered]


def filter_buckets(poems: list[dict], exclude: set[str]) -> list[dict]:
    """Drop poems whose ``meta.bucket`` is in ``exclude`` (e.g. the Manas epic,
    which belongs in continued-pretraining, not the SFT instruction set)."""
    if not exclude:
        return list(poems)
    return [p for p in poems if p.get("meta", {}).get("bucket") not in exclude]


def make_examples(backend: LLMBackend, cp: dict) -> list[dict]:
    """Two SFTExamples (plain + form-explicit) for one CleanPoem (one call each)."""
    examples = []
    for form_explicit in (False, True):
        system, user = build_prompt(cp, form_explicit)
        ex = _assemble(cp, backend.complete(system, user))
        if ex:
            examples.append(ex)
    return examples


def _assemble_from_texts(poems: list[dict], texts: dict[str, str]) -> list[dict]:
    """Map batch results (keyed by ``"<index>-<style>"``) back to SFTExamples.

    ``poems`` must be the SAME list, in the same order, that produced the batch —
    the custom_id encodes the poem's position. Pure; no API.
    """
    out = []
    for i, cp in enumerate(poems):
        for style in (0, 1):
            text = texts.get(f"{i}-{style}")
            if text is None:
                continue
            ex = _assemble(cp, text)
            if ex:
                out.append(ex)
    return out


def _batch_requests(poems: list[dict], model: str):  # pragma: no cover - API types
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = []
    for i, cp in enumerate(poems):
        for style in (0, 1):  # 0 = plain, 1 = form-explicit
            system, user = build_prompt(cp, bool(style))
            requests.append(Request(
                custom_id=f"{i}-{style}",
                params=MessageCreateParamsNonStreaming(
                    model=model, max_tokens=256, system=system,
                    messages=[{"role": "user", "content": user}],
                ),
            ))
    return requests


def _poll_and_collect(client, batch_id: str, poll_seconds: int) -> dict[str, str]:  # pragma: no cover - API
    import time

    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        c = b.request_counts
        done = c.succeeded + c.errored + c.canceled + c.expired
        print(f"  status={b.processing_status} done={done}/{done + c.processing}")
        time.sleep(poll_seconds)

    texts: dict[str, str] = {}
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            texts[r.custom_id] = "".join(
                blk.text for blk in r.result.message.content if blk.type == "text"
            )
    return texts


def make_examples_batched(poems: list[dict], model: str,
                          poll_seconds: int = 30) -> list[dict]:  # pragma: no cover - API
    """Back-generate via the 50%-cheaper Batch API: submit, poll, assemble."""
    client = AnthropicBackend(model=model)._get_client()
    batch = client.messages.batches.create(requests=_batch_requests(poems, model))
    print(f"submitted batch {batch.id} ({2 * len(poems)} requests); polling...")
    print(f"  (if this poll dies, reconnect with: --resume-batch {batch.id})")
    texts = _poll_and_collect(client, batch.id, poll_seconds)
    return _assemble_from_texts(poems, texts)


def resume_batch(poems: list[dict], batch_id: str, model: str,
                 poll_seconds: int = 30) -> list[dict]:  # pragma: no cover - API
    """Reconnect to an already-submitted batch and assemble its results.

    ``poems`` (i.e. --in + --exclude-bucket) must match the original submission so
    the custom_id indices line up.
    """
    client = AnthropicBackend(model=model)._get_client()
    print(f"resuming batch {batch_id}; polling...")
    texts = _poll_and_collect(client, batch_id, poll_seconds)
    return _assemble_from_texts(poems, texts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Back-generate instructions -> SFTExamples")
    ap.add_argument("--in", dest="inp", default="data/clean/poems.jsonl")
    ap.add_argument("--out", default="data/sft/examples.jsonl")
    ap.add_argument("--backend", choices=("anthropic", "ollama"), default="anthropic")
    ap.add_argument("--model", default=None,
                    help="model id (default: claude-haiku-4-5 / aya-expanse per backend)")
    ap.add_argument("--host", default="http://localhost:11434", help="ollama host")
    ap.add_argument("--batch", action="store_true",
                    help="Anthropic only: use the Batch API (50%% cheaper)")
    ap.add_argument("--resume-batch", default=None, metavar="BATCH_ID",
                    help="reconnect to an already-submitted batch and assemble its "
                         "results (use the SAME --in / --exclude-bucket)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="only back-generate poems missing from --out, and keep "
                         "the existing examples (saves re-paying for covered poems)")
    ap.add_argument("--exclude-bucket", nargs="*", default=["manas"],
                    help="skip poems in these meta.buckets (default: manas — it is "
                         "for CPT, not SFT). Pass with no values to include all.")
    args = ap.parse_args(argv)

    poems = list(read_jsonl(args.inp))
    excl = set(args.exclude_bucket)
    kept = filter_buckets(poems, excl)
    if len(kept) != len(poems):
        print(f"excluded {len(poems) - len(kept)} poems from buckets {sorted(excl)}")
    poems = kept

    existing: list[dict] = []
    if args.skip_existing:
        existing = list(read_jsonl(args.out)) if os.path.exists(args.out) else []
        covered = covered_poem_ids(args.out)
        before = len(poems)
        poems = filter_covered(poems, covered)
        print(f"skip-existing: {before - len(poems)} poems already covered "
              f"({len(existing)} examples kept); {len(poems)} left to generate")
        if not poems:
            print("nothing new to generate")
            write_jsonl(args.out, existing)
            return 0

    if args.resume_batch:
        out = resume_batch(poems, args.resume_batch, model=args.model or "claude-haiku-4-5")
    elif args.backend == "ollama":
        backend = OllamaBackend(model=args.model or "aya-expanse", host=args.host)
        out = []
        for cp in poems:
            out.extend(make_examples(backend, cp))
    elif args.batch:
        out = make_examples_batched(poems, model=args.model or "claude-haiku-4-5")
    else:
        backend = AnthropicBackend(model=args.model or "claude-haiku-4-5")
        out = []
        for cp in poems:
            out.extend(make_examples(backend, cp))
    n = write_jsonl(args.out, existing + out)
    print(f"Wrote {n} SFTExamples -> {args.out}"
          + (f" ({len(existing)} existing + {len(out)} new)" if existing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
