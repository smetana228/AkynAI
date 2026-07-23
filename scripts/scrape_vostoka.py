#!/usr/bin/env python3
"""Scrape poems from vostoka.ucoz.com (category 1, sorted by views).

Traverses the views-sorted listing (/publ/1-<page>-10), visits each entry
subpage, and writes poems to data/raw_src/<author_slug>/<poem_slug>.txt.

* Author = the first two words of the entry title (transliterated to Latin for
  the folder name only; poem TEXT stays Cyrillic).
* A subpage is usually one poem, but sometimes a collection: it is split on
  ALL-UPPERCASE title lines, one .txt per poem.

Usage:
    python scripts/scrape_vostoka.py --pages 4 --max 100
"""

from __future__ import annotations

import argparse
import html as H
import os
import re
import ssl
import time
import urllib.request

BASE = "https://vostoka.ucoz.com"
LISTING = BASE + "/publ/1-{page}-10"          # page 1 = /publ/1-1-10
ENTRY_RE = re.compile(r"/publ/\d+-\d+-0-\d+")  # /publ/1-1-0-<id>
OUT_ROOT = "data/raw_src/classic"

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE  # a TLS-intercepting proxy sits in front

# Kyrgyz/Russian Cyrillic -> Latin, for folder/file names only.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "ң": "ng", "о": "o", "ө": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ү": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30, context=_CTX).read().decode(
        "utf-8", "replace"
    )


def translit(text: str) -> str:
    return "".join(_TRANSLIT.get(ch, _TRANSLIT.get(ch.lower(), ch)) for ch in text)


def slug(text: str, maxlen: int = 60) -> str:
    s = translit(text).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:maxlen] or "untitled"


def listing_entries(page: int) -> list[str]:
    """Entry URLs on one listing page, in display (views) order, de-duplicated."""
    html = get(LISTING.format(page=page))
    hrefs = ENTRY_RE.findall(html)
    seen = dict.fromkeys(hrefs)  # preserves order
    return [BASE + h for h in seen]


def entry_title(raw: str) -> str:
    for pat in (r'class="eTitle"[^>]*>(.*?)</', r"<h1[^>]*>(.*?)</h1>",
                r"<title>(.*?)</title>"):
        m = re.search(pat, raw, re.S)
        if m:
            t = re.sub(r"<[^>]+>", "", H.unescape(m.group(1))).strip()
            if t:
                return t
    return "unknown"


def entry_text_lines(raw: str) -> list[str]:
    """Cleaned, non-empty text lines from the eText block.

    Handles the site's three line-separator styles (<br>, <p>, and <div>-per-line),
    strips embedded <script>/<style> blocks, and drops decorative lines that carry
    no letters (e.g. ``***`` stanza separators).
    """
    m = re.search(r'<td class="eText"[^>]*>(.*?)</td>', raw, re.S)
    if not m:
        return []
    block = m.group(1)
    # Drop embedded script/style blocks entirely (tags + their inner code).
    block = re.sub(r"<script\b[^>]*>.*?</script>", "", block, flags=re.I | re.S)
    block = re.sub(r"<style\b[^>]*>.*?</style>", "", block, flags=re.I | re.S)
    # Every block-level boundary becomes a line break.
    block = re.sub(r"<br\s*/?>", "\n", block, flags=re.I)
    block = re.sub(r"</(p|div|h[1-6]|li|tr)\s*>", "\n", block, flags=re.I)
    block = re.sub(r"<[^>]+>", "", block)
    block = H.unescape(block).replace("\xa0", " ")
    # Keep only lines that contain at least one letter (drops ***, — , etc.).
    return [ln.strip() for ln in block.split("\n")
            if ln.strip() and any(c.isalpha() for c in ln)]


def is_title_line(line: str) -> bool:
    """A collection sub-title: short, all-uppercase, enough letters."""
    alpha = [c for c in line if c.isalpha()]
    return len(alpha) >= 3 and len(line) <= 60 and line.upper() == line and any(alpha)


def split_poems(lines: list[str], fallback_title: str):
    """Yield (title, [body lines]) splitting a collection on uppercase titles."""
    poems: list[tuple[str, list[str]]] = []
    cur_title, cur_body = None, []
    for ln in lines:
        if is_title_line(ln):
            if cur_body:
                poems.append((cur_title or fallback_title, cur_body))
            cur_title, cur_body = ln, []
        else:
            cur_body.append(ln)
    if cur_body:
        poems.append((cur_title or fallback_title, cur_body))
    return poems


def scrape(pages: int, max_entries: int, delay: float, start_page: int = 1) -> None:
    # Collect entry URLs in views order across the requested listing pages.
    entries: list[str] = []
    for p in range(start_page, start_page + pages):
        try:
            page_entries = listing_entries(p)
        except Exception as exc:  # past the last listing page -> stop paginating
            print(f"listing page {p}: unavailable ({exc}); end of listing")
            break
        print(f"listing page {p}: {len(page_entries)} entries")
        for url in page_entries:
            if url not in entries:
                entries.append(url)
        time.sleep(delay)
    entries = entries[:max_entries]
    print(f"\nscraping {len(entries)} entries -> {OUT_ROOT}/\n")

    written = 0
    for i, url in enumerate(entries, 1):
        try:
            raw = get(url)
        except Exception as exc:
            print(f"  [{i}] FAILED {url}: {exc}")
            continue
        title = entry_title(raw)
        author = " ".join(title.split()[:2])
        author_dir = os.path.join(OUT_ROOT, slug(author, maxlen=40))
        os.makedirs(author_dir, exist_ok=True)

        lines = entry_text_lines(raw)
        poems = split_poems(lines, fallback_title=title)
        for title_line, body in poems:
            if not body:
                continue
            name = slug(title_line)
            path = os.path.join(author_dir, name + ".txt")
            n = 2
            while os.path.exists(path):
                path = os.path.join(author_dir, f"{name}_{n}.txt")
                n += 1
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(body) + "\n")
            written += 1
        print(f"  [{i}/{len(entries)}] {author:<24} {len(poems)} poem(s)  <- {url.split('/')[-1]}")
        time.sleep(delay)

    print(f"\nDone. {written} poem files written under {OUT_ROOT}/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape vostoka.ucoz.com poems")
    ap.add_argument("--pages", type=int, default=4, help="listing pages (25 entries each)")
    ap.add_argument("--max", type=int, default=100, help="max entries to scrape")
    ap.add_argument("--start-page", type=int, default=1, help="first listing page to walk")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = ap.parse_args(argv)
    scrape(args.pages, args.max, args.delay, args.start_page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
