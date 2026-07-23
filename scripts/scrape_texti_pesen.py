#!/usr/bin/env python3
"""Scrape Kyrgyz song lyrics from texti-pesen.ucoz.ru into data/raw_src/modern/.

Same ucoz CMS as the classic (vostoka) site, so it reuses that extractor
(<td class="eText"> div-per-line, script removal, no-letter-line drop). Per-entry
it is one song. Folder = first two words of the artist (title minus the "Текст
песни" prefix); filename = the song title. Song rules: drop colon-metadata lines
and "Кайырма:" markers; keep a song even if it has no text.

Usage:
    python scripts/scrape_texti_pesen.py --max 100
"""

from __future__ import annotations

import argparse
import html as H
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scrape_vostoka import get, translit, slug, entry_text_lines  # reuse classic code

BASE = "https://texti-pesen.ucoz.ru"
LISTING = BASE + "/publ/pesni_na_kyrgyzskom/1-{page}-10"
OUT_ROOT = "data/raw_src/modern"


def listing_entries(page: int) -> list[str]:
    html = get(LISTING.format(page=page))
    hrefs = re.findall(r'href="([^"]+)"', html)
    entries = [h for h in hrefs
               if re.search(r"/publ/pesni_na_kyrgyzskom/.+/1-\d+-0-\d+$", h)]
    seen = dict.fromkeys(entries)
    return [(BASE + h if h.startswith("/") else h) for h in seen]


def author_song(raw: str) -> tuple[str, str]:
    """(artist, song) from the entry <title> ('Текст песни Artist "Song"')."""
    m = re.search(r"<title>(.*?)</title>", raw, re.S)
    title = re.sub(r"<[^>]+>", "", H.unescape(m.group(1))).strip() if m else ""
    title = re.sub(r"^\s*Текст\s+песни\s*", "", title, flags=re.I)
    words = title.split()
    artist = " ".join(words[:2])
    song = " ".join(words[2:]).strip(" «»\"'“”").strip()
    return artist, song or title


def song_lines(raw: str) -> list[str]:
    """Classic extraction + song rules (drop colon metadata and 'Кайырма')."""
    lines = []
    for ln in entry_text_lines(raw):
        if ":" in ln:
            continue
        if "".join(c for c in ln if c.isalpha()).lower() == "кайырма":
            continue
        lines.append(ln)
    return lines


def scrape(pages: int, max_entries: int, delay: float, start_page: int = 1) -> None:
    entries: list[str] = []
    for p in range(start_page, start_page + pages):
        try:
            page_entries = listing_entries(p)
        except Exception as exc:  # past the last listing page -> stop paginating
            print(f"listing page {p}: unavailable ({exc}); end of listing")
            break
        for url in page_entries:
            if url not in entries:
                entries.append(url)
        if len(entries) >= max_entries:
            break
        time.sleep(delay)
    entries = entries[:max_entries]
    print(f"scraping {len(entries)} songs -> {OUT_ROOT}/\n")

    written = empty = 0
    for i, url in enumerate(entries, 1):
        try:
            raw = get(url)
        except Exception as exc:
            print(f"  [{i}] FAILED {url}: {exc}")
            continue
        artist, song = author_song(raw)
        adir = os.path.join(OUT_ROOT, slug(artist, maxlen=40))
        os.makedirs(adir, exist_ok=True)
        text = "\n".join(song_lines(raw))
        name = slug(song)
        path = os.path.join(adir, name + ".txt")
        if os.path.exists(path):  # already have this song; don't duplicate
            print(f"  [{i}/{len(entries)}] skip (exists)   {artist} — {song[:30]}")
            continue
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + ("\n" if text else ""))
        written += 1
        if not text:
            empty += 1
        print(f"  [{i}/{len(entries)}] {artist:<22} {'(no text)' if not text else str(len(text.splitlines()))+' lines'}  {song[:30]}")
        time.sleep(delay)

    print(f"\nDone. {written} song files under {OUT_ROOT}/ ({empty} had no text)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape texti-pesen.ucoz.ru Kyrgyz songs")
    ap.add_argument("--pages", type=int, default=8, help="max listing pages to walk")
    ap.add_argument("--max", type=int, default=100)
    ap.add_argument("--start-page", type=int, default=1, help="first listing page to walk")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args(argv)
    scrape(args.pages, args.max, args.delay, args.start_page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
