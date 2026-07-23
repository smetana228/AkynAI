#!/usr/bin/env python3
"""Scrape song lyrics for one artist from kyrgyz-audio.com.

Each song is its own WordPress page (/<artist>-<song>-teksti/); lyrics live in
<div id="textpesni">. Files are written to data/raw_src/modern/<artist_folder>/,
one .txt per song, named from the URL slug (already Latin).

Rules (per request):
* Keep a song even if it has no text / no line breaks.
* Drop any line containing a colon (leading "Аткаруучу:"-style metadata) and any
  "Кайырма:" (chorus) marker.

Usage:
    python scripts/scrape_kyrgyz_audio.py \
        --artist-url https://kyrgyz-audio.com/muzyka/taalaj-bekturganov/ \
        --artist-folder taalai_bekturganov
"""

from __future__ import annotations

import argparse
import html as H
import os
import re
import ssl
import time
import urllib.request

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

OUT_ROOT = "data/raw_src/modern"


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Chrome/120"})
    return urllib.request.urlopen(url=req, timeout=30, context=_CTX).read().decode(
        "utf-8", "replace"
    )


def song_urls(artist_url: str) -> list[tuple[str, str]]:
    """(song_name, url) for each unique song on the artist page.

    Prefers the ``-teksti`` (lyrics) page and collapses duplicate ``-2`` posts.
    """
    raw = get(artist_url)
    m = re.match(r"(https?://[^/]+)", artist_url)
    host = m.group(1)
    slug_re = re.compile(re.escape(host) + r"/([a-z0-9-]+)/")
    prefix = re.search(r"/muzyka/([a-z0-9-]+)/", artist_url).group(1) + "-"

    found: dict[str, str] = {}  # base song name -> best url
    for url_slug in dict.fromkeys(slug_re.findall(raw)):
        if not url_slug.startswith(prefix) or "muzyka" in url_slug:
            continue
        song = url_slug[len(prefix):]
        base = re.sub(r"-teksti(-\d+)?$", "", song)
        base = re.sub(r"-\d+$", "", base)
        url = f"{host}/{url_slug}/"
        # prefer a "-teksti" page; otherwise keep the first seen
        if base not in found or "teksti" in url_slug:
            found[base] = url
    return [(base.replace("-", "_"), url) for base, url in found.items()]


def extract_lyrics(raw: str) -> str:
    """Text of <div id="textpesni">, minus share widgets and metadata lines."""
    m = re.search(r'id="textpesni"\s*>(.*?)(?:<div class="addtoany|</div>\s*</div>)',
                  raw, re.S)
    if not m:
        return ""
    block = m.group(1)
    block = re.sub(r"<br\s*/?>", "\n", block, flags=re.I)
    block = re.sub(r"</p\s*>", "\n", block, flags=re.I)
    block = re.sub(r"<[^>]+>", "", block)
    block = H.unescape(block).replace("\xa0", " ")

    kept = []
    for ln in block.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if ":" in ln:  # metadata (Аткаруучу:, Кайырма:, Автор:, …)
            continue
        if "".join(c for c in ln if c.isalpha()).lower() == "кайырма":
            continue
        kept.append(ln)
    return "\n".join(kept)


def scrape(artist_url: str, artist_folder: str, delay: float) -> None:
    out_dir = os.path.join(OUT_ROOT, artist_folder)
    os.makedirs(out_dir, exist_ok=True)

    songs = song_urls(artist_url)
    print(f"{len(songs)} songs -> {out_dir}/\n")

    empty = 0
    for i, (name, url) in enumerate(songs, 1):
        try:
            raw = get(url)
        except Exception as exc:
            print(f"  [{i}] FAILED {url}: {exc}")
            continue
        text = extract_lyrics(raw)
        path = os.path.join(out_dir, name + ".txt")
        n = 2
        while os.path.exists(path):
            path = os.path.join(out_dir, f"{name}_{n}.txt")
            n += 1
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + ("\n" if text else ""))
        if not text:
            empty += 1
        print(f"  [{i}/{len(songs)}] {name:<28} {'(no text)' if not text else str(len(text.splitlines()))+' lines'}")
        time.sleep(delay)

    print(f"\nDone. {len(songs)} files under {out_dir}/ ({empty} had no text)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape one artist's lyrics from kyrgyz-audio.com")
    ap.add_argument("--artist-url", required=True)
    ap.add_argument("--artist-folder", required=True, help="output subfolder name")
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args(argv)
    scrape(args.artist_url, args.artist_folder, args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
