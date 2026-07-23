#!/usr/bin/env python3
"""List song files that contain no Kyrgyz-specific letters (ө, ү, ң).

A Kyrgyz song with zero ө/ү/ң is suspicious — usually a contributor who typed
о/у/н (or romanized) instead. This flags them so you can manually check which
are genuinely faulty vs. just short songs that happen to use none.

Usage:
    python scripts/check_kyrgyz_letters.py                 # scans data/raw_src/modern
    python scripts/check_kyrgyz_letters.py --root data/raw_src --min 1
"""

from __future__ import annotations

import argparse
import glob
import os

KY_LETTERS = set("өүңӨҮҢ")
CYR = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяөүң")


def scan(root: str, minimum: int):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)):
        text = open(path, encoding="utf-8").read()
        ky = sum(1 for ch in text if ch in KY_LETTERS)
        if ky >= minimum:
            continue
        cyr = sum(1 for ch in text.lower() if ch in CYR)
        lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        lines = sum(1 for ln in text.splitlines() if ln.strip())
        rows.append((path, ky, cyr, lat, lines))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flag songs lacking Kyrgyz letters ө/ү/ң")
    ap.add_argument("--root", default="data/raw_src/modern")
    ap.add_argument("--min", type=int, default=1,
                    help="report files with fewer than this many ө/ү/ң (default 1)")
    args = ap.parse_args(argv)

    total = len(glob.glob(os.path.join(args.root, "**", "*.txt"), recursive=True))
    rows = scan(args.root, args.min)

    print(f"{'ө/ү/ң':>6} {'cyr':>5} {'lat':>5} {'lines':>6}  file")
    print("-" * 72)
    for path, ky, cyr, lat, lines in rows:
        tag = " EMPTY" if lines == 0 else (" LATIN" if lat > cyr else "")
        rel = os.path.relpath(path, args.root)
        print(f"{ky:>6} {cyr:>5} {lat:>5} {lines:>6}  {rel}{tag}")
    print("-" * 72)
    print(f"{len(rows)} of {total} files have < {args.min} Kyrgyz-specific letter(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
