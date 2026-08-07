#!/usr/bin/env python3
"""Measure tone entropy to locate the Fano floor (Design §0, §8).

    python scripts/measure_entropy.py corpus.txt [more.txt ...]

Prints H(t), H(t|s), H(t|s,w-1) (Miller–Madow corrected).  If the last drops to
0.05–0.15 bit, a bigram MAP decoder is near-optimal and no LLM is warranted —
run this BEFORE training to size the post-processor (bigram vs trigram).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.eval import measure_entropy


def _lines(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", nargs="+", help="plain-text corpus file(s)")
    args = ap.parse_args()

    report = measure_entropy(_lines(args.corpus))
    print(report.summary())
    if not report.bigram_suffices():
        print("\nNote: H(t|s,w-1) > 0.15 bit — extend the LM to a trigram (§8).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
