#!/usr/bin/env python3
"""Evaluate OCR output with the p_B/p_T split (Design §8).

    python scripts/evaluate.py --ref gt.txt --hyp pred.txt

``gt.txt`` and ``pred.txt`` are line-aligned (one sample per line).  Reports
CER/WER and — the point of §8 — separate base (p_B) and tone (p_T) accuracy, so
you can confirm errors are tone-dominated and see where to spend effort.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.eval import evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="ground-truth file")
    ap.add_argument("--hyp", required=True, help="prediction file")
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip NFC + tone-placement normalisation")
    args = ap.parse_args()

    refs = Path(args.ref).read_text(encoding="utf-8").splitlines()
    hyps = Path(args.hyp).read_text(encoding="utf-8").splitlines()
    if len(refs) != len(hyps):
        print(f"error: {len(refs)} refs vs {len(hyps)} hyps", file=sys.stderr)
        return 2

    m = evaluate(refs, hyps, normalize=not args.no_normalize)
    print(m.summary())
    print()
    if m.tone_error_rate > m.base_error_rate:
        print("→ errors are tone-dominated (as §0 predicts): invest in the "
              "n-gram tone recovery and the 48px/asymmetric-stride recogniser.")
    else:
        print("→ base errors dominate here: check preprocessing, detection, and "
              "the base head before tuning tone recovery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
