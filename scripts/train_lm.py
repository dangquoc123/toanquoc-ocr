#!/usr/bin/env python3
"""Train the tone-recovery language model (Design §5.3, §6.3).

Two backends:

* ``--backend count`` (default): the pure-Python stupid-backoff n-gram model.
  Trains by counting; saves a pickle.  Zero third-party deps.
* ``--backend kenlm``: prints the KenLM command line to build a pruned binary
  from your corpus (needs the KenLM toolkit).  KenLM is the production choice —
  smaller and faster — but the count model is a drop-in for development.

    python scripts/train_lm.py corpus.txt --order 3 --out data/lm/vi.count.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.postprocess import CountLanguageModel


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
    ap.add_argument("--backend", choices=["count", "kenlm"], default="count")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--out", default="data/lm/vi.count.pkl")
    ap.add_argument("--prune", default="0 0 1",
                    help="kenlm pruning (counts per order)")
    args = ap.parse_args()

    if args.backend == "kenlm":
        corpus = " ".join(args.corpus)
        print("# Build a pruned KenLM binary (needs the kenlm toolkit):")
        print(f"lmplz -o {args.order} --prune {args.prune} < {corpus} "
              f"> data/lm/vi.arpa")
        print("build_binary data/lm/vi.arpa data/lm/vi.binary")
        print("# then: KenLMModel('data/lm/vi.binary')")
        return 0

    lm = CountLanguageModel(order=args.order)
    lm.train(_lines(args.corpus))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(lm, f)
    print(f"trained {args.order}-gram count LM "
          f"(vocab {len(lm._vocab)}, {lm._total} tokens) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
