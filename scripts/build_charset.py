#!/usr/bin/env python3
"""Build the syllable inventory and print charset statistics (Design §3.4, §5.1).

    python scripts/build_charset.py --out data/charset/syllables.txt

Generates the phonotactic syllable set (over-generating, intersect with a corpus
frequency list for the tight ~7k inventory — see data/charset/README.md), and
reports the factorised-vs-flat class counts that motivate Lever 1 (§3.2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.charset import default_charset, generate_syllables


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/charset/syllables.txt",
                    help="where to write the syllable list")
    args = ap.parse_args()

    syllables = generate_syllables()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(syllables) + "\n", encoding="utf-8")

    cs = default_charset()
    n_base, n_mod, n_tone = cs.n_base, cs.n_mod, cs.n_tone
    n_valid = len(cs.flat_alphabet())

    print(f"syllables written : {len(syllables):>6}  -> {out}")
    print(f"base classes      : {n_base:>6}")
    print(f"modifier classes  : {n_mod:>6}")
    print(f"tone classes      : {n_tone:>6}")
    print(f"factorised params : {n_base + n_mod + n_tone:>6}  "
          f"(u + v + w heads)")
    print(f"flat classes      : {n_valid:>6}  (valid composed glyphs)")
    print(f"→ Lever 1 collapses a {n_valid}-way soft-max into "
          f"{n_base}+{n_mod}+{n_tone} shared factors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
