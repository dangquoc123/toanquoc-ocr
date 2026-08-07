"""Tone-entropy meter — locating the Fano floor (Design §0, §8).

The whole "no LLM" argument rests on one measurable quantity: the conditional
entropy of the tone given context, ``H(t | context)``.  When it is small
(§0 estimates 0.05–0.15 bit for Vietnamese) a bigram MAP decoder sits within
~1.35× of the information-theoretic limit, and a 7B LLM cannot beat that floor
by enough to matter.

This module measures three quantities on a plain-text corpus, so you can decide
*before* training whether a bigram is enough or you need a trigram:

* ``H(t)``          — marginal tone entropy.
* ``H(t | s)``      — tone given the toneless syllable skeleton (segmental
                      spelling with the tone stripped).
* ``H(t | s, w₋₁)`` — tone given skeleton **and** previous word (the bigram).

All three use the Miller–Madow bias correction, since plug-in entropy is biased
low on finite samples.  Conditional entropies are computed as
``H(X, Y) − H(X)`` with the correction applied to each term.  Pure stdlib.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from ..charset.decompose import Tone, compose, decompose
from ..charset.normalize import normalize_text

__all__ = ["EntropyReport", "measure_entropy", "syllable_tone", "syllable_skeleton"]

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(normalize_text(text))]


def syllable_tone(word: str) -> Tone:
    """The single tone class of a Vietnamese syllable (level if none)."""
    for ch in word:
        t = decompose(ch).tone
        if t != Tone.NONE:
            return t
    return Tone.NONE


def syllable_skeleton(word: str) -> str:
    """The syllable with its tone removed — base+modifier spelling (``s``)."""
    out = []
    for ch in word:
        d = decompose(ch)
        # keep base + modifier, drop tone
        out.append(compose(d.base, d.modifier, Tone.NONE))
    return "".join(out)


def entropy_mm(counts: Iterable[int]) -> float:
    """Miller–Madow-corrected Shannon entropy (bits) from category counts."""
    counts = [c for c in counts if c > 0]
    n = sum(counts)
    if n == 0:
        return 0.0
    plugin = -sum((c / n) * math.log2(c / n) for c in counts)
    k = len(counts)  # number of observed categories with positive mass
    return plugin + (k - 1) / (2 * n * math.log(2))  # /ln2 → bits


def _cond_entropy(joint: Counter, marginal: Counter) -> float:
    """H(Y|X) = H(X,Y) - H(X), each Miller–Madow corrected."""
    return entropy_mm(joint.values()) - entropy_mm(marginal.values())


@dataclass
class EntropyReport:
    H_t: float
    H_t_given_s: float
    H_t_given_s_wprev: float
    n_syllables: int

    def bigram_suffices(self) -> bool:
        """§8 rule of thumb: ≤0.15 bit ⇒ bigram floor is low enough."""
        return self.H_t_given_s_wprev <= 0.15

    def summary(self) -> str:
        verdict = ("bigram is enough" if self.bigram_suffices()
                   else "consider a trigram")
        return (
            f"H(t)              = {self.H_t:.4f} bit\n"
            f"H(t | s)          = {self.H_t_given_s:.4f} bit\n"
            f"H(t | s, w-1)     = {self.H_t_given_s_wprev:.4f} bit  → {verdict}\n"
            f"syllables         = {self.n_syllables}"
        )


def measure_entropy(corpus: Iterable[str]) -> EntropyReport:
    """Measure marginal and conditional tone entropy over a text corpus.

    ``corpus`` is any iterable of text lines/documents.  The plainer and larger
    the better (Wikipedia dump + news, Design §6.3); no images or labels needed.
    """
    c_t: Counter = Counter()
    c_st: Counter = Counter()      # (skeleton, tone)
    c_s: Counter = Counter()       # skeleton
    c_wst: Counter = Counter()     # (prev_word, skeleton, tone)
    c_ws: Counter = Counter()      # (prev_word, skeleton)
    n = 0

    for line in corpus:
        prev = "<s>"
        for w in _tokenize(line):
            t = int(syllable_tone(w))
            s = syllable_skeleton(w)
            c_t[t] += 1
            c_st[(s, t)] += 1
            c_s[s] += 1
            c_wst[(prev, s, t)] += 1
            c_ws[(prev, s)] += 1
            prev = w
            n += 1

    return EntropyReport(
        H_t=entropy_mm(c_t.values()),
        H_t_given_s=_cond_entropy(c_st, c_s),
        H_t_given_s_wprev=_cond_entropy(c_wst, c_ws),
        n_syllables=n,
    )
