"""Noisy-channel correction of frame errors (Design §5.2).

When a recognised token is *not* a valid syllable it is almost surely a frame
error.  We correct it by Bayesian decoding over the valid inventory::

    ŷ = argmax_y  p(o | y) · p(y)

* ``p(o | y)`` is a visual-confusion-weighted edit distance.  The substitution
  cost ``w(a→b) = −log p(observe b | true a)`` is *learned from an empirical
  confusion matrix* on validation data (Design §5.2 is explicit: not hand-tuned).
  Until you have that matrix, a linguistically-informed default is used that
  makes the known visual confusions cheap: hỏi↔ngã, ơ↔o, ư↔u, ê↔e.
* ``p(y)`` is a unigram prior restricted to valid syllables (uniform if you have
  no frequency list yet).

Candidates come from :meth:`SyllableTrie.candidates_within`, so the search is
restricted to real syllables from the start.  Pure stdlib.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..charset.decompose import decompose
from ..charset.normalize import nfc
from .trie import SyllableTrie

__all__ = ["ConfusionModel", "NoisyChannel"]

_NEG = float("inf")


@dataclass
class ConfusionModel:
    """Substitution / indel costs in nats (``−log`` probabilities).

    Costs are derived from the factorised character model so a wrong tone is
    cheap and a wrong base is dear — matching how OCR actually fails on
    Vietnamese.  Override any of these, or rebuild the whole thing from counts
    with :meth:`from_confusion_counts`.
    """

    # default per-factor substitution costs (nats)
    tone_sub: float = 1.2          # generic wrong tone
    tone_sub_hoi_nga: float = 0.5  # the notorious hỏi↔ngã pair — cheapest
    modifier_sub: float = 1.6      # ơ↔o, ư↔u, ê↔e, ...
    base_sub: float = 4.0          # wrong base letter — expensive
    base_sub_visual: float = 2.3   # visually close bases (see _VISUAL_BASE)
    insert: float = 3.0
    delete: float = 3.0

    # learned table overrides everything when present: (a, b) -> cost
    table: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # visually confusable *base* letters (shape-similar in print).
    _VISUAL_BASE = {
        frozenset("cо"), frozenset("oо"), frozenset("il"), frozenset("nh"),
        frozenset("uv"), frozenset("rn"), frozenset("cе"), frozenset("gq"),
        frozenset("ij"), frozenset("mn"),
    }

    def sub(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        if (a, b) in self.table:
            return self.table[(a, b)]

        da, db = decompose(a), decompose(b)
        # same base + modifier, differ only in tone → tone confusion
        if da.base.lower() == db.base.lower() and da.modifier == db.modifier:
            pair = {int(da.tone), int(db.tone)}
            if pair == {3, 4}:  # HOOK(3) ↔ TILDE(4)
                return self.tone_sub_hoi_nga
            return self.tone_sub
        # same base, differ in modifier (± tone) → modifier confusion
        if da.base.lower() == db.base.lower():
            return self.modifier_sub
        # different base
        if frozenset((da.base.lower(), db.base.lower())) in self._VISUAL_BASE:
            return self.base_sub_visual
        return self.base_sub

    @classmethod
    def from_confusion_counts(cls, counts: Dict[Tuple[str, str], int],
                              smoothing: float = 0.5) -> "ConfusionModel":
        """Build ``w(a→b) = −log p(b | a)`` from empirical (true, obs) counts.

        ``counts[(a, b)]`` is how often true char ``a`` was read as ``b`` on the
        validation set.  This is the §5.2-preferred, non-hand-tuned path.
        """
        totals: Counter = Counter()
        for (a, _b), c in counts.items():
            totals[a] += c
        table: Dict[Tuple[str, str], float] = {}
        vocab = {a for a, _ in counts} | {b for _, b in counts}
        for a in vocab:
            denom = totals[a] + smoothing * len(vocab)
            for b in vocab:
                if a == b:
                    continue
                c = counts.get((a, b), 0)
                p = (c + smoothing) / denom if denom else 1e-9
                table[(a, b)] = -math.log(max(p, 1e-9))
        return cls(table=table)


class NoisyChannel:
    def __init__(self, trie: SyllableTrie,
                 confusion: Optional[ConfusionModel] = None,
                 unigram_logprob: Optional[Dict[str, float]] = None,
                 max_edits: int = 2) -> None:
        self.trie = trie
        self.confusion = confusion or ConfusionModel()
        self.unigram = unigram_logprob or {}
        self.max_edits = max_edits

    def _channel_cost(self, observed: str, candidate: str) -> float:
        """Weighted edit distance −log p(observed | candidate)."""
        cm = self.confusion
        n, m = len(observed), len(candidate)
        d = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            d[i][0] = i * cm.delete
        for j in range(1, m + 1):
            d[0][j] = j * cm.insert
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i][j] = min(
                    d[i - 1][j - 1] + cm.sub(candidate[j - 1], observed[i - 1]),
                    d[i - 1][j] + cm.delete,
                    d[i][j - 1] + cm.insert,
                )
        return d[n][m]

    def correct_token(self, token: str) -> Tuple[str, bool]:
        """Return ``(best, changed)``.  Valid or non-syllable tokens pass through.

        The correction preserves the original casing pattern of ``token``.
        """
        word = nfc(token)
        core = word.lower()
        if not core.isalpha() or self.trie.contains(core):
            return word, False

        candidates = self.trie.candidates_within(core, self.max_edits)
        if not candidates:
            return word, False

        best, best_cost = None, _NEG
        for cand in candidates:
            channel = self._channel_cost(core, cand)
            prior = -self.unigram.get(cand, math.log(1e-6))  # −log p(y)
            cost = channel + prior
            if cost < best_cost:
                best, best_cost = cand, cost
        if best is None:
            return word, False
        return _restore_case(word, best), best != core


def _restore_case(template: str, lowered: str) -> str:
    """Apply ``template``'s capitalisation pattern onto ``lowered``."""
    if template.isupper():
        return lowered.upper()
    if template[:1].isupper():
        return lowered[:1].upper() + lowered[1:]
    return lowered
