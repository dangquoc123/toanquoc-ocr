"""OCR accuracy metrics with the Vietnamese-specific split (Design §8).

Beyond aggregate CER/WER we report the two numbers that actually drive this
project's design decisions:

* ``p_B`` — base-glyph accuracy (is the letter frame right?)
* ``p_T`` — tone accuracy (is the tone mark right, on tone-bearing vowels?)

The foundational assumption (Design §0) is that errors are *tone-dominated*:
``1 - p_T  ≫  1 - p_B``.  These metrics are how you confirm that on your own
data and decide where to spend effort — no public benchmark reports them, so
you have to build it, which is what this module is.

Character alignment is a standard Levenshtein (Wagner–Fischer) trace; on aligned
character pairs we compare the *factorised* base / modifier / tone separately.
Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..charset.decompose import VOWELS, Tone, decompose
from ..charset.normalize import normalize_text

__all__ = ["Metrics", "evaluate", "cer", "wer", "align"]


# Alignment op codes.
_MATCH, _SUB, _INS, _DEL = "M", "S", "I", "D"


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Levenshtein alignment trace between two token sequences.

    Returns a list of ``(op, i, j)`` where ``i`` indexes ``ref`` and ``j``
    indexes ``hyp`` (or ``-1`` when not applicable).
    """
    n, m = len(ref), len(hyp)
    # cost matrix + backtrace
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)

    ops: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((_MATCH if ref[i - 1] == hyp[j - 1] else _SUB, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append((_DEL, i - 1, -1))
            i -= 1
        else:
            ops.append((_INS, -1, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _edit_distance(ref: Sequence, hyp: Sequence) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, normalize: bool = True) -> float:
    """Character error rate.  Normalises (NFC + tone placement) first."""
    if normalize:
        ref, hyp = normalize_text(ref), normalize_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(list(ref), list(hyp)) / len(ref)


def wer(ref: str, hyp: str, normalize: bool = True) -> float:
    """Word error rate over whitespace tokens."""
    if normalize:
        ref, hyp = normalize_text(ref), normalize_text(hyp)
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


@dataclass
class Metrics:
    """Corpus-level metrics.  Rates are errors / reference-units in [0, 1]."""

    cer: float
    wer: float
    base_error_rate: float   # 1 - p_B
    tone_error_rate: float   # 1 - p_T  (over tone-bearing ref vowels)
    modifier_error_rate: float
    n_chars: int
    n_tone_vowels: int

    @property
    def p_B(self) -> float:
        return 1.0 - self.base_error_rate

    @property
    def p_T(self) -> float:
        return 1.0 - self.tone_error_rate

    def summary(self) -> str:
        return (
            f"CER={self.cer:.4f}  WER={self.wer:.4f}\n"
            f"base  p_B={self.p_B:.4f}  (err {self.base_error_rate:.4f})\n"
            f"tone  p_T={self.p_T:.4f}  (err {self.tone_error_rate:.4f}, "
            f"n={self.n_tone_vowels})\n"
            f"mod   err={self.modifier_error_rate:.4f}\n"
            f"chars n={self.n_chars}"
        )


def evaluate(refs: Sequence[str], hyps: Sequence[str],
             normalize: bool = True) -> Metrics:
    """Aggregate CER/WER and the factorised p_B / p_T over a corpus.

    A *base* error is any aligned substitution whose base letter differs, plus
    every insertion/deletion.  A *tone* error is counted only on reference
    characters that are tone-bearing vowels: an aligned pair with the wrong
    tone, or a deleted vowel, is an error; the denominator is the number of
    tone-bearing reference vowels.  This is what isolates §0's claim that
    ``1 - p_T ≫ 1 - p_B``.
    """
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must be the same length")

    total_cer_num = total_cer_den = 0
    total_wer_num = total_wer_den = 0
    base_err = 0
    mod_err = 0
    tone_err = 0
    n_chars = 0
    n_tone_vowels = 0

    for ref, hyp in zip(refs, hyps):
        if normalize:
            ref, hyp = normalize_text(ref), normalize_text(hyp)

        total_cer_num += _edit_distance(list(ref), list(hyp))
        total_cer_den += len(ref)
        r_words, h_words = ref.split(), hyp.split()
        total_wer_num += _edit_distance(r_words, h_words)
        total_wer_den += len(r_words)

        n_chars += len(ref)
        for op, i, j in align(list(ref), list(hyp)):
            if op == _INS:
                continue  # spurious hyp char: penalised by CER, not a ref base
            rd = decompose(ref[i])
            # tone-bearing vowel test is on the *base* — "ô", "ệ" etc. are not
            # in the bare-vowel set but their base ("o", "e") is.
            is_tone_vowel = rd.base in VOWELS
            if is_tone_vowel:
                n_tone_vowels += 1
            if op == _DEL:
                base_err += 1
                if rd.modifier != 0:
                    mod_err += 1
                if is_tone_vowel:
                    tone_err += 1
                continue
            hd = decompose(hyp[j])
            if rd.base != hd.base:
                base_err += 1
            if int(rd.modifier) != int(hd.modifier):
                mod_err += 1
            if is_tone_vowel and int(rd.tone) != int(hd.tone):
                tone_err += 1

    return Metrics(
        cer=total_cer_num / total_cer_den if total_cer_den else 0.0,
        wer=total_wer_num / total_wer_den if total_wer_den else 0.0,
        base_error_rate=base_err / n_chars if n_chars else 0.0,
        tone_error_rate=tone_err / n_tone_vowels if n_tone_vowels else 0.0,
        modifier_error_rate=mod_err / n_chars if n_chars else 0.0,
        n_chars=n_chars,
        n_tone_vowels=n_tone_vowels,
    )
