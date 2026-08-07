"""Unicode / orthographic label normalisation (Design §1.6).

Two different byte strings can be the *same* Vietnamese text:

* Composition form: ``ế`` = U+1EBF (precomposed) **or** ``e`` + U+0302 + U+0301
  (combining).  → fix with NFC.
* Tone placement: ``hòa`` (dictionary/official style, mark on the o) vs
  ``hoà`` (phonetic "new" style, mark on the a).  Same word, different code
  points.  We canonicalise to the dominant dictionary style.

If labels and predictions disagree on either convention, measured CER is
inflated for free.  Every training label, every ground-truth string, and every
model output is pushed through :func:`normalize_text` before comparison.

The NFC step is exact and always safe.  The tone-repositioning step implements
the standard modern placement rule and is best-effort for rare triphthongs; it
never changes the *identity* of a syllable, only which vowel carries the mark.
"""

from __future__ import annotations

import unicodedata
from typing import List

from .decompose import Modifier, Tone, VOWELS, compose, decompose

__all__ = ["nfc", "normalize_text", "reposition_tone_word", "place_tone"]


def nfc(s: str) -> str:
    """Canonical NFC composition — the always-safe baseline normalisation."""
    return unicodedata.normalize("NFC", s)


def _is_alpha_vn(ch: str) -> bool:
    return ch.isalpha()


def reposition_tone_word(word: str) -> str:
    """Move the tone mark of a single syllable to its modern canonical vowel.

    ``hoà`` → ``hòa``, ``thuý`` → ``thúy``, ``quí`` → ``quý``.  Words with no
    tone, or non-syllable tokens, are returned unchanged (after NFC).
    """
    word = nfc(word)
    if not word or not any(_is_alpha_vn(c) for c in word):
        return word

    try:
        decs = [decompose(c) for c in word]
    except ValueError:
        return word

    tone_positions = [i for i, d in enumerate(decs) if d.tone != Tone.NONE]
    if not tone_positions:
        return word
    tone = decs[tone_positions[0]].tone

    run_lower = "".join(d.base.lower() for d in decs)
    vowel_idx = [i for i, d in enumerate(decs) if d.base in VOWELS]

    # Glides: the u in "qu…" and the i in "gi…+vowel" are consonantal.
    skip = set()
    if run_lower.startswith("qu") and len(decs) > 1 and decs[1].base.lower() == "u":
        skip.add(1)
    elif run_lower.startswith("gi") and len(decs) > 1 and decs[1].base.lower() == "i":
        if any(j > 1 for j in vowel_idx if j != 1):
            skip.add(1)
    nucleus = [i for i in vowel_idx if i not in skip]
    if not nucleus:
        nucleus = vowel_idx  # degenerate (e.g. bare "gì")
    if not nucleus:
        return word

    target = _canonical_index(decs, nucleus)

    out: List[str] = []
    for i, d in enumerate(decs):
        t = tone if i == target else Tone.NONE
        out.append(compose(d.base, d.modifier, t))
    return "".join(out)


def place_tone(word: str, tone: Tone | int) -> str:
    """Attach ``tone`` to the canonical vowel of a *toneless* syllable.

    Used by the syllable-inventory generator to spell every tone of a rime.
    ``place_tone("hoa", Tone.GRAVE) -> "hòa"``.
    """
    tone = Tone(int(tone))
    word = nfc(word)
    try:
        decs = [decompose(c) for c in word]
    except ValueError:
        return word
    vowel_idx = [i for i, d in enumerate(decs) if d.base in VOWELS]
    if not vowel_idx:
        return word
    run_lower = "".join(d.base.lower() for d in decs)
    skip = set()
    if run_lower.startswith("qu") and len(decs) > 1 and decs[1].base.lower() == "u":
        skip.add(1)
    elif run_lower.startswith("gi") and len(decs) > 1 and decs[1].base.lower() == "i":
        if any(j > 1 for j in vowel_idx if j != 1):
            skip.add(1)
    nucleus = [i for i in vowel_idx if i not in skip] or vowel_idx
    target = _canonical_index(decs, nucleus)
    out = []
    for i, d in enumerate(decs):
        t = tone if i == target else Tone.NONE
        out.append(compose(d.base, d.modifier, t))
    return "".join(out)


def _canonical_index(decs, nucleus: List[int]) -> int:
    # 1. A shape-marked vowel always carries the tone ( â ê ô ă ơ ư).
    #    For the double-horn rime "ươ" the tone sits on the second (ơ).
    marked = [i for i in nucleus if decs[i].modifier != Modifier.NONE]
    if marked:
        return marked[-1]

    if len(nucleus) == 1:
        return nucleus[0]

    last_vowel = nucleus[-1]
    has_coda = last_vowel != len(decs) - 1

    if len(nucleus) == 2:
        # Open two-vowel rime → tone on the FIRST vowel.  This is the dominant
        # dictionary/official convention: hòa (mark on o), khỏe (o), thủy (u),
        # múa (u), mía (i).  The alternative "phonetic/new" style (hoà, thuỷ) is
        # what we normalise *away from* so a word has one canonical spelling.
        # Closed two-vowel rime → tone on the second vowel: toán, hoàn, xuân.
        return nucleus[0] if not has_coda else nucleus[1]

    # Triphthong → tone on the middle vowel: xoài, khuỷu, ngoằn.
    return nucleus[1]


def normalize_text(s: str, reposition: bool = True) -> str:
    """Full label normalisation: NFC, then canonical tone placement per token.

    Parameters
    ----------
    s : str
        Raw text (a label, a ground-truth line, or a model output).
    reposition : bool
        When ``True`` also convert phonetic-style tone placement to the
        dominant dictionary style (``hoà`` → ``hòa``).  Set ``False`` to keep
        NFC only, e.g. when your corpus is already consistent and you want the
        cheapest possible path.
    """
    s = nfc(s)
    if not reposition:
        return s

    # Split on runs of alphabetic characters so punctuation/spacing survive
    # verbatim while each syllable is repositioned independently.
    out: List[str] = []
    buf: List[str] = []
    for ch in s:
        if _is_alpha_vn(ch):
            buf.append(ch)
        else:
            if buf:
                out.append(reposition_tone_word("".join(buf)))
                buf = []
            out.append(ch)
    if buf:
        out.append(reposition_tone_word("".join(buf)))
    return "".join(out)
