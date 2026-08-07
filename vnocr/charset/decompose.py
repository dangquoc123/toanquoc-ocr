"""Factorised Vietnamese character model — the heart of "Lever 1" (Design §3.2).

A precomposed Vietnamese letter such as ``ế`` carries three independent pieces
of information:

* **base**      — the underlying Latin letter (``e``), casing preserved.
* **modifier**  — the *vowel shape* diacritic (circumflex ``â/ê/ô``, breve
                  ``ă``, horn ``ơ/ư``) or the ``đ`` stroke.
* **tone**      — the tone mark (sắc / huyền / hỏi / ngã / nặng), or none.

Flat classification over ~230 composed classes starves the rare glyphs
(``ữ``, ``ặ``, ``ỡ`` …).  By factorising the final logit layer into
``base + modifier + tone`` every tone-bearing sample trains the *tilde* vector
regardless of its base letter, lifting the effective sample size of the rare
combinations by 2–3 orders of magnitude.

This module is the ground-truth reference implementation of that factorisation.
It depends only on :mod:`unicodedata` (stdlib) so it runs anywhere, and it is
what the training-label pipeline and the evaluation metrics (Design §8) both
build on.

The mapping is *exact and reversible*::

    base, mod, tone = decompose(ch)
    assert compose(base, mod, tone) == unicodedata.normalize("NFC", ch)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

__all__ = [
    "Tone",
    "Modifier",
    "Decomp",
    "decompose",
    "compose",
    "is_valid",
    "strip_tone",
    "strip_all",
]


# --- Combining code points (Unicode NFD) ----------------------------------
# Tone marks.  Level tone (ngang) has no combining mark.
_CB_GRAVE = "̀"   # huyền
_CB_ACUTE = "́"   # sắc
_CB_TILDE = "̃"   # ngã
_CB_HOOK = "̉"    # hỏi (hook above)
_CB_DOT = "̣"     # nặng (dot below)

# Vowel-shape marks.
_CB_CIRCUMFLEX = "̂"  # â ê ô
_CB_BREVE = "̆"       # ă
_CB_HORN = "̛"        # ơ ư


class Tone(IntEnum):
    """Six tone classes.  ``NONE`` is the level tone (ngang)."""

    NONE = 0   # ngang
    ACUTE = 1  # sắc
    GRAVE = 2  # huyền
    HOOK = 3   # hỏi
    TILDE = 4  # ngã
    DOT = 5    # nặng


class Modifier(IntEnum):
    """Five vowel-shape / letter-shape classes."""

    NONE = 0        # plain
    BREVE = 1       # ă
    CIRCUMFLEX = 2  # â ê ô
    HORN = 3        # ơ ư
    STROKE = 4      # đ  (handled specially — no NFD decomposition exists)


_TONE_FROM_MARK = {
    _CB_ACUTE: Tone.ACUTE,
    _CB_GRAVE: Tone.GRAVE,
    _CB_HOOK: Tone.HOOK,
    _CB_TILDE: Tone.TILDE,
    _CB_DOT: Tone.DOT,
}
_MARK_FROM_TONE = {v: k for k, v in _TONE_FROM_MARK.items()}

_MOD_FROM_MARK = {
    _CB_BREVE: Modifier.BREVE,
    _CB_CIRCUMFLEX: Modifier.CIRCUMFLEX,
    _CB_HORN: Modifier.HORN,
}
_MARK_FROM_MOD = {v: k for k, v in _MOD_FROM_MARK.items()}

# Vowels that legally accept a tone mark.
VOWELS = set("aeiouyAEIOUY")

# Which base letters accept which shape modifier (casing-agnostic key = lower).
_MOD_ALLOWED = {
    "a": {Modifier.BREVE, Modifier.CIRCUMFLEX},
    "e": {Modifier.CIRCUMFLEX},
    "o": {Modifier.CIRCUMFLEX, Modifier.HORN},
    "u": {Modifier.HORN},
    "d": {Modifier.STROKE},
}


@dataclass(frozen=True)
class Decomp:
    """A factorised character: ``base`` letter + ``modifier`` + ``tone``.

    ``base`` keeps its original case; the model predicts base and case
    together (case is part of the ~70-way base head, Design §3.2).
    """

    base: str
    modifier: Modifier = Modifier.NONE
    tone: Tone = Tone.NONE

    def as_tuple(self) -> Tuple[str, int, int]:
        return (self.base, int(self.modifier), int(self.tone))


def decompose(ch: str) -> Decomp:
    """Split a single character into ``(base, modifier, tone)``.

    Non-Vietnamese characters (digits, punctuation, Latin without diacritics,
    whitespace) pass through with ``NONE`` modifier and tone.
    """
    if len(ch) != 1:
        # Accept an already-composed grapheme cluster of length 1 only; callers
        # that pass multi-codepoint strings should iterate graphemes upstream.
        ch = unicodedata.normalize("NFC", ch)
        if len(ch) != 1:
            raise ValueError(f"decompose expects a single character, got {ch!r}")

    # đ / Đ is a stroke letter with no canonical NFD decomposition.
    if ch in ("đ", "Đ"):
        return Decomp(base="d" if ch == "đ" else "D", modifier=Modifier.STROKE)

    nfd = unicodedata.normalize("NFD", ch)
    base = nfd[0]
    modifier = Modifier.NONE
    tone = Tone.NONE
    for mark in nfd[1:]:
        if mark in _TONE_FROM_MARK:
            tone = _TONE_FROM_MARK[mark]
        elif mark in _MOD_FROM_MARK:
            modifier = _MOD_FROM_MARK[mark]
        # Unknown combining marks are dropped intentionally: the factorised head
        # models only the Vietnamese inventory.
    return Decomp(base=base, modifier=modifier, tone=tone)


def compose(base: str, modifier: Modifier | int = Modifier.NONE,
            tone: Tone | int = Tone.NONE) -> str:
    """Inverse of :func:`decompose` — rebuild a precomposed NFC character."""
    modifier = Modifier(int(modifier))
    tone = Tone(int(tone))

    if modifier == Modifier.STROKE:
        # Only d/D → đ/Đ; a stroke never coexists with a tone in Vietnamese.
        return "đ" if base == "d" else "Đ" if base == "D" else base

    seq = base
    if modifier in _MARK_FROM_MOD:
        seq += _MARK_FROM_MOD[modifier]
    if tone in _MARK_FROM_TONE:
        seq += _MARK_FROM_TONE[tone]
    return unicodedata.normalize("NFC", seq)


def is_valid(base: str, modifier: Modifier | int = Modifier.NONE,
             tone: Tone | int = Tone.NONE) -> bool:
    """The combination mask ``M(b, m, t) ∈ {0, 1}`` from Design §3.2.

    Returns ``True`` only for combinations that exist in written Vietnamese,
    e.g. rejects *breve + ô*, or a tone on a consonant.  The recogniser
    multiplies the factorised soft-max by this mask so impossible glyphs get
    zero probability.
    """
    modifier = Modifier(int(modifier))
    tone = Tone(int(tone))
    low = base.lower()

    if modifier == Modifier.STROKE:
        # đ: no tone, base must be d.
        return low == "d" and tone == Tone.NONE
    if modifier != Modifier.NONE:
        if low not in _MOD_ALLOWED or modifier not in _MOD_ALLOWED[low]:
            return False
    if tone != Tone.NONE:
        # Only vowels bear tones.
        if base not in VOWELS:
            return False
    return True


def strip_tone(ch: str) -> str:
    """Return ``ch`` with its tone removed (base + modifier kept)."""
    d = decompose(ch)
    return compose(d.base, d.modifier, Tone.NONE)


def strip_all(ch: str) -> str:
    """Return the bare base letter (no modifier, no tone)."""
    return decompose(ch).base
