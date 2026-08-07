"""The factorised character vocabulary and the validity mask (Design §3.2).

A :class:`Charset` holds three small class lists —

* ``bases``      the ~90 base symbols (a-z, A-Z, 0-9, punctuation, space),
* ``modifiers``  the 5 vowel/letter shape classes,
* ``tones``      the 6 tone classes —

and the derived objects the recogniser needs:

* :meth:`encode` / :meth:`decode` between a composed character and its
  ``(base_idx, mod_idx, tone_idx)`` triple;
* :meth:`build_mask`, the ``M(b, m, t) ∈ {0,1}`` tensor that zeroes impossible
  glyphs before the soft-max;
* :meth:`flat_alphabet`, every *valid* composed character, for the flat-CTC
  baseline used in the §3.2 ablation.

Pure stdlib; the torch bridge lives in :mod:`vnocr.recognize.head`.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .decompose import Modifier, Tone, compose, decompose, is_valid

__all__ = ["Charset", "default_charset"]


# Default base inventory.  Both cases are real bases: "Ế" decomposes to base "E".
# f/j/w/z are non-native but appear in loanwords — kept for robustness.
_LETTERS = string.ascii_lowercase + string.ascii_uppercase
_DIGITS = string.digits
# Punctuation seen in administrative Vietnamese text.  Space is index-0-friendly
# but we keep it explicit; the CTC blank is separate (see recognize.ctc).
_PUNCT = " .,;:!?…'\"“”‘’(){}[]-–—/\\%°&@#*+=<>|~^_$"


@dataclass
class Charset:
    bases: List[str]
    modifiers: List[Modifier] = field(
        default_factory=lambda: list(Modifier))
    tones: List[Tone] = field(default_factory=lambda: list(Tone))

    _base_index: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # de-dup while preserving order
        seen = set()
        uniq = []
        for b in self.bases:
            if b not in seen:
                seen.add(b)
                uniq.append(b)
        self.bases = uniq
        self._base_index = {b: i for i, b in enumerate(self.bases)}

    # --- sizes -----------------------------------------------------------
    @property
    def n_base(self) -> int:
        return len(self.bases)

    @property
    def n_mod(self) -> int:
        return len(self.modifiers)

    @property
    def n_tone(self) -> int:
        return len(self.tones)

    # --- encode / decode -------------------------------------------------
    def encode(self, ch: str) -> Tuple[int, int, int]:
        """Composed character → ``(base_idx, mod_idx, tone_idx)``.

        Raises ``KeyError`` if the base is outside the inventory.
        """
        d = decompose(ch)
        b = self._base_index[d.base]
        return b, int(d.modifier), int(d.tone)

    def decode(self, base_idx: int, mod_idx: int, tone_idx: int) -> str:
        """``(base_idx, mod_idx, tone_idx)`` → composed NFC character."""
        return compose(self.bases[base_idx], Modifier(mod_idx), Tone(tone_idx))

    def contains(self, ch: str) -> bool:
        try:
            self.encode(ch)
            return True
        except KeyError:
            return False

    # --- mask and alphabet ----------------------------------------------
    def build_mask(self) -> List[List[List[int]]]:
        """``M[b][m][t] ∈ {0,1}`` — 1 iff the combination is real Vietnamese.

        The recogniser adds ``log M`` (i.e. ``-inf`` where ``M==0``) to the
        factorised score before soft-max, so ``ố + breve`` and "tone on a
        consonant" receive exactly zero probability (Design §3.2).
        """
        mask: List[List[List[int]]] = []
        for base in self.bases:
            per_base: List[List[int]] = []
            for m in self.modifiers:
                per_mod = [1 if is_valid(base, m, t) else 0 for t in self.tones]
                per_base.append(per_mod)
            mask.append(per_base)
        return mask

    def flat_alphabet(self) -> List[str]:
        """Every valid composed character — vocabulary of the flat-CTC baseline."""
        out: List[str] = []
        seen = set()
        for base in self.bases:
            for m in self.modifiers:
                for t in self.tones:
                    if not is_valid(base, m, t):
                        continue
                    ch = compose(base, m, t)
                    if ch not in seen:
                        seen.add(ch)
                        out.append(ch)
        return out

    def encode_string(self, s: str) -> List[Tuple[int, int, int]]:
        return [self.encode(ch) for ch in s]

    def decode_triples(self, triples: Sequence[Tuple[int, int, int]]) -> str:
        return "".join(self.decode(*t) for t in triples)


def default_charset() -> Charset:
    """The stock charset: full Latin (both cases), digits, admin punctuation."""
    bases = list(_LETTERS + _DIGITS) + list(_PUNCT)
    return Charset(bases=bases)
