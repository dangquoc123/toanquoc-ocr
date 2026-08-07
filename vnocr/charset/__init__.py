"""Vietnamese character modelling — the factorised base/modifier/tone core.

This subpackage is the linguistic foundation shared by the recogniser head
(Design §3.2), the label pipeline (§1.6) and the evaluation metrics (§8).  It is
pure stdlib so it runs and is unit-tested without torch, numpy or a GPU.
"""

from .charset import Charset, default_charset
from .decompose import (
    Decomp,
    Modifier,
    Tone,
    compose,
    decompose,
    is_valid,
    strip_all,
    strip_tone,
)
from .normalize import nfc, normalize_text, place_tone, reposition_tone_word
from .syllables import generate_syllables

__all__ = [
    "Charset",
    "default_charset",
    "Decomp",
    "Modifier",
    "Tone",
    "compose",
    "decompose",
    "is_valid",
    "strip_all",
    "strip_tone",
    "nfc",
    "normalize_text",
    "place_tone",
    "reposition_tone_word",
    "generate_syllables",
]
