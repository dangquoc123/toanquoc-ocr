"""Vietnamese syllable inventory (Design §5.1).

Vietnamese has on the order of ~7,000 phonotactically valid syllables.  Because
that inventory is small and dense, a recognised "syllable" that falls outside it
is almost certainly a frame error — the structural redundancy that Chinese and
English lack (Design §5.1).  The syllable set is therefore both a *validity
filter* and the vocabulary of the post-processing trie.

This module **generates** a syllable set from a phonotactic table:

    onset × rime × tone   →  spell out  →  apply orthographic spelling rules

The generator intentionally *over-generates* a little (some onset+rime pairs it
allows never actually occur).  For a validity filter that is the safe direction:
a superset never rejects a real word.  In production you intersect this set with
a corpus frequency list (Design §6.3) to get the tight ~7k inventory and, for
free, the unigram priors the noisy-channel model wants.

Everything here is pure stdlib and deterministic.
"""

from __future__ import annotations

from typing import Iterable, List, Set

from .decompose import Tone
from .normalize import nfc, place_tone

__all__ = ["generate_syllables", "ONSETS", "RIMES", "TONES"]


# Initial consonants (âm đầu).  "" is the zero onset.
ONSETS: List[str] = [
    "", "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh", "l",
    "m", "n", "ng", "ngh", "nh", "p", "ph", "qu", "r", "s", "t", "th",
    "tr", "v", "x",
]

# Written rimes (vần) without tone, in modern orthography.  Grouped by coda so
# the checked-syllable tone rule below is easy to apply.  Not exhaustive, but
# covers the overwhelming majority of running-text rimes.
_OPEN = [
    "a", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
    "ia", "ua", "ưa",
    "oa", "oe", "uê", "uơ", "uy", "uya",
    "ai", "ao", "au", "ay", "âu", "ây", "eo", "êu", "iu", "iêu", "yêu",
    "oi", "ôi", "ơi", "ui", "ưi", "ưu", "uôi", "ươi", "ươu",
    "oai", "oao", "oay", "uôi", "uây", "uyu", "oeo",
]
_CODA_M = [
    "am", "ăm", "âm", "em", "êm", "im", "om", "ôm", "ơm", "um", "ưm",
    "iêm", "yêm", "uôm", "ươm", "oam",
]
_CODA_N = [
    "an", "ăn", "ân", "en", "ên", "in", "on", "ôn", "ơn", "un", "ưn",
    "iên", "yên", "uôn", "ươn", "oan", "oăn", "uân", "uôn", "uyên", "oen",
]
_CODA_NG = [
    "ang", "ăng", "âng", "eng", "ong", "ông", "ung", "ưng",
    "iêng", "uông", "ương", "oang", "oăng", "uâng",
]
_CODA_NH = ["anh", "ênh", "inh", "oanh", "uênh", "uynh"]
_CODA_P = [
    "ap", "ăp", "âp", "ep", "êp", "ip", "op", "ôp", "ơp", "up", "ưp",
    "iêp", "yêp", "ươp", "oap",
]
_CODA_T = [
    "at", "ăt", "ât", "et", "êt", "it", "ot", "ôt", "ơt", "ut", "ưt",
    "iêt", "yêt", "uôt", "ươt", "oat", "oăt", "uât", "uyêt", "oet",
]
_CODA_C = [
    "ac", "ăc", "âc", "ec", "oc", "ôc", "uc", "ưc",
    "iêc", "uôc", "ươc", "oac", "oăc",
]
_CODA_CH = ["ach", "êch", "ich", "oach", "uêch", "uych"]

# Rimes ending in a stop (-p -t -c -ch) are "checked" and take only sắc / nặng.
_CHECKED = set(_CODA_P + _CODA_T + _CODA_C + _CODA_CH)
_CHECKED_TONES = (Tone.NONE, Tone.ACUTE, Tone.DOT)

RIMES: List[str] = (
    _OPEN + _CODA_M + _CODA_N + _CODA_NG + _CODA_NH
    + _CODA_P + _CODA_T + _CODA_C + _CODA_CH
)

TONES = list(Tone)

# Front vowels after which c→k, g→gh, ng→ngh (orthographic spelling rule).
_FRONT = ("e", "ê", "i", "y", "iê", "yê")


def _spelling_ok(onset: str, rime: str) -> bool:
    """Reject onset+rime pairs forbidden by Vietnamese spelling rules."""
    starts_front = rime.startswith(_FRONT)
    if onset == "c" and starts_front:
        return False   # must be written "k"
    if onset == "k" and not starts_front:
        return False   # "k" only before front vowels
    if onset == "g" and starts_front:
        return False   # must be "gh"
    if onset == "gh" and not starts_front:
        return False
    if onset == "ng" and starts_front:
        return False   # must be "ngh"
    if onset == "ngh" and not starts_front:
        return False
    if onset == "q":  # only ever written "qu"
        return False
    # "qu" already contains the glide u; avoid doubling into u-initial rimes.
    if onset == "qu" and rime[0] in "uư":
        return False
    # "gi" already contains i; avoid gi + i-initial rime.
    if onset == "gi" and rime[0] == "i":
        return False
    return True


def _tones_for(rime: str) -> Iterable[Tone]:
    return _CHECKED_TONES if rime in _CHECKED else TONES


def generate_syllables() -> List[str]:
    """Return the sorted, de-duplicated, NFC syllable inventory."""
    out: Set[str] = set()
    for onset in ONSETS:
        for rime in RIMES:
            if not _spelling_ok(onset, rime):
                continue
            for tone in _tones_for(rime):
                toned = place_tone(rime, tone)
                out.add(nfc(onset + toned))
    return sorted(out)
