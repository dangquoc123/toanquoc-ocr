"""Assemble a :class:`PostProcessor` from artifacts on disk (Design §5).

    pp = load_postprocessor("data/charset/syllables.txt",
                            lm_path="data/lm/vi.count.pkl")
    clean = pp.run("Vlệt nam")

Loads the syllable trie, an optional language model (a pickled
:class:`CountLanguageModel`, or a KenLM binary via ``kenlm_path``), and wires the
noisy-channel corrector and tone-recovery decoder together.  Pure stdlib unless
a KenLM binary is requested.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from .decode import PostProcessor
from .lm import CountLanguageModel, KenLMModel, LanguageModel, ToneRecovery
from .noisy_channel import ConfusionModel, NoisyChannel
from .trie import SyllableTrie

__all__ = ["load_syllable_trie", "load_postprocessor"]


def load_syllable_trie(path: str) -> SyllableTrie:
    """Load a newline-delimited syllable list into a trie."""
    words = Path(path).read_text(encoding="utf-8").split()
    return SyllableTrie(words)


def load_postprocessor(syllables_path: str,
                       lm_path: Optional[str] = None,
                       kenlm_path: Optional[str] = None,
                       confusion: Optional[ConfusionModel] = None,
                       max_edits: int = 2,
                       lm_weight: float = 1.0) -> PostProcessor:
    trie = load_syllable_trie(syllables_path)
    noisy = NoisyChannel(trie, confusion or ConfusionModel(), max_edits=max_edits)

    lm: Optional[LanguageModel] = None
    if kenlm_path:
        lm = KenLMModel(kenlm_path)
    elif lm_path:
        with open(lm_path, "rb") as f:
            lm = pickle.load(f)

    tone = ToneRecovery(lm, trie, lm_weight=lm_weight) if lm is not None else None
    return PostProcessor(trie=trie, noisy=noisy, tone=tone)
