"""Evaluation: CER/WER, the p_B/p_T split (§8) and the tone-entropy meter (§0)."""

from .entropy import EntropyReport, measure_entropy, syllable_skeleton, syllable_tone
from .metrics import Metrics, align, cer, evaluate, wer

__all__ = [
    "Metrics",
    "evaluate",
    "cer",
    "wer",
    "align",
    "EntropyReport",
    "measure_entropy",
    "syllable_tone",
    "syllable_skeleton",
]
