"""Non-LLM post-processing (Design §5): trie, noisy channel, n-gram, decoding.

This subpackage is what replaces the LLM.  Everything here is statistical /
algorithmic and runs on CPU in sub-millisecond time.  Pure stdlib except the
optional KenLM binding in :mod:`vnocr.postprocess.lm`.
"""

from .build import load_postprocessor, load_syllable_trie
from .decode import PostProcessor, ctc_prefix_beam_search
from .lm import (
    CountLanguageModel,
    KenLMModel,
    LanguageModel,
    ToneRecovery,
    tone_variants,
)
from .noisy_channel import ConfusionModel, NoisyChannel
from .trie import SyllableTrie

__all__ = [
    "SyllableTrie",
    "ConfusionModel",
    "NoisyChannel",
    "LanguageModel",
    "CountLanguageModel",
    "KenLMModel",
    "ToneRecovery",
    "tone_variants",
    "PostProcessor",
    "ctc_prefix_beam_search",
    "load_postprocessor",
    "load_syllable_trie",
]
