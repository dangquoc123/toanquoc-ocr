"""Constrained CTC decoding + the full §5 post-processing chain.

Two entry points:

* :func:`ctc_prefix_beam_search` — decode a CTC frame-log-prob lattice with an
  optional syllable-trie constraint and n-gram LM (Design §5.1).  Constraining
  *on the lattice* (rather than fixing a 1-best afterwards) keeps the model's
  full posterior in play — the §5.1 point.  Pure Python; takes a ``[T][C]`` list
  of log-probs so it runs without numpy (a numpy array indexes identically).

* :class:`PostProcessor` — the token-level chain for a 1-best string when you
  don't have the lattice: trie validity → noisy-channel frame-error correction
  (§5.2) → n-gram tone recovery (§5.3).

Neither path contains an LLM.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .lm import LanguageModel, ToneRecovery
from .noisy_channel import NoisyChannel
from .trie import SyllableTrie

__all__ = ["ctc_prefix_beam_search", "PostProcessor"]

_NINF = float("-inf")
_ALPHA_TAIL = re.compile(r"[^\W\d_]+$", re.UNICODE)
_ALPHA_HEAD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _logsumexp(a: float, b: float) -> float:
    if a == _NINF:
        return b
    if b == _NINF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _partial_word(prefix: str) -> str:
    m = _ALPHA_TAIL.search(prefix)
    return m.group(0).lower() if m else ""


def ctc_prefix_beam_search(
    log_probs: Sequence[Sequence[float]],
    alphabet: Sequence[str],
    blank: int = 0,
    beam_size: int = 16,
    trie: Optional[SyllableTrie] = None,
    lm: Optional[LanguageModel] = None,
    lm_weight: float = 0.3,
) -> str:
    """Prefix beam search over a CTC lattice (Hannun et al.), with constraints.

    Parameters
    ----------
    log_probs : ``[T][C]``
        Per-frame **natural-log** probabilities; ``alphabet[i]`` is the symbol
        of class ``i`` and ``alphabet[blank]`` is unused (blank).
    trie : optional
        If given, a prefix whose in-progress word is not a valid syllable prefix
        is pruned — the §5.1 lattice constraint.
    lm : optional
        If given, ``lm_weight · logprob(word | context)`` is added when a word
        completes (at a non-alphabetic symbol).
    """
    # beam: prefix(str) -> [p_blank, p_nonblank]
    beam: Dict[str, List[float]] = {"": [0.0, _NINF]}

    for t in range(len(log_probs)):
        row = log_probs[t]
        # only the most probable symbols per frame matter; keep it simple/complete
        nxt: Dict[str, List[float]] = defaultdict(lambda: [_NINF, _NINF])

        for prefix, (pb, pnb) in beam.items():
            ptot = _logsumexp(pb, pnb)
            last = prefix[-1] if prefix else ""

            # 1) emit blank → prefix unchanged, lands in p_blank
            cur = nxt[prefix]
            cur[0] = _logsumexp(cur[0], ptot + row[blank])

            for c, ch in enumerate(alphabet):
                if c == blank:
                    continue
                lp = row[c]

                if ch == last:
                    # repeat without blank: stays same prefix, extends p_nonblank
                    same = nxt[prefix]
                    same[1] = _logsumexp(same[1], pnb + lp)
                    # repeat after blank: genuine double letter → new prefix
                    src = pb + lp
                else:
                    src = ptot + lp

                new_prefix = prefix + ch

                # trie constraint on the in-progress word
                if trie is not None and ch.isalpha():
                    if not trie.is_prefix(_partial_word(new_prefix)):
                        continue

                # LM bonus when a word just completed (this symbol is a boundary)
                if lm is not None and not ch.isalpha():
                    word = _partial_word(prefix)
                    if word:
                        ctx = _ALPHA_HEAD.findall(prefix.lower())[-2:]
                        src = src + lm_weight * lm.logprob(word, ctx)

                ext = nxt[new_prefix]
                ext[1] = _logsumexp(ext[1], src)

        # prune to beam_size by total mass
        beam = dict(sorted(
            nxt.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]), reverse=True
        )[:beam_size])

    best = max(beam.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))
    return best[0]


@dataclass
class PostProcessor:
    """The token-level §5 chain for a 1-best string.

    Order matters: fix structurally-broken syllables first (noisy channel), then
    let context correct the tones of the now-valid syllables (tone recovery).
    """

    trie: SyllableTrie
    noisy: Optional[NoisyChannel] = None
    tone: Optional[ToneRecovery] = None

    def run(self, text: str) -> str:
        tokens = _tokenize_keep_delims(text)
        # 1) noisy-channel frame-error correction on each word token
        if self.noisy is not None:
            tokens = [
                self.noisy.correct_token(tok)[0] if _is_word(tok) else tok
                for tok in tokens
            ]
        # 2) n-gram tone recovery over the word sequence
        if self.tone is not None:
            words_idx = [i for i, tok in enumerate(tokens) if _is_word(tok)]
            words = [tokens[i] for i in words_idx]
            rescored = self.tone.rescore(words)
            for i, w in zip(words_idx, rescored):
                tokens[i] = w
        return "".join(tokens)


def _is_word(tok: str) -> bool:
    return bool(tok) and tok[0].isalpha()


def _tokenize_keep_delims(text: str) -> List[str]:
    """Split into alternating word / non-word chunks, preserving everything."""
    out: List[str] = []
    buf = []
    is_alpha = None
    for ch in text:
        a = ch.isalpha()
        if is_alpha is None or a == is_alpha:
            buf.append(ch)
        else:
            out.append("".join(buf))
            buf = [ch]
        is_alpha = a
    if buf:
        out.append("".join(buf))
    return out
