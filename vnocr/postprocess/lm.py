"""N-gram language model + tone recovery (Design §5.3) — the LLM replacement.

Tone is the dominant error class, and only *context* recovers it (a single
syllable's tone is lexically ambiguous — ma/má/mà/mã/mạ are all real).  §0 shows
the needed context is a bigram/trigram, not an LLM: once ``H(t | context)`` is
near the Fano floor, a MAP decoder is within ~1.35× of optimal and an LLM buys
almost nothing.

Two pieces:

* :class:`LanguageModel` — an n-gram scorer.  :class:`KenLMModel` wraps a KenLM
  binary (optional dependency); :class:`CountLanguageModel` is a pure-Python
  stupid-backoff model you can train by counting a plain-text corpus, so this
  module works with zero third-party deps.
* :class:`ToneRecovery` — the §5.3 decoder: for each token it weighs visual
  evidence ``ℓ(t)`` against the context prior ``π(t | w₋₁)`` and flips the tone
  when context outvotes a shaky visual call.

Training data is plain text (Wikipedia + news, §6.3): no GPU, no labels.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..charset.decompose import Modifier, Tone, compose, decompose
from ..charset.normalize import nfc, normalize_text
from .trie import SyllableTrie

__all__ = [
    "LanguageModel",
    "CountLanguageModel",
    "KenLMModel",
    "ToneRecovery",
    "tone_variants",
]

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_BOS, _EOS = "<s>", "</s>"
_LOG0 = -60.0  # a finite stand-in for log(0)


class LanguageModel:
    """Interface: natural-log probability of ``word`` given left ``context``."""

    def logprob(self, word: str, context: Sequence[str]) -> float:
        raise NotImplementedError

    def score(self, tokens: Sequence[str], order: int = 3) -> float:
        toks = [_BOS] + list(tokens) + [_EOS]
        total = 0.0
        for i in range(1, len(toks)):
            ctx = toks[max(0, i - order + 1):i]
            total += self.logprob(toks[i], ctx)
        return total


class CountLanguageModel(LanguageModel):
    """Stupid-backoff n-gram model, trained by counting (pure stdlib)."""

    def __init__(self, order: int = 3, backoff: float = 0.4) -> None:
        self.order = order
        self.backoff = backoff
        self._ngrams: List[Counter] = [Counter() for _ in range(order + 1)]
        self._ctx: List[Counter] = [Counter() for _ in range(order + 1)]
        self._vocab: set = set()
        self._total = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [w.lower() for w in _WORD_RE.findall(normalize_text(text))]

    def train(self, corpus: Iterable[str]) -> "CountLanguageModel":
        for line in corpus:
            toks = [_BOS] + self.tokenize(line) + [_EOS]
            self._vocab.update(toks)
            for tok in toks:
                self._total += 1
            for n in range(1, self.order + 1):
                for i in range(len(toks) - n + 1):
                    gram = tuple(toks[i:i + n])
                    self._ngrams[n][gram] += 1
                    self._ctx[n][gram[:-1]] += 1
        return self

    def logprob(self, word: str, context: Sequence[str]) -> float:
        word = word.lower()
        ctx = tuple(w.lower() for w in context)[-(self.order - 1):]
        # back off from the longest available context
        for n in range(len(ctx) + 1, 0, -1):
            gram = ctx[len(ctx) - (n - 1):] + (word,)
            c = self._ngrams[n][gram]
            if c > 0:
                denom = self._ctx[n][gram[:-1]]
                if denom > 0:
                    discount = (self.order - n)
                    return math.log(c / denom) + discount * math.log(self.backoff)
        # unigram fallback with add-one over vocab
        v = max(len(self._vocab), 1)
        return math.log(1.0 / (self._total + v))


class KenLMModel(LanguageModel):
    """Adapter over a trained KenLM model (``.arpa`` or ``.binary``).

    Requires the optional ``kenlm`` package.  KenLM returns log10 probabilities;
    we convert to natural log for a consistent interface.
    """

    def __init__(self, path: str) -> None:
        try:
            import kenlm  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "KenLMModel needs the 'kenlm' package. Install it, or use "
                "CountLanguageModel for a pure-Python fallback."
            ) from e
        self._model = kenlm.Model(path)
        self._ln10 = math.log(10.0)

    def logprob(self, word: str, context: Sequence[str]) -> float:  # pragma: no cover
        state_in = self._kenlm_state(context)
        import kenlm  # type: ignore
        out = kenlm.State()
        lp10 = self._model.BaseScore(state_in, word.lower(), out)
        return lp10 * self._ln10

    def _kenlm_state(self, context: Sequence[str]):  # pragma: no cover
        import kenlm  # type: ignore
        state = kenlm.State()
        self._model.BeginSentenceWrite(state)
        for w in context:
            nxt = kenlm.State()
            self._model.BaseScore(state, w.lower(), nxt)
            state = nxt
        return state


def tone_variants(word: str, trie: Optional[SyllableTrie] = None) -> List[Tuple[str, Tone]]:
    """All valid tone re-spellings of a syllable, as ``(spelling, tone)``.

    ``mà`` → ``[(ma, NONE), (má, ACUTE), (mà, GRAVE), (mả, HOOK), (mã, TILDE),
    (mạ, DOT)]`` filtered to those the ``trie`` accepts (if given).  This is the
    hypothesis set the tone decoder searches over.
    """
    word = nfc(word)
    decs = [decompose(c) for c in word]
    # locate the tone-bearing vowel (the one that currently carries, or would)
    from ..charset.normalize import place_tone
    base = "".join(compose(d.base, d.modifier, Tone.NONE) for d in decs)
    out: List[Tuple[str, Tone]] = []
    seen = set()
    for t in Tone:
        spelled = place_tone(base, t)
        low = spelled.lower()
        if low in seen:
            continue
        if trie is not None and not trie.contains(low):
            continue
        seen.add(low)
        out.append((spelled, t))
    return out


class ToneRecovery:
    """§5.3 tone decoder: ``t̂ = argmax_t ℓ(t) · π(t | context)``.

    Operates on a 1-best token sequence.  For each token it enumerates valid
    tone variants and picks the one maximising ``log ℓ(t) + log π(word_t | ctx)``.

    ``ℓ(t)`` — the recogniser's visual evidence — is supplied per position as a
    ``{Tone: prob}`` dict when the recogniser exposes it.  With no evidence we
    fall back to a peaked prior around the observed tone controlled by
    ``visual_confidence`` (so context can still flip a tone, but must overcome a
    handicap — never rewrites confident calls blindly).
    """

    def __init__(self, lm: LanguageModel, trie: SyllableTrie,
                 lm_weight: float = 1.0, visual_confidence: float = 0.6,
                 order: int = 3) -> None:
        self.lm = lm
        self.trie = trie
        self.lm_weight = lm_weight
        self.visual_confidence = visual_confidence
        self.order = order

    def _visual_logprob(self, tone: Tone, observed: Tone,
                        evidence: Optional[Dict[Tone, float]]) -> float:
        if evidence is not None:
            p = evidence.get(tone, 1e-6)
            return math.log(max(p, 1e-9))
        # no evidence: peaked around the observed tone
        n_tones = len(Tone)
        if tone == observed:
            p = self.visual_confidence
        else:
            p = (1.0 - self.visual_confidence) / (n_tones - 1)
        return math.log(p)

    def rescore(self, tokens: Sequence[str],
                evidence: Optional[Sequence[Optional[Dict[Tone, float]]]] = None
                ) -> List[str]:
        tokens = list(tokens)
        out: List[str] = []
        history: List[str] = [_BOS]
        for idx, tok in enumerate(tokens):
            low = tok.lower()
            if not low.isalpha():
                out.append(tok)
                if low.strip():
                    history.append(low)
                continue

            variants = tone_variants(tok, self.trie)
            if len(variants) <= 1:
                out.append(tok)
                history.append(low)
                continue

            observed = decompose_tone(tok)
            ev = evidence[idx] if evidence is not None else None
            ctx = tuple(history[-(self.order - 1):])

            best, best_score = tok, -math.inf
            for spelled, tone in variants:
                vis = self._visual_logprob(tone, observed, ev)
                lm = self.lm.logprob(spelled.lower(), ctx) * self.lm_weight
                s = vis + lm
                if s > best_score:
                    best, best_score = spelled, s
            best = _match_case(tok, best)
            out.append(best)
            history.append(best.lower())
        return out


def decompose_tone(word: str) -> Tone:
    for ch in word:
        t = decompose(ch).tone
        if t != Tone.NONE:
            return t
    return Tone.NONE


def _match_case(template: str, spelled: str) -> str:
    if template[:1].isupper():
        return spelled[:1].upper() + spelled[1:]
    return spelled
