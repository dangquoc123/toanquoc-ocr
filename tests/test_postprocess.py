"""Tests for the §5 non-LLM post-processing chain (pure stdlib)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.charset import Tone  # noqa: E402
from vnocr.postprocess import (  # noqa: E402
    ConfusionModel,
    CountLanguageModel,
    NoisyChannel,
    PostProcessor,
    SyllableTrie,
    ToneRecovery,
    ctc_prefix_beam_search,
    tone_variants,
)


def _trie(words):
    return SyllableTrie(words)


# --- trie -----------------------------------------------------------------
def test_trie_contains_and_prefix():
    t = _trie(["tiếng", "việt", "chào"])
    assert t.contains("tiếng") and "việt" in t
    assert t.is_prefix("ti") and t.is_prefix("việ")
    assert not t.is_prefix("xz")
    assert not t.contains("tieng")  # missing tone → not a stored syllable


def test_trie_edit_candidates():
    t = _trie(["việt", "viết", "biết", "chào"])
    cands = t.candidates_within("vlệt", max_edits=1)  # l↔i frame error
    assert "việt" in cands


# --- noisy channel (§5.2) -------------------------------------------------
def test_noisy_channel_fixes_invalid_syllable():
    t = _trie(["việt", "nam", "tiếng"])
    nc = NoisyChannel(t, ConfusionModel(), max_edits=2)
    fixed, changed = nc.correct_token("vlệt")
    assert fixed == "việt" and changed


def test_noisy_channel_passes_valid_syllable():
    t = _trie(["việt", "nam"])
    nc = NoisyChannel(t)
    out, changed = nc.correct_token("việt")
    assert out == "việt" and not changed


def test_noisy_channel_preserves_case():
    t = _trie(["việt", "nam"])
    nc = NoisyChannel(t, max_edits=2)
    out, _ = nc.correct_token("Vlệt")
    assert out == "Việt"


# --- count LM + tone recovery (§5.3) --------------------------------------
def test_count_lm_backoff_scores():
    lm = CountLanguageModel(order=3).train(["tôi là sinh viên"] * 5)
    # seen bigram beats an unseen one
    assert lm.logprob("là", ["tôi"]) > lm.logprob("lã", ["tôi"])


def test_tone_variants_enumerated():
    t = _trie(["ma", "má", "mà", "mã", "mạ", "mả"])
    vs = tone_variants("mà", t)
    tones = {tone for _, tone in vs}
    assert Tone.GRAVE in tones and Tone.TILDE in tones
    assert len(vs) == 6


def test_context_flips_tone():
    # §5.3 in miniature: context makes "la" → "là" even with no visual evidence.
    corpus = ["tôi là sinh viên"] * 30 + ["cái bàn màu xanh"] * 10
    lm = CountLanguageModel(order=3).train(corpus)
    trie = _trie(["tôi", "la", "lá", "là", "lả", "lã", "lạ", "sinh", "viên"])
    tr = ToneRecovery(lm, trie, lm_weight=1.0, visual_confidence=0.6)
    out = tr.rescore(["tôi", "la", "sinh", "viên"])
    assert out[1] == "là"


# --- CTC prefix beam search (§5.1) ---------------------------------------
def _peak(idx, n, hi=0.9):
    lo = (1.0 - hi) / (n - 1)
    return [math.log(hi if i == idx else lo) for i in range(n)]


def test_ctc_beam_plain_decode():
    # alphabet: 0=blank, 1='a', 2='b'; frames spell a, blank, b -> "ab"
    alphabet = ["-", "a", "b"]
    lattice = [_peak(1, 3), _peak(0, 3), _peak(2, 3)]
    assert ctc_prefix_beam_search(lattice, alphabet, blank=0) == "ab"


def test_ctc_beam_collapses_repeats():
    alphabet = ["-", "a"]
    # a, a, a with no blank collapses to single "a"
    lattice = [_peak(1, 2), _peak(1, 2), _peak(1, 2)]
    assert ctc_prefix_beam_search(lattice, alphabet, blank=0) == "a"


def test_ctc_beam_double_letter_via_blank():
    alphabet = ["-", "a"]
    # a, blank, a -> "aa"
    lattice = [_peak(1, 2), _peak(0, 2), _peak(1, 2)]
    assert ctc_prefix_beam_search(lattice, alphabet, blank=0) == "aa"


# --- full chain -----------------------------------------------------------
def test_postprocessor_chain():
    trie = _trie(["việt", "nam", "tôi", "là", "la"])
    nc = NoisyChannel(trie, max_edits=2)
    pp = PostProcessor(trie=trie, noisy=nc, tone=None)
    out = pp.run("Vlệt nam")
    assert out.split()[0] == "Việt"


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
