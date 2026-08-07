"""Tests for CER/WER, the p_B/p_T split (§8) and tone entropy (§0)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.eval import cer, evaluate, measure_entropy, wer  # noqa: E402


def test_perfect_match_is_zero():
    m = evaluate(["tôi yêu tiếng việt"], ["tôi yêu tiếng việt"])
    assert m.cer == 0.0 and m.wer == 0.0
    assert m.base_error_rate == 0.0 and m.tone_error_rate == 0.0
    assert m.p_B == 1.0 and m.p_T == 1.0


def test_pure_tone_error_isolated():
    # tôi -> tối : same base+modifier, only the tone changes (NONE -> ACUTE).
    m = evaluate(["tôi yêu tiếng việt"], ["tối yêu tiếng việt"])
    assert m.base_error_rate == 0.0        # every base letter still correct
    assert m.p_B == 1.0
    assert 0.0 < m.tone_error_rate < 0.2   # exactly one tone-vowel wrong of nine
    assert abs(m.tone_error_rate - 1 / 9) < 1e-9


def test_base_error_counted():
    # nam -> nan : final consonant base error, no tone involved
    m = evaluate(["nam"], ["nan"])
    assert m.base_error_rate > 0.0
    assert m.p_B < 1.0


def test_cer_wer_helpers():
    assert cer("việt nam", "việt nam") == 0.0
    assert wer("việt nam", "viet nam") > 0.0     # one word wrong
    assert 0.0 < cer("việt", "viet") < 1.0


def test_normalization_folds_tone_placement():
    # phonetic vs dictionary tone placement must not count as an error
    assert cer("hòa bình", "hoà bình") == 0.0


def test_entropy_report_shape_and_reduction():
    corpus = [
        "nước cộng hòa xã hội chủ nghĩa việt nam",
        "tôi là sinh viên đại học",
        "hà nội là thủ đô của việt nam",
    ] * 20
    r = measure_entropy(corpus)
    assert r.n_syllables > 0
    assert r.H_t >= 0.0
    # conditioning on skeleton + previous word should not *raise* uncertainty
    assert r.H_t_given_s_wprev <= r.H_t + 1e-6
    assert isinstance(r.bigram_suffices(), bool)
    assert "H(t | s, w-1)" in r.summary()


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
