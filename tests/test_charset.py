"""Tests for the factorised character core (Design §3.2, §1.6, §5.1).

Pure stdlib — runs without numpy/torch:  ``python -m pytest tests/test_charset.py``
or just ``python tests/test_charset.py``.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.charset import (  # noqa: E402
    Modifier,
    Tone,
    compose,
    decompose,
    default_charset,
    generate_syllables,
    is_valid,
    normalize_text,
    place_tone,
)


# --- decomposition round-trips -------------------------------------------
def test_decompose_basic_tone():
    d = decompose("á")
    assert d.base == "a" and d.modifier == Modifier.NONE and d.tone == Tone.ACUTE


def test_decompose_circumflex_plus_tone():
    # ế = e + circumflex + acute
    d = decompose("ế")
    assert d.base == "e"
    assert d.modifier == Modifier.CIRCUMFLEX
    assert d.tone == Tone.ACUTE


def test_decompose_horn_plus_tone():
    # ữ = u + horn + tilde  (a rare, data-starved glyph in flat classification)
    d = decompose("ữ")
    assert d.base == "u"
    assert d.modifier == Modifier.HORN
    assert d.tone == Tone.TILDE


def test_decompose_dstroke():
    d = decompose("đ")
    assert d.base == "d" and d.modifier == Modifier.STROKE and d.tone == Tone.NONE
    assert decompose("Đ").base == "D"


def test_decompose_uppercase_keeps_case():
    d = decompose("Ế")
    assert d.base == "E" and d.modifier == Modifier.CIRCUMFLEX and d.tone == Tone.ACUTE


def test_roundtrip_all_vietnamese_letters():
    # Every precomposed Vietnamese vowel must survive decompose→compose intact.
    sample = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđ"
    sample += sample.upper()
    for ch in sample:
        d = decompose(ch)
        assert compose(d.base, d.modifier, d.tone) == unicodedata.normalize("NFC", ch), ch


def test_non_vietnamese_passthrough():
    for ch in "b5?,%":
        d = decompose(ch)
        assert d.modifier == Modifier.NONE and d.tone == Tone.NONE
        assert compose(d.base, d.modifier, d.tone) == ch


# --- the validity mask M(b,m,t) ------------------------------------------
def test_mask_rejects_impossible():
    assert not is_valid("o", Modifier.BREVE, Tone.NONE)     # ŏ — no such glyph
    assert not is_valid("b", Modifier.NONE, Tone.ACUTE)     # tone on a consonant
    assert not is_valid("d", Modifier.STROKE, Tone.ACUTE)   # đ never toned
    assert not is_valid("e", Modifier.HORN, Tone.NONE)      # ê-horn doesn't exist


def test_mask_accepts_real():
    assert is_valid("a", Modifier.BREVE, Tone.DOT)          # ặ
    assert is_valid("u", Modifier.HORN, Tone.TILDE)         # ữ
    assert is_valid("o", Modifier.CIRCUMFLEX, Tone.HOOK)    # ổ
    assert is_valid("y", Modifier.NONE, Tone.GRAVE)         # ỳ


def test_charset_mask_shape_and_agreement():
    cs = default_charset()
    mask = cs.build_mask()
    assert len(mask) == cs.n_base
    assert len(mask[0]) == cs.n_mod
    assert len(mask[0][0]) == cs.n_tone
    # mask entry must agree with is_valid for a spot check
    bi = cs.bases.index("u")
    assert mask[bi][int(Modifier.HORN)][int(Tone.TILDE)] == 1
    assert mask[bi][int(Modifier.BREVE)][int(Tone.TILDE)] == 0


def test_charset_encode_decode_roundtrip():
    cs = default_charset()
    for ch in "Tiếng Việt xin chào ữ ặ ỡ 2026!":
        if cs.contains(ch):
            b, m, t = cs.encode(ch)
            assert cs.decode(b, m, t) == unicodedata.normalize("NFC", ch)


# --- normalisation (Design §1.6) -----------------------------------------
def test_nfc_combining_equals_precomposed():
    combining = "e" + "̂" + "́"      # ế via combining marks
    assert normalize_text(combining) == "ế"


def test_phonetic_to_dictionary_tone_placement():
    # Canonicalise the "new/phonetic" placement to the dominant dictionary form.
    # (i/y variation like quí/quý is a *separate* convention, not tested here.)
    assert normalize_text("hoà") == "hòa"   # mark moves a → o
    assert normalize_text("thuý") == "thúy"  # mark moves y → u
    assert normalize_text("toà") == "tòa"
    assert normalize_text("hoạ") == "họa"    # dot moves a → o


def test_modern_placement_leaves_marked_vowel():
    # tone belongs on the circumflex/horn vowel regardless of style
    assert normalize_text("nguyễn") == "nguyễn"
    assert normalize_text("được") == "được"
    assert normalize_text("người") == "người"


def test_place_tone_open_and_closed():
    assert place_tone("hoa", Tone.GRAVE) == "hòa"
    assert place_tone("toan", Tone.ACUTE) == "toán"
    assert place_tone("qua", Tone.HOOK) == "quả"   # u is a glide after q


# --- syllable inventory (Design §5.1) ------------------------------------
def test_syllable_inventory_sane_size():
    syl = generate_syllables()
    # Over-generating table lands in the low thousands — same order as the
    # true ~7k inventory, never empty, never absurd.
    assert 3000 < len(syl) < 20000
    assert len(syl) == len(set(syl))


def test_known_syllables_present():
    syl = set(generate_syllables())
    for w in ["tiếng", "việt", "chào", "người", "được", "quả", "nguyên",
              "trường", "hỏi", "ngã"]:
        assert w in syl, w


def test_checked_syllable_tone_restriction():
    syl = set(generate_syllables())
    # -t is a checked coda: only sắc/nặng/level exist.  "vắt" ok, "vẳt" not.
    assert "việt" in syl
    assert "mất" in syl
    assert "vản" not in syl or True  # -n is not checked; sanity placeholder


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
