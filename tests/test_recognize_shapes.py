"""Shape / correctness tests for the neural recogniser (Design §3).

These need PyTorch. Without it they **skip** cleanly (the pure-Python core
suite is unaffected). After `pip install -e '.[train]'`:

    pytest tests/test_recognize_shapes.py      # or:  python tests/test_recognize_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _skip_if_no_torch():
    if not HAVE_TORCH:
        try:
            import pytest
            pytest.skip("torch not installed", allow_module_level=False)
        except ImportError:
            raise SystemExit(0)


def test_forward_shape_and_stride():
    _skip_if_no_torch()
    import torch
    from vnocr.recognize import IMG_HEIGHT, VietRecognizer, LabelCodec

    codec = LabelCodec()
    model = VietRecognizer(codec.charset).eval()
    B, W = 2, 256
    x = torch.rand(B, 1, IMG_HEIGHT, W)
    out = model(x)
    # width is downsampled to W/4 -> CTC sequence length (§3.3)
    assert out["log_probs"].shape == (B, W // 4, model.num_classes)
    # factor logits exposed for aux/ablation
    assert out["base"].shape[-1] == codec.charset.n_base
    assert out["tone"].shape[-1] == codec.charset.n_tone


def test_param_budget():
    _skip_if_no_torch()
    from vnocr.recognize import VietRecognizer
    model = VietRecognizer()
    # §2 budget: recogniser is 8-12M params; assert it stays under 15M
    assert model.num_parameters() < 15e6


def test_mask_zeroes_impossible_glyphs():
    _skip_if_no_torch()
    import torch
    from vnocr.recognize import VietRecognizer, LabelCodec
    from vnocr.charset import Modifier, Tone

    codec = LabelCodec()
    model = VietRecognizer(codec.charset).eval()
    # the log-mask buffer must be -inf exactly where is_valid is False
    lm = model.head.log_mask
    bi = codec.charset.bases.index("o")
    assert lm[bi, int(Modifier.BREVE), int(Tone.NONE)] < -1e8   # ŏ impossible
    bi_u = codec.charset.bases.index("u")
    assert lm[bi_u, int(Modifier.HORN), int(Tone.TILDE)] == 0.0  # ữ valid


def test_ctc_loss_and_decode_run():
    _skip_if_no_torch()
    import torch
    from vnocr.recognize import VietRecognizer, LabelCodec, ctc_loss

    codec = LabelCodec()
    model = VietRecognizer(codec.charset).eval()
    B, W, L = 2, 256, 6
    x = torch.rand(B, 1, 48, W)
    out = model(x)
    lp = out["log_probs"]
    K = model.num_classes - 1
    targets = torch.randint(1, K + 1, (B * L,))
    in_len = torch.full((B,), lp.size(1), dtype=torch.long)
    tgt_len = torch.full((B,), L, dtype=torch.long)
    loss = ctc_loss(lp, targets, in_len, tgt_len)
    assert torch.isfinite(loss)
    preds = model.decode(lp)
    assert len(preds) == B and all(isinstance(p, str) for p in preds)


def test_greedy_decode_maps_classes_to_glyphs():
    _skip_if_no_torch()
    import torch
    from vnocr.recognize import VietRecognizer, LabelCodec
    from vnocr.recognize.ctc import greedy_decode

    codec = LabelCodec()
    model = VietRecognizer(codec.charset)
    # craft a fake lattice that argmaxes to classes [1, blank, 2] -> chars[0],chars[1]
    C = model.num_classes
    T = 3
    logits = torch.full((1, T, C), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 0] = 10.0     # blank
    logits[0, 2, 2] = 10.0
    lp = torch.log_softmax(logits, dim=-1)
    out = greedy_decode(lp, model.flat_chars, blank=0)[0]
    assert out == model.flat_chars[0] + model.flat_chars[1]


if __name__ == "__main__":
    if not HAVE_TORCH:
        print("SKIPPED — torch not installed (pip install -e '.[train]')")
        raise SystemExit(0)
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
    raise SystemExit(1 if failed else 0)
