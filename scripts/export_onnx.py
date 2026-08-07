#!/usr/bin/env python3
"""Export the recogniser to ONNX, with QAT-INT8 guidance (Design §7.2–7.3).

    python scripts/export_onnx.py --ckpt checkpoints/recognizer.pt \
        --out models/recognizer.onnx

Why QAT, not PTQ (§7.3): the tone/diacritic decision boundaries are tiny
(``ả``/``ã`` differ by a few pixels), so the ~S/2 quantisation error is the same
order as the logit margin and would flip argmaxes.  QAT inserts fake-quant nodes
during training so the network learns margins wide enough to survive rounding.
This script exports FP32 ONNX and prints the QAT wiring; run QAT in
``train_recognizer.py`` (insert ``torch.ao.quantization`` fake-quant) before
exporting for production.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="models/recognizer.onnx")
    ap.add_argument("--width", type=int, default=320, help="dummy input width")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    from vnocr.recognize import IMG_HEIGHT, LabelCodec, VietRecognizer

    ckpt = torch.load(args.ckpt, map_location="cpu")
    codec = LabelCodec()
    model = VietRecognizer(codec.charset)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 1, IMG_HEIGHT, args.width)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["log_probs"],
        dynamic_axes={"image": {0: "batch", 3: "width"},
                      "log_probs": {0: "batch", 1: "time"}},
        opset_version=args.opset,
    )
    print(f"exported FP32 ONNX -> {args.out}")
    print("\nNext (production, §7.3):")
    print(" 1. QAT: insert torch.ao.quantization fake-quant, fine-tune a few epochs,")
    print("    verify TONE accuracy separately (p_T) — that's where INT8 hurts.")
    print(" 2. Convert to INT8 ONNX / OpenVINO; serve with ONNX Runtime.")
    print(" 3. Re-run scripts/evaluate.py to confirm p_T survived quantisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
