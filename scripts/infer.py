#!/usr/bin/env python3
"""Run the full OCR pipeline on an image (Design §2).

    python scripts/infer.py page.jpg --ckpt checkpoints/recognizer.pt \
        --syllables data/charset/syllables.txt --lm data/lm/vi.count.pkl \
        --format text

Assembles preprocess → layout → detect/recognise → post-process → merge and
prints text / json / html.  Recognition needs a trained checkpoint (torch);
without ``--ckpt`` it runs the classical stages (preprocess, layout, table
structure) and reports what it found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("--ckpt", help="recogniser checkpoint (.pt)")
    ap.add_argument("--syllables", default="data/charset/syllables.txt")
    ap.add_argument("--lm", help="pickled CountLanguageModel")
    ap.add_argument("--kenlm", help="KenLM binary")
    ap.add_argument("--format", choices=["text", "json", "html"], default="text")
    ap.add_argument("--src-dpi", type=float)
    args = ap.parse_args()

    from vnocr.layout import LayoutAnalyzer
    from vnocr.preprocess import PreprocessConfig, preprocess
    from vnocr.postprocess import load_postprocessor
    from vnocr.pipeline import OCRPipeline
    from vnocr.utils import imread

    image = imread(args.image)

    cfg = PreprocessConfig(src_dpi=args.src_dpi)
    pre = (lambda im: preprocess(im, cfg))

    post = None
    if Path(args.syllables).exists():
        post = load_postprocessor(args.syllables, lm_path=args.lm,
                                  kenlm_path=args.kenlm)

    recognizer = None
    if args.ckpt:
        import torch
        from vnocr.recognize import LabelCodec, VietRecognizer
        codec = LabelCodec()
        ckpt = torch.load(args.ckpt, map_location="cpu")
        # honour the §3.2 ablation flag the checkpoint was trained with
        recognizer = VietRecognizer(
            codec.charset, use_interaction=ckpt.get("use_interaction", True))
        recognizer.load_state_dict(ckpt["model"])
        recognizer.eval()

    pipe = OCRPipeline(
        preprocessor=pre,
        layout=LayoutAnalyzer(),
        recognizer=recognizer,
        postprocessor=post,
    )
    result = pipe.run(image)

    if args.format == "text":
        print(result.text())
    elif args.format == "json":
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
    else:
        print(result.to_html())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
