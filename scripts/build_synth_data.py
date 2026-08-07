#!/usr/bin/env python3
"""Synthetic line-image generator (Design §6.2).

Renders corpus text to line crops with realistic degradations (JPEG, blur,
print-scan, low DPI/contrast) — the bulk of recogniser training data.

    python scripts/build_synth_data.py --corpus corpus.txt --fonts fonts/ \
        --out data/synth --n 100000

**Font warning (§6.2):** many Latin fonts render Vietnamese diacritics wrong.
``--verify-fonts`` renders the stress set (ữ ặ ỡ ẫ ...) and drops any font whose
glyphs are missing/tofu, so bad fonts never poison the labels.

Needs Pillow (+ numpy/opencv for degradations).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.charset.normalize import normalize_text

# Diacritic stress set — any font that can't render these is unfit (§6.2).
STRESS_GLYPHS = "ữ ặ ỡ ẫ ề ộ ở ẳ ẹ ứ đ Ữ Ặ Đ"


def verify_font(font_path: str, size: int = 40) -> bool:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return False
    for ch in STRESS_GLYPHS.split():
        try:
            mask = font.getmask(ch)
        except Exception:
            return False
        if mask is None or mask.getbbox() is None:
            return False  # tofu / missing glyph
    return True


def degrade(img):
    """Apply a random realistic degradation chain to a PIL image."""
    import numpy as np
    from PIL import Image, ImageFilter
    import io

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    if random.random() < 0.4:  # JPEG recompression
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=random.randint(30, 75))
        buf.seek(0)
        img = Image.open(buf).convert("L")
    if random.random() < 0.3:  # low DPI: downscale then upscale
        w, h = img.size
        s = random.uniform(0.5, 0.8)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s)))).resize((w, h))
    if random.random() < 0.4:  # additive noise
        arr = np.asarray(img).astype("float32")
        arr += np.random.normal(0, random.uniform(3, 12), arr.shape)
        img = Image.fromarray(arr.clip(0, 255).astype("uint8"))
    return img


def render_line(text: str, font, pad: int = 6):
    from PIL import Image, ImageDraw
    dummy = Image.new("L", (1, 1))
    box = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text, font=font, fill=0)
    return img


def find_fonts(spec: str):
    """Resolve ``--fonts`` (a dir or comma-separated files) to font paths."""
    paths = []
    for part in spec.split(","):
        p = Path(part.strip())
        if p.is_dir():
            for ext in ("*.ttf", "*.otf", "*.ttc", "*.TTF", "*.OTF", "*.TTC"):
                paths += [str(x) for x in p.glob(ext)]
        elif p.exists():
            paths.append(str(p))
    return sorted(set(paths))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fonts", required=True,
                    help="dir of fonts, or comma-separated font files "
                         "(.ttf/.otf/.ttc). macOS: /System/Library/Fonts")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--mode", choices=["words", "lines", "mixed"], default="mixed",
                    help="words=random word salad (augmentation); "
                         "lines=real corpus lines (realism); mixed=both")
    ap.add_argument("--min-words", type=int, default=1)
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--no-verify-fonts", dest="verify_fonts",
                    action="store_false", default=True)
    args = ap.parse_args()

    from PIL import ImageFont

    font_paths = find_fonts(args.fonts)
    if args.verify_fonts:
        good = [f for f in font_paths if verify_font(f)]
        print(f"fonts: {len(good)}/{len(font_paths)} pass the diacritic check")
        for f in font_paths:
            if f not in good:
                print(f"  dropped (bad diacritics): {f}")
        font_paths = good
    if not font_paths:
        print("error: no usable fonts (see docs/TRAINING.md for open VN fonts)",
              file=sys.stderr)
        return 2

    raw_lines = [l.strip() for l in
                 Path(args.corpus).read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    words = " ".join(raw_lines).split()
    lines = [l for l in raw_lines if 1 <= len(l.split()) <= args.max_words]
    if not words:
        print("error: empty corpus", file=sys.stderr)
        return 2

    def sample_text() -> str:
        mode = args.mode
        if mode == "mixed":
            mode = random.choice(["words", "lines"] if lines else ["words"])
        if mode == "lines" and lines:
            return random.choice(lines)
        k = random.randint(args.min_words, args.max_words)
        return " ".join(random.choice(words) for _ in range(k))

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = (out / "labels.txt").open("w", encoding="utf-8")

    for i in range(args.n):
        text = normalize_text(sample_text())
        font = ImageFont.truetype(random.choice(font_paths), random.randint(28, 52))
        img = degrade(render_line(text, font))
        rel = f"images/{i:07d}.png"
        img.save(out / rel)
        manifest.write(f"{rel}\t{text}\n")
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{args.n}")
    manifest.close()
    print(f"done -> {out}/labels.txt  ({args.n} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
