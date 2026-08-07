"""Contrast, denoise and DPI conditioning (Design §1.3–1.5, §1.1).

All operations keep the image **grayscale** — never binarised — because tone
marks are high-frequency strokes with soft anti-aliased edges, and the deep
recogniser uses that grey ramp to tell ``hỏi`` from ``ngã`` (§1.1).

* :func:`clahe` — local contrast equalisation (not global; global blows up
  noise), §1.4.
* :func:`bilateral_denoise` — edge-preserving denoise (bilateral, not Gaussian,
  so diacritic edges survive), §1.5.
* :func:`normalize_dpi` — up-sample sub-300-DPI input, cap at ~400 (§1.1); above
  600 DPI adds cost with no benefit.
"""

from __future__ import annotations

try:
    import cv2
    import numpy as np
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.preprocess needs numpy + opencv-python.") from _e

__all__ = ["clahe", "bilateral_denoise", "normalize_dpi", "unsharp"]


def clahe(gray: "np.ndarray", clip: float = 2.0,
          tiles: int = 8) -> "np.ndarray":
    """Contrast-Limited Adaptive Histogram Equalisation on a grayscale image."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return op.apply(gray)


def bilateral_denoise(gray: "np.ndarray", d: int = 5,
                      sigma_color: float = 30.0,
                      sigma_space: float = 5.0) -> "np.ndarray":
    """Edge-preserving denoise — keeps the sharp edges of tone marks."""
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def unsharp(gray: "np.ndarray", amount: float = 1.0,
            radius: float = 1.0) -> "np.ndarray":
    """Mild unsharp mask, used after up-sampling low-DPI input."""
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    return sharp


def normalize_dpi(gray: "np.ndarray", src_dpi: float = None,
                  target_min: int = 300, target: int = 400,
                  min_text_height: int = None) -> "np.ndarray":
    """Scale toward ~300–400 DPI.

    If ``src_dpi`` is known and below ``target_min``, up-scale (bicubic +
    unsharp).  Above ~600 DPI, down-scale to ``target`` to save compute (§1.1).
    Real super-resolution (ESRGAN-lite) can slot in here for severe cases.
    """
    if src_dpi is None:
        return gray
    if src_dpi < target_min:
        scale = target / src_dpi
        h, w = gray.shape[:2]
        up = cv2.resize(gray, (round(w * scale), round(h * scale)),
                        interpolation=cv2.INTER_CUBIC)
        return unsharp(up, amount=0.6, radius=1.0)
    if src_dpi > 600:
        scale = target / src_dpi
        h, w = gray.shape[:2]
        return cv2.resize(gray, (round(w * scale), round(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return gray
