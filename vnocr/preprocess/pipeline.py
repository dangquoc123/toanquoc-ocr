"""The classical preprocessing chain (Design §1.2–1.5).

Order (all grayscale, no binarisation on this path — §1.1):

    DPI normalise → deskew → (dewarp, optional) → CLAHE → bilateral denoise

Dewarp (§1.3) is only needed for phone photos / open books; for flatbed scans it
is skipped.  A DocUNet-lite model can be dropped into :func:`dewarp` later; the
stub returns the input unchanged so the pipeline runs today.

Everything here is OpenCV (CPU, a few ms/image) and parameterised by
:class:`PreprocessConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import cv2
    import numpy as np
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.preprocess needs numpy + opencv-python.") from _e

from .deskew import deskew
from .enhance import bilateral_denoise, clahe, normalize_dpi

__all__ = ["PreprocessConfig", "preprocess", "dewarp", "to_grayscale"]


@dataclass
class PreprocessConfig:
    do_dpi: bool = True
    src_dpi: Optional[float] = None
    do_deskew: bool = True
    skew_limit: float = 15.0
    do_dewarp: bool = False
    do_clahe: bool = True
    clahe_clip: float = 2.0
    do_denoise: bool = True


def to_grayscale(image: "np.ndarray") -> "np.ndarray":
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def dewarp(gray: "np.ndarray") -> "np.ndarray":
    """Page dewarping (§1.3).  Stub — plug in a DocUNet-lite grid here.

    Flatbed scans don't need it; the identity keeps the pipeline runnable.
    """
    return gray


def preprocess(image: "np.ndarray",
               config: Optional[PreprocessConfig] = None) -> "np.ndarray":
    """Run the full grayscale conditioning chain on a page image."""
    cfg = config or PreprocessConfig()
    gray = to_grayscale(image)
    if cfg.do_dpi:
        gray = normalize_dpi(gray, src_dpi=cfg.src_dpi)
    if cfg.do_deskew:
        gray, _ = deskew(gray)
        # re-limit exposed by cfg.skew_limit is applied inside estimate_skew
    if cfg.do_dewarp:
        gray = dewarp(gray)
    if cfg.do_clahe:
        gray = clahe(gray, clip=cfg.clahe_clip)
    if cfg.do_denoise:
        gray = bilateral_denoise(gray)
    return gray
