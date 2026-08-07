"""Binarisation — for table-line detection ONLY (Design §1.1, §4.1).

Hard binarisation (Otsu / Sauvola) is the *wrong* input for the SVTR recogniser
because it erases the grey ramp at diacritic edges.  It is however exactly what
the morphological table-line extractor needs (§4.1): a clean 0/1 mask.  So this
lives in its own module and is used only by :mod:`vnocr.table.morphology` — never
on the path to the recogniser.
"""

from __future__ import annotations

try:
    import cv2
    import numpy as np
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.preprocess needs numpy + opencv-python.") from _e

__all__ = ["otsu", "sauvola"]


def otsu(gray: "np.ndarray") -> "np.ndarray":
    """Global Otsu threshold → uint8 mask in {0, 255} (ink = 255)."""
    return cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def sauvola(gray: "np.ndarray", window: int = 25, k: float = 0.2,
            r: float = 128.0) -> "np.ndarray":
    """Sauvola local thresholding — better than Otsu on uneven illumination.

    ``T(x,y) = m(x,y) · (1 + k·(s(x,y)/r − 1))`` over a sliding window.
    """
    g = gray.astype("float32")
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(window, window),
                         normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(g * g, ddepth=-1, ksize=(window, window),
                            normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = cv2.sqrt(cv2.max(mean_sq - mean * mean, 0))
    thresh = mean * (1 + k * (std / r - 1))
    return ((g < thresh).astype("uint8")) * 255  # ink = 255
