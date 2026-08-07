"""Classical image preprocessing (Design §1).  Requires numpy + opencv-python.

Grayscale throughout — binarisation (``binarize``) is exported but used only by
the table module, never on the recogniser path (§1.1).
"""

from .binarize import otsu, sauvola
from .deskew import deskew, estimate_skew
from .enhance import bilateral_denoise, clahe, normalize_dpi, unsharp
from .pipeline import PreprocessConfig, dewarp, preprocess, to_grayscale

__all__ = [
    "preprocess",
    "PreprocessConfig",
    "to_grayscale",
    "dewarp",
    "deskew",
    "estimate_skew",
    "clahe",
    "bilateral_denoise",
    "normalize_dpi",
    "unsharp",
    "otsu",
    "sauvola",
]
