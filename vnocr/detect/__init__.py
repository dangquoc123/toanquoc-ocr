"""Text detection (Design §3): DBNet.  Network needs torch; box extraction
needs numpy/opencv.  Both optional relative to the pure-Python core.
"""

from .dbnet import DBNet, boxes_from_prob

__all__ = ["DBNet", "boxes_from_prob"]
