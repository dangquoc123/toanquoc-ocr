"""Recognition block (Design §3): SVTR backbone + factorised head + CTC/GTC.

Imports require PyTorch.  Importing this subpackage raises a clear error if
torch is missing; the pure-Python core (``vnocr.charset`` / ``vnocr.postprocess``
/ ``vnocr.eval``) is unaffected and remains usable on its own.
"""

from .ctc import ctc_loss, flat_log_probs_to_lattice, greedy_decode
from .dataset import IMG_HEIGHT, LabelCodec, LineRecognitionDataset, collate_fn
from .gtc import GTCConfig, NRTRHead, kd_loss
from .head import FactorisedHead, build_flat_index
from .model import VietRecognizer
from .svtr import SVTRBackbone, svtr_tiny

__all__ = [
    "VietRecognizer",
    "SVTRBackbone",
    "svtr_tiny",
    "FactorisedHead",
    "build_flat_index",
    "ctc_loss",
    "greedy_decode",
    "flat_log_probs_to_lattice",
    "NRTRHead",
    "kd_loss",
    "GTCConfig",
    "LabelCodec",
    "LineRecognitionDataset",
    "collate_fn",
    "IMG_HEIGHT",
]
