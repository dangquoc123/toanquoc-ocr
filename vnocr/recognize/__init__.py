"""Recognition block (Design §3): SVTR backbone + factorised head + CTC/GTC.

Re-exports are **lazy**: the torch-free parts (:class:`LabelCodec`,
:class:`LineRecognitionDataset`, :func:`collate_fn`, ``IMG_HEIGHT``) import
without PyTorch, while touching any neural component (:class:`VietRecognizer`,
:class:`NRTRHead`, …) raises the usual clear ImportError if torch is missing.
The pure-Python core (``vnocr.charset`` / ``vnocr.postprocess`` / ``vnocr.eval``)
is unaffected either way.
"""

from importlib import import_module

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

_EXPORTS = {
    "VietRecognizer": "model",
    "SVTRBackbone": "svtr",
    "svtr_tiny": "svtr",
    "FactorisedHead": "head",
    "build_flat_index": "head",
    "ctc_loss": "ctc",
    "greedy_decode": "ctc",
    "flat_log_probs_to_lattice": "ctc",
    "NRTRHead": "gtc",
    "kd_loss": "gtc",
    "GTCConfig": "gtc",
    "LabelCodec": "dataset",
    "LineRecognitionDataset": "dataset",
    "collate_fn": "dataset",
    "IMG_HEIGHT": "dataset",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module = import_module(f".{_EXPORTS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
