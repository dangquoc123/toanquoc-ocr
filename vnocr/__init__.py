"""vnocr — a light, LLM-free Vietnamese OCR pipeline.

Wins in a narrow domain (printed Vietnamese + administrative tables) through
specialisation, not scale.  Every component runs on CPU; the neural nets total
<30 MB (FP32) and there is no LLM anywhere in the inference path.

The subpackages mirror the design document:

* :mod:`vnocr.charset`     factorised base/modifier/tone character model (§3.2)
* :mod:`vnocr.preprocess`  classical OpenCV image conditioning (§1)
* :mod:`vnocr.layout`      page layout analysis (§2)
* :mod:`vnocr.detect`      DBNet text detection (§3)
* :mod:`vnocr.recognize`   SVTR recogniser with the factorised head (§3)
* :mod:`vnocr.table`       morphology + SLANet table reconstruction (§4)
* :mod:`vnocr.postprocess` trie + noisy-channel + KenLM n-gram, no LLM (§5)
* :mod:`vnocr.eval`        CER/WER plus separate p_B / p_T metrics (§8)

Only :mod:`vnocr.charset`, :mod:`vnocr.postprocess` (core) and
:mod:`vnocr.eval` import at top level without heavy dependencies; the neural
blocks import torch lazily so the pure-Python core stays usable on its own.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Keep the top-level import light: only the dependency-free linguistic core is
# eagerly exposed.  Neural blocks are imported on demand by their subpackages.
from . import charset  # noqa: F401

__all__ = ["charset", "__version__"]
