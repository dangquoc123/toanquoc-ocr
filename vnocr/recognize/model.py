"""The Vietnamese recogniser: SVTR backbone + factorised head (Design §3).

Inference path is deliberately small: grayscale line image → SVTR encoder
(48px, asymmetric stride) → factorised CTC head → log-prob lattice.  The GTC
teacher (§3.1) attaches only for training and is absent here.

    rec = VietRecognizer(default_charset())
    out = rec(images)                    # {'log_probs': [B,T,C], 'base', ...}
    texts = rec.decode(out['log_probs']) # greedy 1-best strings

Feed ``out['log_probs']`` to :func:`vnocr.postprocess.decode.ctc_prefix_beam_search`
for dictionary-constrained decoding, or ``rec.decode`` for a quick greedy read.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.recognize needs PyTorch (`pip install torch`).") from _e

from ..charset.charset import Charset, default_charset
from .ctc import greedy_decode
from .head import FactorisedHead
from .svtr import SVTRBackbone, svtr_tiny

__all__ = ["VietRecognizer"]


class VietRecognizer(nn.Module):
    def __init__(self, charset: Optional[Charset] = None,
                 backbone: Optional[SVTRBackbone] = None,
                 in_ch: int = 1, use_interaction: bool = True) -> None:
        super().__init__()
        self.charset = charset or default_charset()
        self.backbone = backbone or svtr_tiny(in_ch=in_ch)
        self.head = FactorisedHead(
            self.backbone.out_dim, self.charset, use_interaction=use_interaction)
        self.num_classes = self.head.num_classes

    @property
    def flat_chars(self) -> List[str]:
        return self.head.flat_chars

    def forward(self, images: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """``images`` ``[B, C, 48, W]`` → head outputs (see :class:`FactorisedHead`)."""
        feats = self.backbone(images)      # [B, T, out_dim]
        return self.head(feats)

    def encode_features(self, images: "torch.Tensor") -> "torch.Tensor":
        """Shared encoder features — exposed for the GTC teacher during training."""
        return self.backbone(images)

    @torch.no_grad()
    def decode(self, log_probs: "torch.Tensor") -> List[str]:
        """Greedy best-path decode to composed strings."""
        return greedy_decode(log_probs, self.flat_chars, blank=0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
