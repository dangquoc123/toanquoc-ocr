"""Guided Training of CTC (GTC) — Design §3.1.

Train the fast student (SVTR + CTC, conditional-independence) under the guidance
of a stronger attention teacher (an NRTR-style seq2seq that models
``p(y_u | y_<u)``)::

    L = λ₁·L_CTC + λ₂·L_KD

At **inference the teacher is discarded** — only SVTR+CTC runs, so you keep CTC
speed with accuracy close to attention.  Because the encoder is shared, the
teacher's gradients shape it into producing more discriminative features (the
KD signal is most valuable on the diacritic-dense rare classes, §7.2).

This module provides:

* :class:`NRTRHead` — a lightweight Transformer-decoder teacher head.
* :func:`kd_loss` — temperature-scaled KL from teacher to student.
* :class:`GTCTrainer` mixin/utility to combine the losses.

The teacher head is only instantiated during training; :class:`~vnocr.recognize
.model.VietRecognizer` runs without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.recognize needs PyTorch (`pip install torch`).") from _e

__all__ = ["NRTRHead", "kd_loss", "GTCConfig"]


class NRTRHead(nn.Module):
    """Transformer-decoder teacher (NRTR-style) over shared encoder features.

    Autoregressive: models ``p(y_u | y_<u, features)``.  Used only to guide the
    student; thrown away at inference (§3.1).
    """

    def __init__(self, enc_dim: int, num_classes: int, dim: int = 256,
                 heads: int = 8, layers: int = 4, max_len: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_classes, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.enc_proj = nn.Linear(enc_dim, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(dim, num_classes)
        self.max_len = max_len

    def forward(self, features: "torch.Tensor",
                tgt_in: "torch.Tensor") -> "torch.Tensor":
        """``features`` ``[B, T, enc_dim]``, ``tgt_in`` ``[B, L]`` → logits ``[B, L, C]``."""
        mem = self.enc_proj(features)
        L = tgt_in.size(1)
        y = self.embed(tgt_in) + self.pos[:, :L]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=y.device), 1)
        h = self.decoder(y, mem, tgt_mask=causal)
        return self.out(h)


def kd_loss(student_logits: "torch.Tensor", teacher_logits: "torch.Tensor",
            temperature: float = 2.0) -> "torch.Tensor":
    """Temperature-scaled KL(teacher ‖ student), the §7.2 distillation term.

    The value is in the distribution over the *wrong* classes — the teacher says
    "this is ``ẫ``; next most likely ``ầ``/``ẩ``, never ``k``" — transferring
    visual-similarity structure, which is exactly what the diacritic-dense
    inventory needs.
    """
    t = temperature
    s_log = F.log_softmax(student_logits / t, dim=-1)
    t_prob = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(s_log, t_prob, reduction="batchmean") * (t * t)


@dataclass
class GTCConfig:
    lambda_ctc: float = 1.0
    lambda_kd: float = 0.5
    temperature: float = 2.0
    # optional auxiliary per-factor cross-entropy weights (base/mod/tone heads)
    lambda_base: float = 0.0
    lambda_mod: float = 0.0
    lambda_tone: float = 0.0
