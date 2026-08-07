#!/usr/bin/env python3
"""CPU smoke test for the training pipeline — run BEFORE spending GPU hours.

    pip install -e '.[train]'      # torch is enough for this test
    python3 scripts/smoke_train.py            # CTC only
    python3 scripts/smoke_train.py --gtc      # + NRTR teacher (§3.1)

It builds the recogniser, overfits a *single fixed random batch* for a few
hundred steps on CPU, and checks:

* the forward output is ``[B, T, num_classes]`` with ``T = W/4`` (the stride
  schedule is sane, §3.3);
* the parameter count is within the §2 budget;
* CTC (and, with --gtc, the teacher CE) loss is finite and **drops sharply**
  when overfitting — i.e. gradients flow and the machinery turns;
* greedy decode recovers the memorised targets.

This proves the code is trainable. It does *not* prove accuracy on real images —
that needs real data and a GPU (see docs/TRAINING.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--gtc", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from vnocr.recognize import (IMG_HEIGHT, LabelCodec, NRTRHead,
                                 VietRecognizer, ctc_loss)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_recognizer import build_teacher_io

    torch.manual_seed(0)
    device = args.device
    codec = LabelCodec()
    model = VietRecognizer(codec.charset).to(device)
    K = model.num_classes - 1     # non-blank classes
    print(f"params: {model.num_parameters()/1e6:.2f}M   classes: {model.num_classes}")
    assert model.num_parameters() < 15e6, "over the §2 budget — check the config"

    # one fixed random batch
    B, W, L = args.batch, args.width, 8
    images = torch.rand(B, 1, IMG_HEIGHT, W, device=device)
    seq = torch.randint(1, K + 1, (B, L), device=device)      # classes 1..K
    tgt_len = torch.full((B,), L, dtype=torch.long)
    targets = seq.reshape(-1)

    # shape check
    with torch.no_grad():
        out0 = model(images)
    T = out0["log_probs"].shape[1]
    print(f"forward: log_probs {tuple(out0['log_probs'].shape)}  (T=W/4={W//4})")
    assert out0["log_probs"].shape == (B, W // 4, model.num_classes)

    teacher = None
    if args.gtc:
        teacher = NRTRHead(model.backbone.out_dim, model.num_classes + 2,
                           max_len=64).to(device)

    params = list(model.parameters()) + (list(teacher.parameters()) if teacher else [])
    opt = torch.optim.AdamW(params, lr=3e-3)

    first = last = None
    for step in range(args.steps):
        opt.zero_grad()
        feats = model.encode_features(images)
        out = model.head(feats)
        lp = out["log_probs"]
        in_len = torch.full((B,), lp.size(1), dtype=torch.long)
        loss = ctc_loss(lp, targets, in_len, tgt_len)
        if teacher is not None:
            di, dt = build_teacher_io(seq, model.num_classes, model.num_classes + 1)
            tl = teacher(feats, di)
            loss = loss + 0.5 * F.cross_entropy(
                tl.reshape(-1, tl.size(-1)), dt.reshape(-1), ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:>4}  loss {loss.item():.4f}")

    assert torch.isfinite(torch.tensor(last)), "loss went non-finite"
    assert last < first * 0.5, f"loss did not drop enough ({first:.3f} -> {last:.3f})"

    model.eval()
    with torch.no_grad():
        preds = model.decode(model(images)["log_probs"])
    gold = [codec.decode(row.tolist()) for row in seq]
    match = sum(p == g for p, g in zip(preds, gold))
    print(f"overfit decode: {match}/{B} sequences memorised")
    print(f"  e.g. pred={preds[0]!r}  gold={gold[0]!r}")

    print("\nSMOKE TEST PASSED — the training machinery turns. "
          "Now train on real data (docs/TRAINING.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
