#!/usr/bin/env python3
"""Train the Vietnamese recogniser (Design §3, §7).

    python3 scripts/train_recognizer.py --train train.txt --val val.txt \
        --epochs 30 --batch 128 --gtc

Manifests are ``image_path<TAB>text`` (see vnocr.recognize.dataset).

* CTC is the solid default path.
* ``--gtc`` attaches the NRTR seq2seq teacher (§3.1): it is trained with
  teacher-forced cross-entropy over the **shared encoder**, so its gradient
  shapes the encoder into more discriminative features — most valuable on the
  diacritic-dense rare glyphs.  The teacher is used only here and is dropped at
  export; inference stays pure SVTR+CTC.

Validate the machinery on CPU first with ``scripts/smoke_train.py`` before
spending GPU hours.  Needs torch (+ Pillow/opencv for image IO):
``pip install -e '.[train]'``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _code_version() -> str:
    """Short git hash of the running checkout — makes stale code obvious."""
    import subprocess
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[1])
        return r.stdout.strip() or "unknown (not a git checkout)"
    except Exception:  # noqa: BLE001
        return "unknown"


def build_teacher_io(seq_targets, sos: int, eos: int, pad: int = 0):
    """From padded targets ``[B, L]`` build ``(decoder_input, decoder_target)``.

    decoder_input  = [SOS, y1, ..., yL]        (teacher forcing)
    decoder_target = [y1,  ..., yL, EOS]        (EOS placed after each real seq)
    Positions past each sequence's length are PAD (ignored by the CE loss).
    """
    import torch
    B, L = seq_targets.shape
    lengths = (seq_targets != pad).sum(dim=1)          # [B]
    dec_in = torch.full((B, L + 1), pad, dtype=torch.long, device=seq_targets.device)
    dec_in[:, 0] = sos
    dec_in[:, 1:] = seq_targets
    dec_tgt = torch.full((B, L + 1), pad, dtype=torch.long, device=seq_targets.device)
    dec_tgt[:, :L] = seq_targets
    for i in range(B):
        dec_tgt[i, int(lengths[i])] = eos              # EOS right after the target
    return dec_in, dec_tgt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gtc", action="store_true", help="Guided Training of CTC (§3.1)")
    ap.add_argument("--lambda-kd", type=float, default=0.5)
    ap.add_argument("--no-interaction", action="store_true",
                    help="ablation: drop the β interaction table (§3.2)")
    ap.add_argument("--clip", type=float, default=5.0, help="grad-norm clip")
    ap.add_argument("--out", default="checkpoints/recognizer.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from vnocr.recognize import (LabelCodec, LineRecognitionDataset, NRTRHead,
                                 VietRecognizer, collate_fn, ctc_loss)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"code  : {_code_version()}")

    codec = LabelCodec()
    model = VietRecognizer(codec.charset,
                           use_interaction=not args.no_interaction).to(device)
    print(f"recogniser: {model.num_parameters()/1e6:.2f}M params, "
          f"{model.num_classes} classes")

    train_ds = LineRecognitionDataset(args.train, codec=codec)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_fn, num_workers=args.workers,
                          drop_last=True)

    teacher = None
    sos = eos = None
    if args.gtc:
        # teacher vocab = CTC classes + SOS + EOS  (PAD reuses blank id 0)
        sos, eos = model.num_classes, model.num_classes + 1
        teacher = NRTRHead(model.backbone.out_dim, model.num_classes + 2,
                           max_len=192).to(device)
        print("GTC: NRTR teacher attached (dropped at export)")

    params = list(model.parameters()) + (list(teacher.parameters())
                                         if teacher else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, len(train_dl)))
    use_amp = device == "cuda"
    try:  # torch >= 2.3 unified AMP API; fall back for older builds
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        model.train()
        if teacher:
            teacher.train()
        running = running_ctc = running_kd = 0.0
        for batch in train_dl:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            tgt_len = batch["target_lengths"].to(device)

            opt.zero_grad()
            with torch.autocast("cuda", enabled=use_amp):
                feats = model.encode_features(images)     # shared encoder
                out = model.head(feats)
                lp = out["log_probs"]
                in_len = torch.full((images.size(0),), lp.size(1),
                                    dtype=torch.long, device=device)
                loss_ctc = ctc_loss(lp, targets, in_len, tgt_len)
                loss = loss_ctc
                loss_kd = torch.tensor(0.0, device=device)

                if teacher is not None:
                    dec_in, dec_tgt = build_teacher_io(
                        batch["seq_targets"].to(device), sos, eos)
                    t_logits = teacher(feats, dec_in)     # [B, L+1, vocab]
                    loss_kd = F.cross_entropy(
                        t_logits.reshape(-1, t_logits.size(-1)),
                        dec_tgt.reshape(-1), ignore_index=0)
                    loss = loss_ctc + args.lambda_kd * loss_kd

            scaler.scale(loss).backward()
            if args.clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            running_ctc += loss_ctc.item()
            running_kd += float(loss_kd)

        n = max(1, len(train_dl))
        msg = f"epoch {epoch+1:>3}/{args.epochs}  loss {running/n:.4f}  ctc {running_ctc/n:.4f}"
        if teacher:
            msg += f"  kd {running_kd/n:.4f}"
        print(msg)

        if args.val:
            _quick_eval(model, codec, args.val, device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # save ONLY the student — the teacher is training scaffolding (§3.1)
    torch.save({"model": model.state_dict(),
                "charset_bases": codec.charset.bases,
                "use_interaction": not args.no_interaction}, args.out)
    print(f"saved student -> {args.out}")
    return 0


def _quick_eval(model, codec, val_manifest, device):
    import torch
    from torch.utils.data import DataLoader

    from vnocr.eval import evaluate
    from vnocr.recognize import LineRecognitionDataset, collate_fn

    ds = LineRecognitionDataset(val_manifest, codec=codec)
    dl = DataLoader(ds, batch_size=64, collate_fn=collate_fn)
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for batch in dl:
            out = model(batch["images"].to(device))
            hyps.extend(model.decode(out["log_probs"]))
            refs.extend(batch["texts"])
    m = evaluate(refs, hyps)
    print("  val:", m.summary().replace("\n", " | "))


if __name__ == "__main__":
    raise SystemExit(main())
