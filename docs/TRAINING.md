# Training guide — from zero to reading a real image

This is the real path (§9 Phase 1–3). Recognition needs a **trained model**;
there is no pretrained checkpoint in the repo. The route:

```
corpus + fonts → synthetic data → [CPU smoke test] → train on GPU → evaluate (p_B/p_T)
                                                                          ↓
                              train n-gram LM → run pipeline on a real image
```

Nothing here needs a GPU **except** the recogniser/detector training itself.
Kaggle's free T4/P100 (30 h/week) is enough for the first stages (§7.1).

---

## 0. Install

```bash
pip install -e '.[train]'      # torch + numpy + opencv + pillow
```

The pure-Python core (charset/postprocess/eval) needs nothing; this adds the
neural + image stack.

---

## 1. Get a text corpus (§6.3)

Used for **both** the synthetic image text and the tone-recovery LM.

- Starter (in the repo): [`data/corpus/vi_sample.txt`](../data/corpus/vi_sample.txt)
  — ~40 original sentences, enough to smoke-test the *plumbing*, **not** to train
  a usable model.
- Production: a Vietnamese **Wikipedia dump** + a **news** corpus (tens of
  millions of sentences). Extract to plain UTF-8 text, one sentence/line, then
  normalise:

```python
from vnocr.charset import normalize_text
with open("corpus.raw") as fi, open("corpus.txt", "w") as fo:
    for line in fi:
        fo.write(normalize_text(line))
```

Size the LM order first (this decides bigram vs trigram, §8):

```bash
python3 scripts/measure_entropy.py corpus.txt
```

---

## 2. Get diacritic-correct fonts (§6.2 — do NOT skip)

Many Latin fonts render Vietnamese diacritics wrong; a bad font silently
poisons labels. Use open fonts with full Vietnamese coverage:

- **Noto Sans / Noto Serif** (Google, OFL) — excellent Vietnamese coverage.
- **Be Vietnam Pro**, **Sarabun**, **Roboto** (Google Fonts, OFL).
- macOS system fonts also work (`/System/Library/Fonts`, `/Library/Fonts`) —
  Times New Roman, Arial, and many `.ttc` files cover Vietnamese.

Put `.ttf/.otf/.ttc` in `fonts/`. The generator verifies every font against the
stress set (ữ ặ ỡ ẫ …) and **drops** any that fail:

```bash
# quick check of which fonts are usable, before generating anything
python3 - <<'PY'
from scripts.build_synth_data import find_fonts, verify_font
for f in find_fonts("fonts/,/System/Library/Fonts"):
    print("OK " if verify_font(f) else "BAD", f)
PY
```

---

## 3. Generate synthetic training data (§6.2)

```bash
python3 scripts/build_synth_data.py \
    --corpus corpus.txt \
    --fonts fonts/ \
    --out data/synth --n 200000 --mode mixed
```

- `--mode mixed` interleaves real corpus **lines** (realism) and random **word
  salad** (n-gram diversity / augmentation).
- Output: `data/synth/images/*.png` + `data/synth/labels.txt`
  (`image_path<TAB>text`) — the manifest format the dataset expects.

Split into train/val (e.g. 95/5):

```bash
shuf data/synth/labels.txt > /tmp/all.txt
split -l $(( $(wc -l < /tmp/all.txt) * 95 / 100 )) /tmp/all.txt data/synth/split_
mv data/synth/split_aa data/synth/train.txt
cat data/synth/split_* 2>/dev/null | tail -n +$(( $(wc -l < data/synth/train.txt)+1 )) > data/synth/val.txt
```

Add the **real** datasets (§6.1: VinText, 5CD-AI handwriting, …) as extra
manifests in the same format and concatenate.

---

## 4. Validate the machinery on CPU (before any GPU)

```bash
python3 scripts/smoke_train.py          # CTC only
python3 scripts/smoke_train.py --gtc    # + NRTR teacher (§3.1)
```

Overfits one random batch and asserts: correct output shape `[B, W/4, classes]`,
param count within budget, loss drops sharply, greedy decode recovers the
targets. If this passes, the training code is correct — only then spend GPU time.

---

## 5. Train the recogniser (§3, §7)

### Local GPU
```bash
python3 scripts/train_recognizer.py \
    --train data/synth/train.txt --val data/synth/val.txt \
    --epochs 30 --batch 128 --lr 1e-3 --gtc \
    --out checkpoints/recognizer.pt
```

- `--gtc` — Guided Training of CTC (§3.1); teacher dropped at export.
- `--no-interaction` — the §3.2 **ablation** (flat-additive vs β table). Run
  both and compare **tone accuracy** to decide whether Lever 1's interaction
  term earns its keep.

### Kaggle (free T4/P100) — recommended when you have no local GPU
A ready-made, click-to-run notebook is in the repo:
**[notebooks/kaggle_train.ipynb](../notebooks/kaggle_train.ipynb)** — it fetches
the code (GitHub URL *or* an uploaded `make kaggle-zip` Dataset), verifies
fonts, generates synthetic data, smoke-tests, trains with `--gtc`, and saves
`recognizer.pt` + `vi.count.pkl` to the notebook Output.

Full walkthrough (upload, GPU/Internet settings, quota tips, troubleshooting):
**[docs/KAGGLE.md](KAGGLE.md)**.

Watch the `val:` line each epoch — it prints CER/WER **and the p_B/p_T split**.

---

## 6. Evaluate — and confirm errors are tone-dominated (§8)

```bash
# dump predictions vs ground truth, then:
python3 scripts/evaluate.py --ref gt.txt --hyp pred.txt
```

Expect `p_T` error ≫ `p_B` error. This is the number that tells you where to
spend the next effort (§0).

---

## 7. Train the tone-recovery LM + build the post-processor (§5)

```bash
python3 scripts/build_charset.py --out data/charset/syllables.txt
python3 scripts/train_lm.py corpus.txt --order 3 --out data/lm/vi.count.pkl
# (production: --backend kenlm, then build the pruned binary)
```

---

## 8. Read a real image (§2)

```bash
python3 scripts/infer.py page.jpg \
    --ckpt checkpoints/recognizer.pt \
    --syllables data/charset/syllables.txt \
    --lm data/lm/vi.count.pkl \
    --format json
```

The pipeline runs preprocess → layout → detect/recognise → post-process → merge.
(Detection is a stub callable by default — for multi-line pages, wire a trained
DBNet or a line-segmenter into `OCRPipeline(detector=…)`; single-line crops work
as-is.)

---

## 9. Optimise for production (§7.2–7.3)

```bash
python3 scripts/export_onnx.py --ckpt checkpoints/recognizer.pt \
    --out models/recognizer.onnx
```

Then QAT→INT8 (not PTQ — the tone margins are too small, §7.3) and serve with
ONNX Runtime / OpenVINO. **Re-check `p_T` after quantising** — that's where INT8
hurts.

---

## Reality check

- The design's accuracy claims are **not yet verified on Vietnamese** by anyone
  publicly (§Appendix). Every number must be re-measured on your labelled set.
- Synthetic-only models read clean print well but struggle on real scans — mix
  in real data (§6.1) and the degradation pipeline (already in
  `build_synth_data.py`).
- The factorised head (§3.2) is the one **unproven** lever — the ablation in
  step 5 is how you find out if it helps *your* data.
