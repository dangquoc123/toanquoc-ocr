# vnocr — light, LLM-free Vietnamese OCR

A modular Vietnamese OCR stack that wins in a **narrow domain** — printed
Vietnamese + administrative tables — through specialisation, not scale. Every
component runs on CPU, the neural nets total **<30 MB** (FP32; ~8–10 MB INT8),
and **there is no LLM anywhere in the inference path**.

> Full design rationale: [`docs/DESIGN.md`](docs/DESIGN.md). Code references its
> sections (§) throughout.

## The one idea

Vietnamese OCR errors are **tone-dominated**: the base letters read fine, the
tone marks (sắc/huyền/hỏi/ngã/nặng) don't. Two consequences shape everything:

1. **Recognition** factorises each glyph into `base + modifier + tone` so the
   rare diacritic-dense glyphs (ữ, ặ, ỡ) stop starving for data (Lever 1, §3.2),
   reads at 48px with a height-preserving stride schedule so tone marks survive
   downsampling (Lever 2, §3.3).
2. **Tone recovery needs context, not an LLM.** A single syllable's tone is
   lexically ambiguous (ma/má/mà/mã/mạ/mả are all real); only context resolves
   it — and the required context is a **bigram/trigram**, which sits near the
   information-theoretic (Fano) floor. An LLM here is paying for what an n-gram
   already does (§0).

## Architecture (§2)

```
image → preprocess → layout ─┬─ text  → detect → recognise → post-process (trie+noisy+n-gram)
                             └─ table → structure (morphology | SLANet) → per-cell recognise
                                                        ↓
                                              merge → text / JSON / HTML
```

Modular and **non-generative**: no block emits language, so it cannot
hallucinate; boxes are exact; each block is swappable and debuggable on its own.

## Install (layered — the core needs nothing)

The linguistic core (`charset`, `postprocess`, `eval`) is **pure stdlib**.

```bash
pip install -e '.[cv]'     # + numpy/opencv: preprocess, table morphology, layout
pip install -e '.[train]'  # + torch/pillow: recogniser, detector, SLANet, training
pip install -e '.[dev]'    # everything + pytest
```

## Quickstart — the pure-Python core (no deps)

```python
from vnocr.charset import decompose, compose, normalize_text, generate_syllables

d = decompose("ữ")          # Decomp(base='u', modifier=HORN, tone=TILDE)
compose(d.base, d.modifier, d.tone)      # 'ữ'
normalize_text("hoà bình")  # 'hòa bình'  (canonical tone placement, §1.6)

# p_B / p_T split (§8) — the metric no public benchmark reports
from vnocr.eval import evaluate
m = evaluate(["tôi yêu tiếng việt"], ["tối yêu tiếng viet"])
print(m.summary())          # p_B=1.0000, p_T<1.0  → errors are tone-dominated

# non-LLM post-processing (§5): trie + noisy channel + n-gram tone recovery
from vnocr.postprocess import SyllableTrie, NoisyChannel, CountLanguageModel, \
    ToneRecovery, PostProcessor
trie = SyllableTrie(generate_syllables())
lm = CountLanguageModel(order=3).train(open("corpus.txt", encoding="utf-8"))
pp = PostProcessor(trie=trie, noisy=NoisyChannel(trie),
                   tone=ToneRecovery(lm, trie))
pp.run("Vlệt nam")          # 'Việt nam'  (frame error fixed, structurally)
```

## CLI

```bash
# (re)generate the syllable inventory + print the factorised-vs-flat class counts
python3 scripts/build_charset.py --out data/charset/syllables.txt

# size the LM before training: is a bigram enough, or do you need a trigram? (§8)
python3 scripts/measure_entropy.py corpus.txt

# train the tone-recovery LM (pure-Python count model, or emit the KenLM cmd)
python3 scripts/train_lm.py corpus.txt --order 3 --out data/lm/vi.count.pkl

# score predictions with the p_B / p_T split
python3 scripts/evaluate.py --ref gt.txt --hyp pred.txt

# train the recogniser (needs a GPU; §7)
python3 scripts/train_recognizer.py --train train.txt --val val.txt --gtc

# full pipeline on an image
python3 scripts/infer.py page.jpg --ckpt checkpoints/recognizer.pt \
    --syllables data/charset/syllables.txt --lm data/lm/vi.count.pkl --format json
```

> Scripts are also directly executable (`./scripts/build_charset.py`) thanks to
> their `#!/usr/bin/env python3` shebang. On macOS there is no bare `python` —
> use `python3`, or run the `make` targets which default to it.

## Training a real model → reading real images

There is **no pretrained checkpoint** — recognition needs training first. The
full path (data → CPU smoke test → GPU/Kaggle train → evaluate → infer) is in
**[docs/TRAINING.md](docs/TRAINING.md)**. The short version:

```bash
pip install -e '.[train]'
python3 scripts/smoke_train.py                 # validate the machinery on CPU first
python3 scripts/build_synth_data.py --corpus corpus.txt --fonts /System/Library/Fonts \
    --out data/synth --n 200000                # generate labelled line images (§6.2)
python3 scripts/train_recognizer.py --train data/synth/train.txt \
    --val data/synth/val.txt --gtc --out checkpoints/recognizer.pt   # GPU (§3, §7)
python3 scripts/infer.py page.jpg --ckpt checkpoints/recognizer.pt \
    --syllables data/charset/syllables.txt --lm data/lm/vi.count.pkl # read an image
```

Always run `smoke_train.py` before spending GPU hours — it overfits one random
batch on CPU and asserts the shapes, budget, loss curve, and decode are all
correct.

**No GPU?** A click-to-run notebook for **Kaggle's free GPU** (T4/P100,
30 h/week) is at [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) —
step-by-step setup in [docs/KAGGLE.md](docs/KAGGLE.md), and `make kaggle-zip`
packages this repo for upload as a Kaggle Dataset.

## Module map

| Package | Design | Needs | What |
|---|---|---|---|
| `vnocr.charset` | §3.2, §1.6, §5.1 | stdlib | base/modifier/tone decomposition, NFC + tone placement, validity mask, syllable inventory |
| `vnocr.eval` | §8, §0 | stdlib | CER/WER, **p_B/p_T split**, Miller–Madow tone entropy |
| `vnocr.postprocess` | §5 | stdlib (KenLM optional) | syllable trie, noisy-channel corrector, n-gram tone recovery, dictionary-constrained CTC beam |
| `vnocr.preprocess` | §1 | numpy, opencv | grayscale deskew/CLAHE/bilateral; binarise (tables only) |
| `vnocr.layout` | §2 | numpy, opencv | text/table region routing (heuristic + model hook) |
| `vnocr.table` | §4 | numpy, opencv (+torch for SLANet) | exact morphology grid + SLANet fallback |
| `vnocr.detect` | §3 | torch | DBNet |
| `vnocr.recognize` | §3 | torch | SVTR backbone (Lever 2) + factorised head (Lever 1) + CTC/GTC |
| `vnocr.pipeline` | §2 | (per components) | end-to-end orchestration → text/JSON/HTML |

## Tests

The core suite runs with **no third-party packages**:

```bash
make test-core          # plain python3, 36 tests
# or, with pytest installed:
make test
```

## Roadmap (§9) — validate cheapest-first

1. **48px + asymmetric stride** — one config line (Lever 2). ← cheapest
2. **Dictionary-constrained decoding** — no retraining (§5.1).
3. **Bigram tone recovery** — train KenLM on CPU (§5.3).
4. **Factorised diacritic head** — needs retraining **+ a real ablation** (§3.2).
   ← highest risk, do last.

## Honest caveats (see the Appendix in `docs/DESIGN.md`)

- This does **not** beat general OCR everywhere — it trades generality for the
  Vietnamese-printed + admin-table domain.
- The **factorised head (§3.2) is an untested hypothesis** — the other three
  levers are firmer. Ablate it (`train_recognizer.py --no-interaction` toggles
  the β table; run flat-vs-factorised and compare **tone accuracy on rare
  glyphs**).
- Everything rides on **synthetic-data quality and diacritic-correct fonts**
  (`build_synth_data.py --verify-fonts`).
- **No public benchmark measures DeepSeek-OCR / Paddle on Vietnamese** — every
  accuracy claim must be re-run on your own labelled set.
- The budget numbers (p_B≈0.5%, p_T≈4%, Fano floor ~0.74%) are **illustrative** —
  plug your measurements into `scripts/measure_entropy.py` and `evaluate.py`.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
