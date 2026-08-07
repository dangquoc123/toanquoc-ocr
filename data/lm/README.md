# Language-model data (Design §5.3, §6.3)

The tone-recovery LM is trained on **plain Vietnamese text** — no images, no
labels, no GPU. Wikipedia dump + a news corpus (tens of millions of sentences)
is the target.

## Count model (pure Python, dev)
```bash
python scripts/train_lm.py corpus.txt --order 3 --out data/lm/vi.count.pkl
```
Produces a pickled `CountLanguageModel` (stupid-backoff). Zero dependencies;
good for development and small corpora.

## KenLM (production)
Smaller and faster. Needs the KenLM toolkit:
```bash
lmplz -o 3 --prune 0 0 1 < corpus.txt > data/lm/vi.arpa
build_binary data/lm/vi.arpa data/lm/vi.binary
```
Then load with `KenLMModel("data/lm/vi.binary")`. Pruned binaries land around
20–40 MB (§5.3).

## Sizing the order first
Run `python scripts/measure_entropy.py corpus.txt` **before** training. If
`H(t | s, w-1)` is 0.05–0.15 bit a bigram suffices; if higher, use a trigram
(§8). This is how you avoid buying an LLM to do a bigram's job (§0).
