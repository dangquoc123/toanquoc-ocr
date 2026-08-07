# Use python3 by default (macOS/Homebrew have no bare `python`).
# Override with:  make PYTHON=python3.12 test-core
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help test test-core charset entropy lint install install-dev clean kaggle-zip

help:
	@echo "vnocr — light, LLM-free Vietnamese OCR"
	@echo ""
	@echo "  make install       core only (stdlib) + classical CV extras"
	@echo "  make install-dev   everything + pytest"
	@echo "  make test          run the full test suite (pure-Python core)"
	@echo "  make test-core     run tests with plain python3 (no pytest needed)"
	@echo "  make charset       (re)generate data/charset/syllables.txt"
	@echo "  make entropy C=f   measure tone entropy on corpus file C"
	@echo "  make kaggle-zip    package the repo for upload as a Kaggle Dataset"
	@echo "  make clean         remove caches and build artefacts"
	@echo ""
	@echo "  (override interpreter with: make PYTHON=python3.12 <target>)"

install:
	$(PIP) install -e '.[cv]'

install-dev:
	$(PIP) install -e '.[dev]'

# The core suite needs no third-party packages.
test:
	$(PYTHON) -m pytest -q

test-core:
	@$(PYTHON) tests/test_charset.py
	@$(PYTHON) tests/test_postprocess.py
	@$(PYTHON) tests/test_eval.py

charset:
	$(PYTHON) scripts/build_charset.py --out data/charset/syllables.txt

entropy:
	$(PYTHON) scripts/measure_entropy.py $(C)

# Package the repo into dist/vnocr-kaggle.zip for upload as a Kaggle Dataset
# (used by notebooks/kaggle_train.ipynb when REPO_URL is left empty).
kaggle-zip:
	mkdir -p dist
	rm -f dist/vnocr-kaggle.zip
	zip -qr dist/vnocr-kaggle.zip . \
	    -x '.git/*' -x 'dist/*' -x 'checkpoints/*' -x 'models/*' \
	    -x 'data/synth/*' -x '*__pycache__*' -x '*.pyc' \
	    -x '.DS_Store' -x '*/.DS_Store' -x '.pytest_cache/*'
	@echo "-> dist/vnocr-kaggle.zip  ($$(du -h dist/vnocr-kaggle.zip | cut -f1))"
	@echo "   Upload at kaggle.com -> Datasets -> New Dataset, then Add Input"
	@echo "   in the notebook. See docs/KAGGLE.md."

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache \
	       build dist *.egg-info
	find . -name '*.pyc' -delete
