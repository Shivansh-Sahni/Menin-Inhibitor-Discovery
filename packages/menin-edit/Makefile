PYTHON ?= python

.PHONY: test lint audit score-demo optimize-demo

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

audit:
	$(PYTHON) -m menin_edit.cli --config config/default.yaml audit --load-models

score-demo:
	$(PYTHON) -m menin_edit.cli --config config/default.yaml score --smiles "CCN1CCCCC1"

optimize-demo:
	$(PYTHON) -m menin_edit.cli --config config/default.yaml optimize \
		--request examples/request.yaml --output-dir artifacts/sessions/tutorial --top 5
