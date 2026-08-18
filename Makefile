.PHONY: all collect curate quality models analyze report manifest verify smoke test lint typecheck format

PYTHON ?= python3
PIPELINE ?= $(PYTHON) -m menin_discovery.cli

all:
	$(PIPELINE) --stage all --skip-network

collect:
	$(PIPELINE) --stage collect

curate:
	$(PIPELINE) --stage curate --skip-network

quality:
	$(PIPELINE) --stage quality --skip-network

models:
	$(PIPELINE) --stage models --skip-network

analyze:
	$(PIPELINE) --stage analyze --skip-network

report:
	$(PIPELINE) --stage report --skip-network

manifest:
	$(PIPELINE) --stage manifest --skip-network

verify:
	$(PIPELINE) --stage verify --skip-network

smoke:
	$(PIPELINE) --stage all --skip-network --fast

test:
	$(PYTHON) -m pytest -q pipeline/tests
	$(PYTHON) -m pytest -q -c packages/menin-edit/pyproject.toml packages/menin-edit/tests

lint:
	$(PYTHON) -m ruff check pipeline/src pipeline/scripts pipeline/tests packages/menin-edit/src packages/menin-edit/tests

typecheck:
	$(PYTHON) -m mypy pipeline/src/menin_discovery packages/menin-edit/src/menin_edit

format:
	$(PYTHON) -m ruff format pipeline/src pipeline/scripts pipeline/tests packages/menin-edit/src packages/menin-edit/tests
