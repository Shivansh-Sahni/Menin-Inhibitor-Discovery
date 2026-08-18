.PHONY: all collect curate quality models analyze report manifest verify smoke test lint typecheck format platform-status platform-contracts platform-normalize-chembl platform-canonicalize-chembl platform-verify-canonical-determinism platform-normalize-external platform-verify-external-normalized platform-analyze-external-admission platform-verify-external-admission platform-analyze-deep-leakage platform-verify-deep-leakage platform-prepare-structure-metadata platform-verify-structure-metadata platform-prepare-context-splits platform-verify-context-splits platform-prepare-clinical-results platform-verify-clinical-results platform-prepare-regulatory-records platform-verify-regulatory-records platform-acquire-pkdb-candidates platform-prepare-pkdb-candidates platform-verify-pkdb-candidates platform-analyze-canonical platform-verify-statistical-analysis platform-prepare-split-suite platform-verify-split-suite platform-prepare-static platform-prepare-corpus-readiness platform-verify-corpus-readiness platform-audit-non-hpc-governance platform-verify-non-hpc-completion platform-verify-final-artifacts

PYTHON ?= python3
PIPELINE ?= $(PYTHON) -m menin_discovery.cli
PLATFORM ?= $(PYTHON) -m menin_discovery.platform_cli
PLATFORM_EVIDENCE_DATE ?= 2026-08-04

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
	$(PYTHON) -m pytest --cov=menin_discovery --cov-report=term-missing
	$(PYTHON) -m pytest -q -c packages/menin-edit/pyproject.toml packages/menin-edit/tests

lint:
	$(PYTHON) -m ruff check pipeline/src pipeline/scripts pipeline/tests packages/menin-edit/src packages/menin-edit/tests

typecheck:
	$(PYTHON) -m mypy pipeline/src/menin_discovery packages/menin-edit/src/menin_edit

format:
	$(PYTHON) -m ruff format pipeline/src pipeline/scripts pipeline/tests packages/menin-edit/src packages/menin-edit/tests

platform-status:
	$(PLATFORM) status

platform-contracts:
	$(PLATFORM) contracts

platform-normalize-chembl:
	$(PLATFORM) normalize-chembl-exports

platform-canonicalize-chembl:
	$(PLATFORM) canonicalize-chembl

platform-verify-canonical-determinism:
	$(PLATFORM) verify-canonical-determinism

platform-normalize-external:
	$(PLATFORM) normalize-external

platform-verify-external-normalized:
	$(PLATFORM) verify-external-normalized

platform-analyze-external-admission:
	$(PLATFORM) analyze-external-admission

platform-verify-external-admission:
	$(PLATFORM) verify-external-admission

platform-analyze-deep-leakage:
	$(PLATFORM) analyze-deep-leakage

platform-verify-deep-leakage:
	$(PLATFORM) verify-deep-leakage

platform-prepare-structure-metadata:
	$(PLATFORM) prepare-structure-metadata

platform-verify-structure-metadata:
	$(PLATFORM) verify-structure-metadata

platform-prepare-context-splits:
	$(PLATFORM) prepare-context-splits

platform-verify-context-splits:
	$(PLATFORM) verify-context-splits

platform-prepare-clinical-results:
	$(PLATFORM) prepare-clinical-results

platform-verify-clinical-results:
	$(PLATFORM) verify-clinical-results

platform-prepare-regulatory-records:
	$(PLATFORM) prepare-regulatory-records

platform-verify-regulatory-records:
	$(PLATFORM) verify-regulatory-records

platform-acquire-pkdb-candidates:
	$(PLATFORM) acquire-pkdb-candidates

platform-prepare-pkdb-candidates:
	$(PLATFORM) prepare-pkdb-candidates

platform-verify-pkdb-candidates:
	$(PLATFORM) verify-pkdb-candidates

platform-analyze-canonical:
	$(PLATFORM) analyze-canonical

platform-verify-statistical-analysis:
	$(PLATFORM) verify-statistical-analysis

platform-prepare-split-suite:
	$(PLATFORM) prepare-split-suite

platform-verify-split-suite:
	$(PLATFORM) verify-split-suite

platform-prepare-static:
	$(PLATFORM) prepare-static --evidence-checked-date $(PLATFORM_EVIDENCE_DATE)

platform-prepare-corpus-readiness:
	$(PLATFORM) prepare-corpus-readiness

platform-verify-corpus-readiness:
	$(PLATFORM) verify-corpus-readiness

platform-audit-non-hpc-governance:
	$(PLATFORM) audit-non-hpc-governance

platform-verify-non-hpc-completion:
	$(PLATFORM) verify-non-hpc-completion

platform-verify-final-artifacts:
	$(PLATFORM) verify-final-artifacts
