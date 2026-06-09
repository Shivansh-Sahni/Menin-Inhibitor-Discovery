.PHONY: all data models report test clean

all: data models report

data:
	python3 scripts/run_pipeline.py --stage data

models:
	python3 scripts/run_pipeline.py --stage models --skip-network

report:
	python3 scripts/run_pipeline.py --stage report --skip-network

test:
	python3 -m pytest

clean:
	rm -rf data/interim/* models/* reports/figures/*
