# ml-nlp-triage-engine
# `make demo` builds everything from nothing and serves it. `docker compose up`
# does the same in a container if you would rather not touch your Python.

PYTHON ?= python
PORT ?= 8000
# Override to shorten the fine-tune, e.g. `make transformer EPOCHS=2`.
# The numbers in the README come from the default.
EPOCHS ?= 8
LR ?= 5e-5

.DEFAULT_GOAL := help
.PHONY: help setup data baseline transformer train reference experiments \
        test lint serve demo docker-build docker-up docker-down clean clean-all

help:
	@echo "make setup        install pinned dependencies"
	@echo "make data         download BANKING77 and build the splits"
	@echo "make baseline     train the TF-IDF linear and gradient-boosting models"
	@echo "make transformer  fine-tune DistilBERT (EPOCHS=$(EPOCHS), CPU: roughly an hour)"
	@echo "make train        data + baseline + transformer + drift references"
	@echo "make experiments  regenerate every number quoted in the README"
	@echo "make test         run the test suite"
	@echo "make serve        run the API on port $(PORT) (needs models)"
	@echo "make demo         everything above, end to end, from nothing"
	@echo "make docker-up    same thing in Docker"
	@echo "make clean        remove trained models and processed data"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

data:
	$(PYTHON) -m src.etl.render_taxonomy
	$(PYTHON) -m src.etl.build_dataset

baseline:
	$(PYTHON) -m src.models.baseline.train --model both

transformer:
	$(PYTHON) -m src.models.transformer.train --epochs $(EPOCHS) --lr $(LR)

reference:
	$(PYTHON) -m src.monitoring.build_reference

train: data baseline transformer reference

experiments:
	$(PYTHON) -m experiments.compare_baseline_transformer
	$(PYTHON) -m experiments.tune_confidence_threshold
	$(PYTHON) -m experiments.sentiment_face_validity
	$(PYTHON) -m experiments.run_ab_experiment

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests experiments

serve:
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port $(PORT)

demo: train experiments
	bash scripts/run_demo.sh

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down -v

clean:
	rm -rf artifacts/models artifacts/data/processed artifacts/logs

clean-all: clean
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
