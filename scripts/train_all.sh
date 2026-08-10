#!/usr/bin/env bash
# Build every artefact the service needs, then regenerate every reported number.
#
#   bash scripts/train_all.sh
#
# This is what the `trainer` container runs. Safe to re-run: with
# TRIAGE_SKIP_IF_TRAINED=1 (the default) it exits early when models already
# exist, so `docker compose up` a second time starts the API immediately
# instead of retraining for an hour.

set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${TRIAGE_EPOCHS:-8}"
LR="${TRIAGE_LR:-5e-5}"
SKIP_IF_TRAINED="${TRIAGE_SKIP_IF_TRAINED:-1}"
ARTIFACTS="${TRIAGE_ARTIFACTS_DIR:-artifacts}"

if [ "$SKIP_IF_TRAINED" = "1" ] && [ -f "$ARTIFACTS/models/transformer-distilbert-v1/metadata.json" ]; then
    echo "==> models already trained under $ARTIFACTS/models; nothing to do"
    echo "    set TRIAGE_SKIP_IF_TRAINED=0 to force a rebuild"
    exit 0
fi

if [ "$EPOCHS" != "8" ]; then
    echo "!!  TRIAGE_EPOCHS=$EPOCHS, not the default 8."
    echo "!!  The transformer will be under-trained and the results will NOT"
    echo "!!  match the numbers reported in the README."
fi

echo "==> 1/5 taxonomy"
python -m src.etl.render_taxonomy

echo "==> 2/5 ETL"
python -m src.etl.build_dataset

echo "==> 3/5 scikit-learn baselines"
python -m src.models.baseline.train --model both

echo "==> 4/5 transformer fine-tune (${EPOCHS} epochs, lr ${LR})"
python -m src.models.transformer.train --epochs "$EPOCHS" --lr "$LR"

echo "==> 5/5 drift references and experiments"
python -m src.monitoring.build_reference
python -m experiments.compare_baseline_transformer
python -m experiments.tune_confidence_threshold
python -m experiments.sentiment_face_validity
python -m experiments.run_ab_experiment

echo
echo "==> done. Reports are in experiments/results/"
