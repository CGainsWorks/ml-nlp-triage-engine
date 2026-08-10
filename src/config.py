"""Paths and constants shared across the pipeline.

Every path is derived from the repository root so the same code works from a
checkout, from a container (`WORKDIR /app`) and from pytest.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Committed inputs.
DATA_DIR = REPO_ROOT / "data"
TAXONOMY_PATH = DATA_DIR / "taxonomy.json"
TAXONOMY_MD_PATH = DATA_DIR / "TAXONOMY.md"
CONFIG_DIR = REPO_ROOT / "config"
AB_CONFIG_PATH = CONFIG_DIR / "ab_config.json"
ROUTING_POLICY_PATH = CONFIG_DIR / "routing_policy.json"

# Committed outputs (small, text-only: metrics, tables, reports).
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Build artefacts (never committed). Overridable so tests can redirect them.
ARTIFACTS_DIR = Path(os.environ.get("TRIAGE_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
RAW_DATA_DIR = ARTIFACTS_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "data" / "processed"
MODELS_DIR = Path(os.environ.get("TRIAGE_MODELS_DIR", ARTIFACTS_DIR / "models"))

# Upstream dataset.
BANKING77_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data"
)

# Reproducibility.
SEED = 42
VAL_FRACTION = 0.15

# Model version identifiers used by the registry and the A/B layer.
BASELINE_LINEAR_VERSION = "baseline-linear-v1"
BASELINE_GBDT_VERSION = "baseline-gbdt-v1"
TRANSFORMER_VERSION = "transformer-distilbert-v1"

# Below this predicted-intent probability the record is flagged for a human
# rather than auto-routed. This is only the fallback for a version with no
# entry in config/routing_policy.json — the thresholds actually used are
# per-version, because the models' confidence distributions differ far too much
# to share one (mean top-1 probability runs from 0.79 to 0.96 across the three).
# Every value is chosen on the validation split against an explicit cost model:
# see experiments/tune_confidence_threshold.py.
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

# Number of alternative intents returned alongside the top prediction.
TOP_K_INTENTS = 3


def ensure_dirs() -> None:
    """Create the artefact directories. Safe to call repeatedly."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
