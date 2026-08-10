"""Write each model's confidence reference distribution for drift monitoring.

    python -m src.monitoring.build_reference

Runs every registered version over the validation split and stores the shape of
its top-1 probability distribution in ``artifacts/models/<version>/
drift_reference.json``. The live drift monitor compares serving confidence
against this.

Validation is used rather than train, because a model's confidence on data it
was fitted on is pushed towards 1.0 and would make ordinary live traffic look
like drift.
"""

from __future__ import annotations

import argparse
import json
import logging

from src.config import MODELS_DIR, ensure_dirs
from src.etl.stats import CONFIDENCE_BINS, histogram
from src.logging_setup import configure
from src.models.data import load_split
from src.models.registry import ModelRegistry

LOGGER = logging.getLogger("drift-reference")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="val", choices=("val", "test"))
    args = parser.parse_args(argv)

    configure()
    ensure_dirs()

    frame = load_split(args.split)
    texts = frame["text"].tolist()
    registry = ModelRegistry()
    if not registry.versions:
        LOGGER.error("no trained models found in %s", MODELS_DIR)
        return 1

    for version in registry.versions:
        model = registry.get(version)
        probabilities = model.predict_proba(texts)
        confidence = probabilities.max(axis=1)
        payload = {
            "version": version,
            "built_from": f"{args.split} split ({len(texts)} rows)",
            "confidence": {
                "bin_edges": CONFIDENCE_BINS,
                "proportions": histogram(confidence.tolist(), CONFIDENCE_BINS),
                "mean": round(float(confidence.mean()), 4),
                "p05": round(float(sorted(confidence)[max(0, len(confidence) // 20)]), 4),
            },
        }
        path = MODELS_DIR / version / "drift_reference.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("%s: mean confidence %.4f -> %s",
                    version, payload["confidence"]["mean"], path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
