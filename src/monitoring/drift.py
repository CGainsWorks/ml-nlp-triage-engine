"""Drift detection: is live traffic still shaped like the training data?

Population Stability Index over fixed bins. PSI is used rather than a KS test
because it is cheap to compute incrementally, it works the same way on the
categorical category mix as on continuous lengths, and operations teams already
read it. The conventional bands are applied:

    < 0.10   stable
    0.10-0.25 moderate shift, worth watching
    > 0.25   significant shift, investigate before trusting the model

Two families of reference are used:

* **Input drift** — text length and category mix, against the training split
  (`artifacts/data/processed/drift_reference.json`). Shared by all versions,
  because they were all trained on the same data.
* **Output drift** — the confidence distribution, against each version's own
  validation-set confidences (`artifacts/models/<version>/drift_reference.json`,
  written by `python -m src.monitoring.build_reference`). Per version, because
  a transformer and a TF-IDF model have genuinely different confidence shapes
  and comparing either to the other's would flag drift that is not there.

What this does **not** do is detect concept drift — labels changing meaning
while inputs look the same. Nothing here can, without ground truth arriving
later. See docs/LIMITATIONS.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.etl.stats import CHAR_BINS, CONFIDENCE_BINS, WORD_BINS, histogram
from src.monitoring.store import PredictionRecord

LOGGER = logging.getLogger(__name__)

STABLE = 0.10
SIGNIFICANT = 0.25

#: Smoothing so an empty bin does not send PSI to infinity.
_EPSILON = 1e-6


def psi(reference: Sequence[float], actual: Sequence[float]) -> float:
    """Population stability index between two proportion vectors."""
    ref = np.clip(np.asarray(reference, dtype=float), _EPSILON, None)
    act = np.clip(np.asarray(actual, dtype=float), _EPSILON, None)
    if ref.shape != act.shape:
        raise ValueError(f"bin mismatch: reference {ref.shape} vs actual {act.shape}")
    ref = ref / ref.sum()
    act = act / act.sum()
    return float(np.sum((act - ref) * np.log(act / ref)))


def band(value: float) -> str:
    if value < STABLE:
        return "stable"
    if value < SIGNIFICANT:
        return "moderate"
    return "significant"


@dataclass
class DriftReading:
    feature: str
    psi: float
    band: str
    n_observed: int
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "band": self.band,
            "n_observed": self.n_observed,
            "note": self.note,
        }


#: Below this many records PSI is dominated by sampling noise, so it is
#: reported but explicitly marked as not yet meaningful.
MIN_SAMPLES = 200


class DriftMonitor:
    """Compares a window of served predictions against stored references."""

    def __init__(
        self,
        input_reference: Optional[Dict[str, object]] = None,
        confidence_references: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None:
        self.input_reference = input_reference or {}
        self.confidence_references = confidence_references or {}

    @classmethod
    def from_artifacts(
        cls,
        versions: Sequence[str],
        processed_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
    ) -> "DriftMonitor":
        input_reference = {}
        path = Path(processed_dir) / "drift_reference.json"
        if path.exists():
            input_reference = json.loads(path.read_text(encoding="utf-8"))
        else:
            LOGGER.warning("no input drift reference at %s; input drift disabled", path)

        confidence_references: Dict[str, Dict[str, object]] = {}
        for version in versions:
            candidate = Path(models_dir) / version / "drift_reference.json"
            if candidate.exists():
                confidence_references[version] = json.loads(
                    candidate.read_text(encoding="utf-8")
                )
            else:
                LOGGER.warning(
                    "no confidence reference for %s; run "
                    "`python -m src.monitoring.build_reference`", version
                )
        return cls(input_reference, confidence_references)

    # -- readings --------------------------------------------------------

    def evaluate(
        self, version: str, records: List[PredictionRecord]
    ) -> List[DriftReading]:
        if not records:
            return []

        readings: List[DriftReading] = []
        note = (
            "" if len(records) >= MIN_SAMPLES
            else f"fewer than {MIN_SAMPLES} records; treat as noise"
        )

        text_ref = self.input_reference.get("text_length", {})
        if text_ref:
            for key, bins, values in (
                ("text_chars", CHAR_BINS, [r.n_chars for r in records]),
                ("text_words", WORD_BINS, [r.n_words for r in records]),
            ):
                reference = text_ref.get(key.split("_")[1], {}).get("proportions")
                if reference:
                    value = psi(reference, histogram(values, bins))
                    readings.append(
                        DriftReading(key, value, band(value), len(records), note)
                    )

        mix_ref = self.input_reference.get("category_mix", {})
        if mix_ref:
            labels = list(mix_ref["labels"])
            observed = [
                sum(1 for r in records if r.category == label) / len(records)
                for label in labels
            ]
            value = psi(mix_ref["proportions"], observed)
            caveat = (
                "compares predicted categories against the training label mix, "
                "so model error and true drift are confounded here"
            )
            readings.append(
                DriftReading("predicted_category_mix", value, band(value), len(records),
                             "; ".join(filter(None, (note, caveat))))
            )

        conf_ref = self.confidence_references.get(version, {}).get("confidence")
        if conf_ref:
            observed = histogram([r.confidence for r in records], CONFIDENCE_BINS)
            value = psi(conf_ref["proportions"], observed)
            readings.append(
                DriftReading("confidence", value, band(value), len(records), note)
            )

        return readings

    def report(
        self, records_by_version: Dict[str, List[PredictionRecord]]
    ) -> Dict[str, object]:
        return {
            "thresholds": {"stable_below": STABLE, "significant_above": SIGNIFICANT},
            "min_samples_for_signal": MIN_SAMPLES,
            "has_input_reference": bool(self.input_reference),
            "versions_with_confidence_reference": sorted(self.confidence_references),
            "by_version": {
                version: [reading.as_dict() for reading in self.evaluate(version, records)]
                for version, records in records_by_version.items()
            },
        }
