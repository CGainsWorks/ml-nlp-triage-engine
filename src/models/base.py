"""The interface every servable model satisfies.

The API layer never imports a concrete model class. It asks the registry for
something that looks like :class:`IntentClassifier` and calls ``predict``. That
is what makes A/B between a TF-IDF model and a transformer possible without the
service knowing which is which.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.config import TOP_K_INTENTS


@dataclass
class IntentPrediction:
    """One model's verdict on one record."""

    intent: str
    confidence: float
    top_k: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def margin(self) -> float:
        """Gap between the top and runner-up probability.

        A small margin means the model is torn between two labels, which is a
        different failure mode from uniformly low confidence and is worth
        routing on separately.
        """
        if len(self.top_k) < 2:
            return self.confidence
        return float(self.top_k[0][1] - self.top_k[1][1])


class IntentClassifier:
    """Base class holding the label space and the probability -> prediction step."""

    #: Registry identifier, e.g. ``baseline-linear-v1``.
    version: str = "unset"
    #: Family name used for grouping in reports: ``baseline`` or ``transformer``.
    family: str = "unset"
    #: Label order matching the columns of :meth:`predict_proba`.
    labels: List[str] = []

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(
        self, texts: Sequence[str], top_k: int = TOP_K_INTENTS
    ) -> List[IntentPrediction]:
        probabilities = self.predict_proba(texts)
        if probabilities.shape[1] != len(self.labels):
            raise ValueError(
                f"{self.version}: predict_proba returned {probabilities.shape[1]} "
                f"columns for {len(self.labels)} labels"
            )

        k = max(1, min(top_k, len(self.labels)))
        # argpartition gets the top k without a full sort of 77 columns per row.
        cut = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]

        out: List[IntentPrediction] = []
        for row, idx in zip(probabilities, cut):
            ordered = idx[np.argsort(-row[idx])]
            pairs = [(self.labels[i], float(row[i])) for i in ordered]
            out.append(
                IntentPrediction(
                    intent=pairs[0][0], confidence=pairs[0][1], top_k=pairs
                )
            )
        return out


def write_metadata(
    directory: Path,
    version: str,
    family: str,
    labels: List[str],
    train_config: Dict[str, Any],
    metrics: Dict[str, Any],
    extra: Dict[str, Any] = None,
) -> Path:
    """Write the model card the registry and the reports read back."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "family": family,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_labels": len(labels),
        "labels": labels,
        "train_config": train_config,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    path = directory / "metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_metadata(directory: Path) -> Dict[str, Any]:
    return json.loads((Path(directory) / "metadata.json").read_text(encoding="utf-8"))
