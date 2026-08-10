"""Load a persisted scikit-learn baseline and serve predictions from it."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from src.models.base import IntentClassifier, read_metadata


class BaselineClassifier(IntentClassifier):
    """Wraps the pickled ``Pipeline`` behind the common classifier interface."""

    def __init__(self, directory: Path) -> None:
        directory = Path(directory)
        metadata = read_metadata(directory)
        self.version = metadata["version"]
        self.family = metadata["family"]
        self.labels = list(metadata["labels"])
        self.metadata = metadata
        self.pipeline = joblib.load(directory / "model.joblib")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, len(self.labels)), dtype=np.float64)
        return np.asarray(self.pipeline.predict_proba(list(texts)), dtype=np.float64)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BaselineClassifier(version={self.version!r}, labels={len(self.labels)})"
