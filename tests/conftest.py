"""Shared fixtures.

The unit tests must not need trained models — a suite that only runs after a
40-minute fine-tune is a suite nobody runs. So the service, routing and API
tests are driven by :class:`StubClassifier`, which satisfies the same
:class:`~src.models.base.IntentClassifier` interface over the real 77-label
space and returns probabilities the test chooses.

Tests that genuinely need a trained artefact are marked ``integration`` and skip
themselves when ``artifacts/models`` is empty.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest

from src.config import MODELS_DIR
from src.models.base import IntentClassifier
from src.taxonomy import load_taxonomy


class StubClassifier(IntentClassifier):
    """Deterministic classifier: keyword -> intent, with a chosen confidence."""

    def __init__(
        self,
        version: str = "stub-v1",
        family: str = "baseline",
        rules: Optional[Dict[str, str]] = None,
        confidence: float = 0.9,
        fallback_confidence: float = 0.10,
    ) -> None:
        self.version = version
        self.family = family
        self.labels = load_taxonomy().intents
        self.rules = rules or {}
        self.confidence = confidence
        self.fallback_confidence = fallback_confidence

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        n_labels = len(self.labels)
        out = np.zeros((len(texts), n_labels), dtype=np.float64)
        for row, text in enumerate(texts):
            lowered = text.lower()
            target = next(
                (intent for keyword, intent in self.rules.items() if keyword in lowered),
                None,
            )
            if target is None:
                # Spread mass so the top-1 is deliberately below any sensible
                # auto-route threshold: this is the "model is unsure" case.
                out[row, :] = (1.0 - self.fallback_confidence) / (n_labels - 1)
                out[row, row % n_labels] = self.fallback_confidence
            else:
                index = self.labels.index(target)
                out[row, :] = (1.0 - self.confidence) / (n_labels - 1)
                out[row, index] = self.confidence
        return out


class StubRegistry:
    """Minimal stand-in for :class:`~src.models.registry.ModelRegistry`."""

    def __init__(self, models: Dict[str, StubClassifier]) -> None:
        self._models = models

    @property
    def versions(self) -> List[str]:
        return sorted(self._models)

    @property
    def loaded_versions(self) -> List[str]:
        return self.versions

    def metadata(self, version: str) -> Dict[str, object]:
        model = self._models[version]
        return {
            "version": version,
            "family": model.family,
            "n_labels": len(model.labels),
            "created_utc": "2026-01-01T00:00:00+00:00",
            "train_config": {"algorithm": "stub"},
            "metrics": {},
            "path": f"/stub/{version}",
        }

    def get(self, version: str) -> StubClassifier:
        return self._models[version]

    def preload(self, versions: List[str]) -> None:
        for version in versions:
            self.get(version)

    def describe(self) -> List[Dict[str, object]]:
        return [
            {
                "version": version,
                "family": self._models[version].family,
                "created_utc": "2026-01-01T00:00:00+00:00",
                "n_labels": len(self._models[version].labels),
                "algorithm": "stub",
                "loaded": True,
                "test_intent_macro_f1": None,
                "test_category_macro_f1": None,
            }
            for version in self.versions
        ]


@pytest.fixture(scope="session")
def taxonomy():
    return load_taxonomy()


@pytest.fixture
def stub_rules() -> Dict[str, str]:
    return {
        "stolen": "lost_or_stolen_card",
        "exchange rate": "exchange_rate",
        "not arrived": "card_arrival",
        "charged twice": "transaction_charged_twice",
    }


@pytest.fixture
def stub_registry(stub_rules) -> StubRegistry:
    return StubRegistry({
        "control-v1": StubClassifier("control-v1", "baseline", stub_rules, 0.9),
        "treatment-v1": StubClassifier("treatment-v1", "transformer", stub_rules, 0.8),
    })


@pytest.fixture
def ab_payload() -> Dict[str, object]:
    return {
        "experiment_id": "unit-test",
        "enabled": True,
        "salt": "t1",
        "unit_field": "client_id",
        "control_version": "control-v1",
        "weights": {"control-v1": 0.5, "treatment-v1": 0.5},
    }


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs trained artefacts under artifacts/models"
    )


def has_trained_models() -> bool:
    return MODELS_DIR.exists() and any(
        (d / "metadata.json").exists() for d in MODELS_DIR.iterdir() if d.is_dir()
    )


requires_models = pytest.mark.skipif(
    not has_trained_models(),
    reason="no trained models; run `make train` first",
)
