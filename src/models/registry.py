"""Discovers and loads trained model versions from disk.

A version is any directory under ``artifacts/models/`` containing a
``metadata.json``. The ``family`` field in that file decides which loader is
used, which is what lets the service hold a TF-IDF pipeline and a fine-tuned
transformer side by side and route between them without caring which is which.

Loading is lazy and cached: a version registered with zero A/B weight costs a
directory listing, not 250 MB of resident memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.config import MODELS_DIR
from src.models.base import IntentClassifier, read_metadata

LOGGER = logging.getLogger(__name__)


class UnknownVersionError(KeyError):
    pass


def _load_for_family(family: str, directory: Path) -> IntentClassifier:
    if family == "baseline":
        from src.models.baseline.predict import BaselineClassifier

        return BaselineClassifier(directory)
    if family == "transformer":
        from src.models.transformer.predict import TransformerClassifier

        return TransformerClassifier(directory)
    raise ValueError(f"{directory}: unknown model family {family!r}")


class ModelRegistry:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self._metadata: Dict[str, Dict[str, object]] = {}
        self._loaded: Dict[str, IntentClassifier] = {}
        self.refresh()

    def refresh(self) -> None:
        """Re-scan the models directory. Does not evict anything already loaded."""
        self._metadata = {}
        if not self.models_dir.exists():
            LOGGER.warning("models directory %s does not exist", self.models_dir)
            return
        for directory in sorted(self.models_dir.iterdir()):
            if not (directory / "metadata.json").exists():
                continue
            try:
                metadata = read_metadata(directory)
            except Exception as exc:  # noqa: BLE001 - one bad dir must not hide the rest
                LOGGER.error("skipping %s: %s", directory, exc)
                continue
            version = str(metadata.get("version", directory.name))
            metadata["path"] = str(directory)
            self._metadata[version] = metadata
        LOGGER.info("registry sees %d version(s): %s",
                    len(self._metadata), sorted(self._metadata))

    # -- access ----------------------------------------------------------

    @property
    def versions(self) -> List[str]:
        return sorted(self._metadata)

    @property
    def loaded_versions(self) -> List[str]:
        return sorted(self._loaded)

    def metadata(self, version: str) -> Dict[str, object]:
        try:
            return self._metadata[version]
        except KeyError as exc:
            raise UnknownVersionError(
                f"unknown model version {version!r}; available: {self.versions}"
            ) from exc

    def get(self, version: str) -> IntentClassifier:
        if version in self._loaded:
            return self._loaded[version]
        metadata = self.metadata(version)
        LOGGER.info("loading %s from %s", version, metadata["path"])
        model = _load_for_family(str(metadata["family"]), Path(str(metadata["path"])))
        self._loaded[version] = model
        return model

    def preload(self, versions: List[str]) -> None:
        """Load ahead of first request so no user pays the cold-start cost.

        Loading is not enough on its own. Torch defers a good deal of kernel
        setup to the first forward pass, which made the first transformer
        request take 5.4 seconds against roughly 50 ms for every one after it.
        So each model is also run once on a throwaway input.
        """
        for version in versions:
            try:
                model = self.get(version)
                model.predict_proba(["warm up the inference path"])
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("failed to preload %s: %s", version, exc)

    def describe(self) -> List[Dict[str, object]]:
        out = []
        for version in self.versions:
            metadata = self._metadata[version]
            test_metrics = (metadata.get("metrics") or {}).get("test") or {}
            out.append({
                "version": version,
                "family": metadata.get("family"),
                "created_utc": metadata.get("created_utc"),
                "n_labels": metadata.get("n_labels"),
                "algorithm": (metadata.get("train_config") or {}).get("algorithm"),
                "loaded": version in self._loaded,
                "test_intent_macro_f1": (test_metrics.get("intent") or {}).get("macro_f1"),
                "test_category_macro_f1": (test_metrics.get("category") or {}).get("macro_f1"),
            })
        return out


_DEFAULT: Optional[ModelRegistry] = None


def default_registry(models_dir: Path = MODELS_DIR) -> ModelRegistry:
    """Process-wide registry, built on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ModelRegistry(models_dir)
    return _DEFAULT


def reset_default_registry() -> None:
    """Test hook: drop the cached registry so a new models dir takes effect."""
    global _DEFAULT
    _DEFAULT = None
