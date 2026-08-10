"""A/B assignment between model versions.

Assignment is a pure function of the unit identifier: hash it, map the hash
into [0, 1), and walk the cumulative weights. That buys three properties worth
having.

* **Sticky.** The same customer always sees the same model version for the life
  of the experiment, so a conversation does not switch models mid-thread.
* **Stateless.** No assignment table, no database, no coordination between
  service replicas. Any replica computes the same answer.
* **Reproducible.** The offline analysis in `experiments/run_ab_experiment.py`
  recomputes the exact assignments the service would have made.

Changing the salt reshuffles everyone, which is what you want when starting a
genuinely new experiment and what you must *not* do mid-experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import AB_CONFIG_PATH

_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Assignment:
    version: str
    bucket: float
    reason: str


class ABConfig:
    """Traffic weights over model versions, loaded from JSON."""

    def __init__(self, payload: Dict[str, object]) -> None:
        self.experiment_id = str(payload["experiment_id"])
        self.enabled = bool(payload.get("enabled", True))
        self.salt = str(payload.get("salt", ""))
        self.control_version = str(payload["control_version"])
        self.unit_field = str(payload.get("unit_field", "client_id"))
        self.description = str(payload.get("description", ""))

        weights: Dict[str, float] = {
            str(k): float(v) for k, v in dict(payload["weights"]).items()  # type: ignore[arg-type]
        }
        self.weights = self._validate(weights)

        if self.control_version not in self.weights:
            raise ValueError(
                f"control_version {self.control_version!r} is not among the "
                f"weighted versions {sorted(self.weights)}"
            )

    @staticmethod
    def _validate(weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            raise ValueError("A/B config must weight at least one version")
        if any(w < 0 for w in weights.values()):
            raise ValueError(f"negative A/B weight in {weights}")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("A/B weights sum to zero")
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(
                f"A/B weights must sum to 1.0, got {total:.6f}. "
                "Normalising silently would hide a config mistake."
            )
        # Sorted so the cumulative walk is deterministic across processes.
        return dict(sorted(weights.items()))

    @property
    def versions(self) -> List[str]:
        return list(self.weights)

    @property
    def active_versions(self) -> List[str]:
        """Versions that can actually receive traffic under this config."""
        if not self.enabled:
            return [self.control_version]
        return [v for v, w in self.weights.items() if w > 0]

    def as_dict(self) -> Dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "enabled": self.enabled,
            "control_version": self.control_version,
            "unit_field": self.unit_field,
            "description": self.description,
            "weights": self.weights,
        }


def load_ab_config(path: Optional[Path] = None) -> ABConfig:
    target = Path(path) if path is not None else AB_CONFIG_PATH
    with open(target, "r", encoding="utf-8") as handle:
        return ABConfig(json.load(handle))


def bucket_for(unit_id: str, experiment_id: str, salt: str) -> float:
    """Map a unit identifier into [0, 1) uniformly and deterministically."""
    digest = hashlib.sha256(f"{experiment_id}:{salt}:{unit_id}".encode("utf-8")).digest()
    # 8 bytes is 64 bits of resolution; far more than enough for traffic splits.
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign(
    unit_id: str, config: ABConfig, force_version: Optional[str] = None
) -> Assignment:
    """Pick the model version this unit should be served by."""
    if force_version:
        if force_version not in config.weights:
            raise KeyError(
                f"forced version {force_version!r} is not in the experiment; "
                f"known versions: {sorted(config.weights)}"
            )
        return Assignment(force_version, -1.0, "forced by caller")

    if not config.enabled:
        return Assignment(config.control_version, -1.0, "experiment disabled")

    bucket = bucket_for(unit_id, config.experiment_id, config.salt)
    cumulative = 0.0
    for version, weight in config.weights.items():
        cumulative += weight
        if bucket < cumulative:
            return Assignment(version, round(bucket, 6), f"hash bucket {bucket:.4f}")

    # Only reachable through floating-point drift at the very top of the range.
    return Assignment(
        config.control_version, round(bucket, 6), "fell through to control"
    )


def observed_split(unit_ids: List[str], config: ABConfig) -> Dict[str, Tuple[int, float]]:
    """Realised counts and shares — used to check the split is not skewed."""
    counts: Dict[str, int] = {version: 0 for version in config.weights}
    for unit_id in unit_ids:
        counts[assign(unit_id, config).version] += 1
    total = max(len(unit_ids), 1)
    return {v: (n, round(n / total, 4)) for v, n in counts.items()}
