"""Loader for the triage taxonomy in ``data/taxonomy.json``.

The taxonomy maps the 77 BANKING77 intents onto 9 coarse triage categories, and
attaches the operational metadata (queue, SLA, base priority) that the routing
layer needs. It is the single source of truth: ``data/TAXONOMY.md`` is rendered
from it and ``tests/test_taxonomy.py`` fails if the two drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from src.config import TAXONOMY_PATH


@dataclass(frozen=True)
class Category:
    """A coarse triage category and how the routing layer should treat it."""

    key: str
    display_name: str
    queue: str
    sla_minutes: int
    base_priority: int
    security_critical: bool
    description: str


class Taxonomy:
    """Intent -> category mapping plus per-category routing metadata."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        self.schema_version = str(payload["schema_version"])
        self.source_label_set = str(payload["source_label_set"])

        raw_categories: Mapping[str, Mapping[str, object]] = payload["categories"]  # type: ignore[assignment]
        self.categories: Dict[str, Category] = {
            key: Category(
                key=key,
                display_name=str(spec["display_name"]),
                queue=str(spec["queue"]),
                sla_minutes=int(spec["sla_minutes"]),
                base_priority=int(spec["base_priority"]),
                security_critical=bool(spec["security_critical"]),
                description=str(spec["description"]),
            )
            for key, spec in raw_categories.items()
        }

        self.intent_to_category: Dict[str, str] = dict(
            payload["intent_to_category"]  # type: ignore[arg-type]
        )

        unknown = set(self.intent_to_category.values()) - set(self.categories)
        if unknown:
            raise ValueError(
                f"taxonomy.json maps intents to undefined categories: {sorted(unknown)}"
            )
        unused = set(self.categories) - set(self.intent_to_category.values())
        if unused:
            raise ValueError(
                f"taxonomy.json defines categories no intent maps to: {sorted(unused)}"
            )

    # -- lookups ---------------------------------------------------------

    @property
    def intents(self) -> List[str]:
        """All known intents, sorted. This is the model's label space order."""
        return sorted(self.intent_to_category)

    @property
    def category_keys(self) -> List[str]:
        """All category keys, ordered most to least urgent then alphabetically."""
        return sorted(
            self.categories,
            key=lambda k: (self.categories[k].base_priority, k),
        )

    def category_for_intent(self, intent: str) -> Category:
        try:
            return self.categories[self.intent_to_category[intent]]
        except KeyError as exc:
            raise KeyError(f"intent not present in taxonomy: {intent!r}") from exc

    def intents_in_category(self, category_key: str) -> List[str]:
        return sorted(
            intent
            for intent, cat in self.intent_to_category.items()
            if cat == category_key
        )

    def map_intents(self, intents: List[str]) -> List[str]:
        """Vectorised intent -> category-key mapping."""
        return [self.intent_to_category[i] for i in intents]


@lru_cache(maxsize=4)
def load_taxonomy(path: Optional[Path] = None) -> Taxonomy:
    """Load and cache the taxonomy. Pass ``path`` in tests to load a variant."""
    target = Path(path) if path is not None else TAXONOMY_PATH
    with open(target, "r", encoding="utf-8") as handle:
        return Taxonomy(json.load(handle))
