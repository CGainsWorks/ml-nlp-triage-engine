"""The taxonomy is the contract between the model, routing and the docs."""

from __future__ import annotations

import json

import pytest

from src.config import TAXONOMY_MD_PATH, TAXONOMY_PATH
from src.etl.render_taxonomy import render
from src.taxonomy import Taxonomy, load_taxonomy


def test_every_intent_maps_to_a_defined_category(taxonomy):
    for intent, category in taxonomy.intent_to_category.items():
        assert category in taxonomy.categories, f"{intent} -> unknown {category}"


def test_label_space_is_the_expected_size(taxonomy):
    assert len(taxonomy.intents) == 77
    assert len(taxonomy.categories) == 9


def test_intents_partition_the_categories(taxonomy):
    """Every intent appears in exactly one category and nothing is orphaned."""
    seen = [i for key in taxonomy.category_keys for i in taxonomy.intents_in_category(key)]
    assert sorted(seen) == taxonomy.intents
    assert len(seen) == len(set(seen))


def test_category_keys_are_ordered_by_urgency(taxonomy):
    priorities = [taxonomy.categories[k].base_priority for k in taxonomy.category_keys]
    assert priorities == sorted(priorities)


def test_exactly_one_security_critical_category(taxonomy):
    critical = [c for c in taxonomy.categories.values() if c.security_critical]
    assert [c.key for c in critical] == ["identity_security"]
    assert critical[0].base_priority == 1


def test_sla_increases_as_priority_falls(taxonomy):
    """A P1 category must not promise a slower response than a P4 one."""
    by_priority = {}
    for category in taxonomy.categories.values():
        by_priority.setdefault(category.base_priority, []).append(category.sla_minutes)
    worst = {p: max(v) for p, v in by_priority.items()}
    for higher, lower in zip(sorted(worst), sorted(worst)[1:]):
        assert worst[higher] <= worst[lower]


def test_markdown_is_regenerated_from_json():
    """`data/TAXONOMY.md` must not drift from `data/taxonomy.json`."""
    assert TAXONOMY_MD_PATH.exists(), "run: python -m src.etl.render_taxonomy"
    assert TAXONOMY_MD_PATH.read_text(encoding="utf-8") == render(load_taxonomy()), (
        "data/TAXONOMY.md is stale. Run: python -m src.etl.render_taxonomy"
    )


def test_unknown_intent_raises(taxonomy):
    with pytest.raises(KeyError):
        taxonomy.category_for_intent("not_a_real_intent")


def test_rejects_category_with_no_intents():
    payload = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    payload["categories"]["orphan"] = {
        "display_name": "Orphan", "queue": "NOWHERE", "sla_minutes": 60,
        "base_priority": 4, "security_critical": False, "description": "unused",
    }
    with pytest.raises(ValueError, match="no intent maps to"):
        Taxonomy(payload)


def test_rejects_intent_pointing_at_missing_category():
    payload = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    payload["intent_to_category"]["card_arrival"] = "does_not_exist"
    with pytest.raises(ValueError, match="undefined categories"):
        Taxonomy(payload)
