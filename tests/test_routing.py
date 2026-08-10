"""Routing rules and A/B assignment.

The first two tests here are the ones that matter operationally: they pin the
two invariants stated in `src/routing/rules.py`.
"""

from __future__ import annotations

import json

import pytest

from src.extraction.intent import IntentSignals
from src.extraction.sentiment import SentimentResult
from src.routing.ab import ABConfig, assign, bucket_for, observed_split
from src.routing.rules import REVIEW_QUEUE, RoutingPolicy, effective_sla, route


def signals(intent, category, confidence=0.9, **kwargs):
    defaults = {
        "margin": 0.5, "top_k": [(intent, confidence)], "alternative_categories": [],
        "speech_act": "statement", "urgency": 0.0, "urgency_markers": [],
        "is_ambiguous": False,
    }
    defaults.update(kwargs)
    return IntentSignals(
        intent=intent, confidence=confidence, category=category, **defaults
    )


def sentiment(compound=0.0):
    label = "negative" if compound <= -0.2 else "positive" if compound >= 0.05 else "neutral"
    return SentimentResult(
        compound=compound, label=label, positive=0.0, neutral=1.0, negative=0.0
    )


class TestRoutingInvariants:
    def test_priority_is_never_lowered_below_the_category_base(self, taxonomy):
        """Rules escalate; they must never de-escalate."""
        for key, category in taxonomy.categories.items():
            intent = taxonomy.intents_in_category(key)[0]
            decision = route(
                signals(intent, key), sentiment(0.9), taxonomy
            )
            assert decision.priority <= category.base_priority
            assert decision.base_priority == category.base_priority

    def test_security_critical_is_never_parked_in_manual_triage(self, taxonomy):
        """Low confidence must not delay a suspected fraud report."""
        decision = route(
            signals("lost_or_stolen_card", "identity_security", confidence=0.05),
            sentiment(-0.8), taxonomy,
        )
        assert decision.queue == "FRAUD_SECURITY"
        assert decision.needs_human_review is True
        assert decision.auto_routed is True
        assert decision.priority == 1


class TestRoutingRules:
    def test_base_case_uses_the_category_queue_and_priority(self, taxonomy):
        decision = route(signals("exchange_rate", "fees_fx"), sentiment(), taxonomy)
        assert decision.queue == "BILLING"
        assert decision.priority == 4
        assert decision.sla_minutes == 480
        assert decision.auto_routed is True
        assert decision.needs_human_review is False

    def test_strong_negative_sentiment_escalates_one_step(self, taxonomy):
        decision = route(signals("exchange_rate", "fees_fx"), sentiment(-0.8), taxonomy)
        assert decision.priority == 3
        assert decision.sla_minutes == 240
        assert any("negative sentiment" in r for r in decision.reasons)

    def test_escalation_is_capped(self, taxonomy):
        """Three escalating signals must not run priority off the end."""
        decision = route(
            signals("exchange_rate", "fees_fx", urgency=0.9, speech_act="complaint"),
            sentiment(-0.9), taxonomy,
        )
        assert decision.priority == 2  # base 4, capped at 2 steps
        assert decision.priority >= 1

    def test_low_confidence_parks_non_critical_records(self, taxonomy):
        decision = route(
            signals("exchange_rate", "fees_fx", confidence=0.10), sentiment(), taxonomy
        )
        assert decision.queue == REVIEW_QUEUE
        assert decision.auto_routed is False
        assert decision.needs_human_review is True

    def test_ambiguous_split_across_queues_needs_review_but_still_routes(self, taxonomy):
        decision = route(
            signals("exchange_rate", "fees_fx", confidence=0.8, margin=0.02,
                    is_ambiguous=True, alternative_categories=["payments_disputes"]),
            sentiment(), taxonomy,
        )
        assert decision.needs_human_review is True
        assert decision.auto_routed is True
        assert decision.queue == "BILLING"

    def test_threshold_is_configurable(self, taxonomy):
        lenient = RoutingPolicy(confidence_threshold=0.05)
        decision = route(
            signals("exchange_rate", "fees_fx", confidence=0.10), sentiment(),
            taxonomy, lenient,
        )
        assert decision.auto_routed is True


class TestPerVersionThresholds:
    """A 0.80 probability from a TF-IDF model and from a transformer are not
    the same claim, so they do not share an auto-route threshold."""

    @pytest.fixture
    def policy(self):
        return RoutingPolicy(
            confidence_threshold=0.75,
            version_thresholds={"linear-v1": 0.75, "transformer-v1": 0.95},
        )

    def test_same_confidence_routes_differently_per_version(self, taxonomy, policy):
        record = signals("exchange_rate", "fees_fx", confidence=0.80)
        lenient = route(record, sentiment(), taxonomy, policy, version="linear-v1")
        strict = route(record, sentiment(), taxonomy, policy, version="transformer-v1")
        assert lenient.auto_routed is True
        assert strict.auto_routed is False

    def test_unknown_version_falls_back_to_the_default(self, policy):
        assert policy.threshold_for("never-trained-v9") == 0.75

    def test_no_version_uses_the_default(self, policy):
        assert policy.threshold_for(None) == 0.75

    def test_known_version_uses_its_own(self, policy):
        assert policy.threshold_for("transformer-v1") == 0.95

    def test_the_reason_names_the_version(self, taxonomy, policy):
        decision = route(
            signals("exchange_rate", "fees_fx", confidence=0.80), sentiment(),
            taxonomy, policy, version="transformer-v1",
        )
        assert any("transformer-v1" in r for r in decision.reasons)

    def test_loads_from_file(self, tmp_path):
        path = tmp_path / "routing_policy.json"
        path.write_text(json.dumps({
            "confidence_threshold": 0.6,
            "version_thresholds": {"a-v1": 0.9},
            "max_escalation_steps": 1,
        }), encoding="utf-8")
        policy = RoutingPolicy.from_file(path)
        assert policy.confidence_threshold == 0.6
        assert policy.threshold_for("a-v1") == 0.9
        assert policy.max_escalation_steps == 1

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        policy = RoutingPolicy.from_file(tmp_path / "absent.json")
        assert policy.confidence_threshold == RoutingPolicy().confidence_threshold

    def test_committed_config_covers_every_registered_version(self):
        """The shipped config must not leave a servable version on the fallback."""
        from src.config import ROUTING_POLICY_PATH
        from src.routing.ab import load_ab_config

        policy = RoutingPolicy.from_file(ROUTING_POLICY_PATH)
        for version in load_ab_config().versions:
            assert version in policy.version_thresholds, (
                f"{version} is weighted in the A/B config but has no tuned "
                "threshold; run tune_confidence_threshold.py --write-config"
            )

    def test_reasons_are_always_populated(self, taxonomy):
        decision = route(signals("exchange_rate", "fees_fx"), sentiment(), taxonomy)
        assert decision.reasons

    @pytest.mark.parametrize("base_sla,base,priority,expected", [
        (480, 4, 4, 480), (480, 4, 3, 240), (480, 4, 2, 120), (15, 1, 1, 15),
    ])
    def test_effective_sla_halves_per_step(self, base_sla, base, priority, expected):
        assert effective_sla(base_sla, base, priority) == expected

    def test_effective_sla_has_a_floor(self):
        assert effective_sla(10, 4, 1) == 5


class TestABAssignment:
    def test_is_deterministic(self, ab_payload):
        config = ABConfig(ab_payload)
        first = assign("customer-1", config).version
        assert all(assign("customer-1", config).version == first for _ in range(50))

    def test_salt_change_reshuffles(self, ab_payload):
        a = ABConfig(dict(ab_payload, salt="one"))
        b = ABConfig(dict(ab_payload, salt="two"))
        units = [f"c{i}" for i in range(500)]
        moved = sum(1 for u in units if assign(u, a).version != assign(u, b).version)
        assert moved > 100, "a new salt should re-randomise assignment"

    def test_split_is_close_to_the_configured_weights(self, ab_payload):
        config = ABConfig(ab_payload)
        split = observed_split([f"customer-{i}" for i in range(20_000)], config)
        for version, weight in config.weights.items():
            assert abs(split[version][1] - weight) < 0.02

    def test_uneven_weights_are_respected(self, ab_payload):
        config = ABConfig(dict(
            ab_payload, weights={"control-v1": 0.9, "treatment-v1": 0.1}
        ))
        split = observed_split([f"c{i}" for i in range(20_000)], config)
        assert abs(split["treatment-v1"][1] - 0.1) < 0.02

    def test_buckets_are_inside_the_unit_interval(self):
        values = [bucket_for(f"u{i}", "exp", "salt") for i in range(1000)]
        assert all(0.0 <= v < 1.0 for v in values)

    def test_zero_weight_version_receives_no_traffic(self, ab_payload):
        config = ABConfig(dict(
            ab_payload,
            weights={"control-v1": 1.0, "treatment-v1": 0.0},
        ))
        split = observed_split([f"c{i}" for i in range(2000)], config)
        assert split["treatment-v1"][0] == 0
        assert "treatment-v1" not in config.active_versions

    def test_disabled_experiment_serves_control(self, ab_payload):
        config = ABConfig(dict(ab_payload, enabled=False))
        assert assign("anyone", config).version == "control-v1"
        assert config.active_versions == ["control-v1"]

    def test_force_version_overrides(self, ab_payload):
        config = ABConfig(ab_payload)
        assert assign("x", config, force_version="treatment-v1").version == "treatment-v1"

    def test_force_unknown_version_raises(self, ab_payload):
        with pytest.raises(KeyError):
            assign("x", ABConfig(ab_payload), force_version="nope")

    def test_weights_must_sum_to_one(self, ab_payload):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ABConfig(dict(ab_payload, weights={"control-v1": 0.3, "treatment-v1": 0.3}))

    def test_negative_weights_rejected(self, ab_payload):
        with pytest.raises(ValueError, match="negative"):
            ABConfig(dict(ab_payload, weights={"control-v1": 1.2, "treatment-v1": -0.2}))

    def test_control_must_be_weighted(self, ab_payload):
        with pytest.raises(ValueError, match="control_version"):
            ABConfig(dict(ab_payload, control_version="ghost-v1"))
