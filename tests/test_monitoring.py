"""Prediction store, insight aggregation, drift and metrics."""

from __future__ import annotations

import time

import pytest

from src.etl.stats import CHAR_BINS, histogram
from src.monitoring import insights
from src.monitoring.drift import DriftMonitor, band, psi
from src.monitoring.metrics import TriageMetrics
from src.monitoring.store import PredictionRecord, PredictionStore


def record(version="v1", **kwargs):
    defaults = {
        "timestamp": time.time(), "version": version, "intent": "card_arrival",
        "category": "cards_lifecycle", "queue": "CARD_OPS", "priority": 3,
        "confidence": 0.8, "margin": 0.4, "sentiment_compound": -0.1,
        "sentiment_label": "neutral", "urgency": 0.0, "speech_act": "question",
        "needs_human_review": False, "auto_routed": True, "escalate": False,
        "n_chars": 40, "n_words": 8, "latency_ms": 5.0,
    }
    defaults.update(kwargs)
    return PredictionRecord(**defaults)


class TestStore:
    def test_evicts_oldest_beyond_capacity(self):
        store = PredictionStore(capacity=10)
        for i in range(25):
            store.append(record(intent=f"i{i}"))
        assert len(store) == 10
        assert store.stats["total_seen"] == 25
        assert store.stats["evicted"] == 15

    def test_filters_by_version(self):
        store = PredictionStore()
        store.append(record("a"))
        store.append(record("b"))
        assert len(store.snapshot(version="a")) == 1
        assert store.versions() == ["a", "b"]

    def test_filters_by_time_window(self):
        store = PredictionStore()
        store.append(record(timestamp=time.time() - 3600))
        store.append(record())
        assert len(store.snapshot(since_seconds=60)) == 1

    def test_reports_that_it_is_not_durable(self):
        """The limitation must be visible in the API response, not just the docs."""
        assert PredictionStore().stats["durable"] is False


class TestInsights:
    def test_empty_window_does_not_explode(self):
        assert insights.summarise([])["n_records"] == 0

    def test_breaks_down_by_version(self):
        records = [record("a") for _ in range(3)] + [record("b") for _ in range(2)]
        report = insights.report(records)
        assert report["versions_seen"] == ["a", "b"]
        assert report["by_version"]["a"]["n_records"] == 3
        assert report["overall"]["n_records"] == 5

    def test_rates_are_computed(self):
        records = [
            record(needs_human_review=True, auto_routed=False),
            record(escalate=True),
            record(),
            record(),
        ]
        summary = insights.summarise(records)
        assert summary["auto_route_rate"] == 0.75
        assert summary["escalation_rate"] == 0.25
        assert summary["confidence"]["below_threshold_share"] == 0.25

    def test_accuracy_is_none_without_labels(self):
        assert insights.summarise([record(), record()])["accuracy"] is None

    def test_accuracy_is_computed_for_replayed_labels(self):
        records = [
            record(intent="card_arrival", true_intent="card_arrival"),
            record(intent="card_arrival", true_intent="exchange_rate"),
        ]
        accuracy = insights.summarise(records)["accuracy"]
        assert accuracy["n_labelled"] == 2
        assert accuracy["intent_accuracy"] == 0.5

    def test_sentiment_carries_its_caveat(self):
        summary = insights.summarise([record()])
        assert "uncalibrated" in summary["sentiment"]["caveat"]


class TestDrift:
    def test_identical_distributions_have_zero_psi(self):
        reference = [0.2, 0.3, 0.5]
        assert psi(reference, reference) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_is_flagged(self):
        assert psi([0.8, 0.15, 0.05], [0.05, 0.15, 0.8]) > 0.25

    def test_psi_is_symmetric_in_magnitude_sign(self):
        a, b = [0.6, 0.4], [0.4, 0.6]
        assert psi(a, b) == pytest.approx(psi(b, a), rel=1e-9)

    def test_empty_bins_do_not_produce_infinity(self):
        import numpy as np

        assert np.isfinite(psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]))

    def test_mismatched_bin_counts_raise(self):
        with pytest.raises(ValueError, match="bin mismatch"):
            psi([0.5, 0.5], [0.3, 0.3, 0.4])

    @pytest.mark.parametrize("value,expected", [
        (0.05, "stable"), (0.15, "moderate"), (0.4, "significant"),
    ])
    def test_bands(self, value, expected):
        assert band(value) == expected

    def test_monitor_without_references_returns_nothing(self):
        monitor = DriftMonitor()
        assert monitor.evaluate("v1", [record()]) == []

    def test_monitor_detects_a_length_shift(self):
        reference = {
            "text_length": {
                "chars": {"proportions": histogram([30] * 100, CHAR_BINS)},
                "words": {"proportions": histogram([6] * 100, CHAR_BINS)},
            }
        }
        monitor = DriftMonitor(input_reference=reference)
        long_records = [record(n_chars=400, n_words=90) for _ in range(300)]
        readings = {r.feature: r for r in monitor.evaluate("v1", long_records)}
        assert readings["text_chars"].psi > 0.25
        assert readings["text_chars"].band == "significant"

    def test_small_samples_are_marked_as_noise(self):
        reference = {
            "text_length": {"chars": {"proportions": histogram([30] * 100, CHAR_BINS)}}
        }
        readings = DriftMonitor(reference).evaluate("v1", [record()])
        assert "noise" in readings[0].note


class TestMetrics:
    def test_series_are_labelled_by_version(self):
        metrics = TriageMetrics()
        metrics.observe_prediction(
            version="v1", category="fees_fx", queue="BILLING", priority=4,
            confidence=0.9, sentiment_label="neutral", low_confidence=False,
            escalated=False, review_reason=None,
        )
        rendered = metrics.render().decode()
        assert 'version="v1"' in rendered
        assert "triage_requests_total" in rendered

    def test_two_instances_do_not_collide(self):
        """Separate registries, so tests and the app can coexist in one process."""
        TriageMetrics()
        TriageMetrics()

    def test_ab_weights_are_exported(self):
        metrics = TriageMetrics()
        metrics.set_ab_weights("exp-1", {"a": 0.7, "b": 0.3})
        rendered = metrics.render().decode()
        assert "triage_ab_traffic_weight" in rendered
        assert 'experiment="exp-1"' in rendered
