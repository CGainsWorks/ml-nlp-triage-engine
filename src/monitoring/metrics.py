"""Prometheus metrics, labelled by model version throughout.

Every counter and histogram carries a ``version`` label. That is the whole
point: with an A/B split running, an aggregate latency or error rate is close
to useless, because it is a weighted average of two different models and moves
when the traffic weights move rather than when either model changes.

Metrics live on their own :class:`CollectorRegistry` rather than the process
default, so tests can build a fresh set without tripping duplicate-registration
errors.
"""

from __future__ import annotations

from typing import Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

__all__ = ["TriageMetrics", "CONTENT_TYPE_LATEST"]

_CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class TriageMetrics:
    """All service metrics, bound to one registry."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.requests = Counter(
            "triage_requests_total",
            "Triage decisions served.",
            ["version", "category", "queue", "priority"],
            registry=self.registry,
        )
        self.records = Counter(
            "triage_records_total",
            "Individual records classified, regardless of batching.",
            ["version"],
            registry=self.registry,
        )
        self.low_confidence = Counter(
            "triage_low_confidence_total",
            "Records whose confidence fell below the auto-route threshold.",
            ["version"],
            registry=self.registry,
        )
        self.human_review = Counter(
            "triage_human_review_total",
            "Records flagged for human review, by the reason that triggered it.",
            ["version", "reason"],
            registry=self.registry,
        )
        self.escalations = Counter(
            "triage_escalations_total",
            "Records escalated to a security-critical queue.",
            ["version"],
            registry=self.registry,
        )
        self.errors = Counter(
            "triage_errors_total",
            "Failures while serving, by kind.",
            ["version", "kind"],
            registry=self.registry,
        )
        self.sentiment = Counter(
            "triage_sentiment_total",
            "Sentiment labels observed. Uncalibrated lexicon output; see "
            "src/extraction/sentiment.py before drawing conclusions from it.",
            ["version", "label"],
            registry=self.registry,
        )

        self.latency = Histogram(
            "triage_latency_seconds",
            "Wall-clock time to produce a triage decision.",
            ["version", "stage"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.confidence = Histogram(
            "triage_confidence",
            "Top-1 probability of the served model.",
            ["version"],
            buckets=_CONFIDENCE_BUCKETS,
            registry=self.registry,
        )

        self.ab_weight = Gauge(
            "triage_ab_traffic_weight",
            "Configured share of traffic for each model version.",
            ["version", "experiment"],
            registry=self.registry,
        )
        self.drift_psi = Gauge(
            "triage_drift_psi",
            "Population stability index of live traffic against the training "
            "reference. Above 0.25 is a significant shift.",
            ["version", "feature"],
            registry=self.registry,
        )
        self.model_up = Gauge(
            "triage_model_loaded",
            "1 if the model version is loaded and servable.",
            ["version", "family"],
            registry=self.registry,
        )
        self.window_size = Gauge(
            "triage_monitoring_window_records",
            "Records currently held in the in-memory monitoring window.",
            registry=self.registry,
        )

    # -- recording -------------------------------------------------------

    def observe_prediction(
        self,
        version: str,
        category: str,
        queue: str,
        priority: int,
        confidence: float,
        sentiment_label: str,
        low_confidence: bool,
        escalated: bool,
        review_reason: Optional[str],
    ) -> None:
        self.requests.labels(version, category, queue, f"P{priority}").inc()
        self.records.labels(version).inc()
        self.confidence.labels(version).observe(confidence)
        self.sentiment.labels(version, sentiment_label).inc()
        if low_confidence:
            self.low_confidence.labels(version).inc()
        if escalated:
            self.escalations.labels(version).inc()
        if review_reason:
            self.human_review.labels(version, review_reason).inc()

    def observe_latency(self, version: str, stage: str, seconds: float) -> None:
        self.latency.labels(version, stage).observe(seconds)

    def observe_error(self, version: str, kind: str) -> None:
        self.errors.labels(version, kind).inc()

    def set_ab_weights(self, experiment: str, weights: Dict[str, float]) -> None:
        for version, weight in weights.items():
            self.ab_weight.labels(version, experiment).set(weight)

    def set_drift(self, version: str, feature: str, psi: float) -> None:
        self.drift_psi.labels(version, feature).set(psi)

    def set_model_loaded(self, version: str, family: str, loaded: bool) -> None:
        self.model_up.labels(version, family).set(1 if loaded else 0)

    def render(self) -> bytes:
        return generate_latest(self.registry)
