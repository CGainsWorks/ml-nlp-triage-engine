"""The triage pipeline, independent of HTTP.

One record's journey: clean -> assign a model version -> classify -> extract
sentiment and intent shape -> apply routing rules -> record metrics. Keeping it
out of the FastAPI layer means the offline A/B experiment replays traffic
through exactly the same path the service uses, so its numbers describe the
service rather than a re-implementation of it.

Batching is by version, not by arrival order. With a 50/50 split, a batch of
100 records becomes two model calls of roughly 50 rather than 100 calls of one,
which matters a great deal for the transformer and not at all for TF-IDF.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.etl.clean import clean_text
from src.extraction.intent import extract
from src.extraction.sentiment import score as score_sentiment
from src.models.registry import ModelRegistry
from src.monitoring.drift import DriftMonitor
from src.monitoring.metrics import TriageMetrics
from src.monitoring.store import PredictionRecord, PredictionStore
from src.routing.ab import ABConfig, Assignment, assign
from src.routing.rules import RoutingPolicy, route
from src.taxonomy import Taxonomy, load_taxonomy

LOGGER = logging.getLogger(__name__)


@dataclass
class TriageInput:
    text: str
    client_id: Optional[str] = None
    request_id: Optional[str] = None
    true_intent: Optional[str] = None

    @property
    def unit_id(self) -> Optional[str]:
        return self.client_id or self.request_id


class TriageService:
    def __init__(
        self,
        registry: ModelRegistry,
        ab_config: ABConfig,
        taxonomy: Optional[Taxonomy] = None,
        policy: Optional[RoutingPolicy] = None,
        metrics: Optional[TriageMetrics] = None,
        store: Optional[PredictionStore] = None,
        drift: Optional[DriftMonitor] = None,
    ) -> None:
        self.registry = registry
        self.ab_config = ab_config
        self.taxonomy = taxonomy or load_taxonomy()
        self.policy = policy or RoutingPolicy()
        self.metrics = metrics or TriageMetrics()
        # `store or PredictionStore()` would be a bug: PredictionStore defines
        # __len__, so a freshly built empty store is falsy and the caller's
        # instance would be silently thrown away. The caller then watches a
        # store that never fills up. Compare against None explicitly.
        self.store = store if store is not None else PredictionStore()
        self.drift = drift if drift is not None else DriftMonitor()

        self.metrics.set_ab_weights(ab_config.experiment_id, ab_config.weights)
        for version in registry.versions:
            self.metrics.set_model_loaded(
                version,
                str(registry.metadata(version).get("family", "unknown")),
                version in registry.loaded_versions,
            )

    # -- A/B -------------------------------------------------------------

    def _assign(self, item: TriageInput, force_version: Optional[str]) -> Assignment:
        if force_version:
            return assign("", self.ab_config, force_version=force_version)
        if not item.unit_id:
            # No stable identifier means no stable bucket. Serving the control
            # version is the conservative choice: an unidentified caller never
            # lands in the treatment arm and so never pollutes the experiment.
            return Assignment(
                self.ab_config.control_version, -1.0,
                "no unit identifier; served by control",
            )
        return assign(item.unit_id, self.ab_config)

    def set_weights(
        self, weights: Dict[str, float], enabled: Optional[bool] = None
    ) -> ABConfig:
        payload = self.ab_config.as_dict()
        payload["weights"] = weights
        payload["salt"] = self.ab_config.salt
        if enabled is not None:
            payload["enabled"] = enabled
        updated = ABConfig(payload)

        unknown = set(updated.weights) - set(self.registry.versions)
        if unknown:
            raise ValueError(
                f"cannot weight versions that are not registered: {sorted(unknown)}; "
                f"registered: {self.registry.versions}"
            )

        self.ab_config = updated
        self.metrics.set_ab_weights(updated.experiment_id, updated.weights)
        self.registry.preload(updated.active_versions)
        LOGGER.info("A/B weights updated to %s", updated.weights)
        return updated

    # -- serving ---------------------------------------------------------

    def triage_many(
        self, items: List[TriageInput], force_version: Optional[str] = None
    ) -> List[Dict[str, object]]:
        if not items:
            return []

        started = time.perf_counter()
        cleaned = [clean_text(item.text) for item in items]
        assignments = [self._assign(item, force_version) for item in items]

        by_version: Dict[str, List[int]] = {}
        for i, assignment in enumerate(assignments):
            by_version.setdefault(assignment.version, []).append(i)

        predictions: Dict[int, object] = {}
        model_seconds: Dict[str, float] = {}
        for version, indices in by_version.items():
            model = self.registry.get(version)
            call_start = time.perf_counter()
            batch = model.predict([cleaned[i] for i in indices])
            elapsed = time.perf_counter() - call_start
            model_seconds[version] = elapsed / max(len(indices), 1)
            self.metrics.observe_latency(version, "model", elapsed)
            for i, prediction in zip(indices, batch):
                predictions[i] = prediction

        results: List[Dict[str, object]] = []
        for i, (item, assignment) in enumerate(zip(items, assignments)):
            item_start = time.perf_counter()
            text = cleaned[i]
            prediction = predictions[i]

            signals = extract(text, prediction, self.taxonomy)
            sentiment = score_sentiment(text)
            decision = route(
                signals, sentiment, self.taxonomy, self.policy,
                version=assignment.version,
            )

            post_seconds = time.perf_counter() - item_start
            latency_ms = 1000 * (model_seconds[assignment.version] + post_seconds)
            self.metrics.observe_latency(assignment.version, "post_model", post_seconds)

            # Per-version threshold: see RoutingPolicy.version_thresholds.
            low_confidence = signals.confidence < self.policy.threshold_for(
                assignment.version
            )
            review_reason = None
            if decision.needs_human_review:
                review_reason = "low_confidence" if low_confidence else "ambiguous_queue_split"

            self.metrics.observe_prediction(
                version=assignment.version,
                category=decision.category,
                queue=decision.queue,
                priority=decision.priority,
                confidence=signals.confidence,
                sentiment_label=sentiment.label,
                low_confidence=low_confidence,
                escalated=decision.escalate,
                review_reason=review_reason,
            )

            self.store.append(PredictionRecord(
                timestamp=time.time(),
                version=assignment.version,
                intent=signals.intent,
                category=decision.category,
                queue=decision.queue,
                priority=decision.priority,
                confidence=signals.confidence,
                margin=signals.margin,
                sentiment_compound=sentiment.compound,
                sentiment_label=sentiment.label,
                urgency=signals.urgency,
                speech_act=signals.speech_act,
                needs_human_review=decision.needs_human_review,
                auto_routed=decision.auto_routed,
                escalate=decision.escalate,
                n_chars=len(text),
                n_words=len(text.split()),
                latency_ms=round(latency_ms, 3),
                request_id=item.request_id,
                unit_id=item.unit_id,
                true_intent=item.true_intent,
            ))

            results.append({
                "request_id": item.request_id,
                "model_version": assignment.version,
                "ab": {
                    "experiment_id": self.ab_config.experiment_id,
                    "version": assignment.version,
                    "bucket": assignment.bucket,
                    "reason": assignment.reason,
                    "unit_id": item.unit_id,
                },
                "cleaned_text": text,
                "classification": signals.as_dict(),
                "sentiment": sentiment.as_dict(),
                "routing": decision.as_dict(),
                "latency_ms": round(latency_ms, 3),
            })

        self.metrics.window_size.set(len(self.store))
        LOGGER.debug("triaged %d record(s) in %.1f ms",
                     len(items), 1000 * (time.perf_counter() - started))
        return results

    def triage(
        self, item: TriageInput, force_version: Optional[str] = None
    ) -> Dict[str, object]:
        return self.triage_many([item], force_version)[0]

    # -- monitoring ------------------------------------------------------

    def drift_report(self, window_seconds: Optional[float] = None) -> Dict[str, object]:
        records = self.store.snapshot(since_seconds=window_seconds)
        by_version: Dict[str, List[PredictionRecord]] = {}
        for record in records:
            by_version.setdefault(record.version, []).append(record)

        report = self.drift.report(by_version)
        for version, readings in report["by_version"].items():
            for reading in readings:
                self.metrics.set_drift(version, reading["feature"], reading["psi"])
        report["window_seconds"] = window_seconds
        report["store"] = self.store.stats
        return report
