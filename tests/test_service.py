"""End-to-end pipeline behaviour, driven by stub models."""

from __future__ import annotations

import pytest

from src.api.service import TriageInput, TriageService
from src.monitoring.metrics import TriageMetrics
from src.monitoring.store import PredictionStore
from src.routing.ab import ABConfig
from src.routing.rules import RoutingPolicy


@pytest.fixture
def service(stub_registry, ab_payload, taxonomy):
    return TriageService(
        registry=stub_registry,
        ab_config=ABConfig(ab_payload),
        taxonomy=taxonomy,
        policy=RoutingPolicy(),
        metrics=TriageMetrics(),
        store=PredictionStore(capacity=1000),
    )


class TestTriage:
    def test_full_result_shape(self, service):
        result = service.triage(TriageInput("my card was stolen", client_id="c1"))
        assert set(result) == {
            "request_id", "model_version", "ab", "cleaned_text",
            "classification", "sentiment", "routing", "latency_ms",
        }
        assert result["classification"]["intent"] == "lost_or_stolen_card"
        assert result["routing"]["queue"] == "FRAUD_SECURITY"
        assert result["routing"]["priority"] == "P1"

    def test_text_is_cleaned_before_the_model_sees_it(self, service):
        result = service.triage(
            TriageInput("card 4111 1111 1111 1111 was stolen", client_id="c1")
        )
        assert "<CARDNUM>" in result["cleaned_text"]
        assert "4111" not in result["cleaned_text"]

    def test_low_confidence_is_held_for_review(self, service):
        result = service.triage(
            TriageInput("something the stub has no rule for", client_id="c1")
        )
        assert result["routing"]["needs_human_review"] is True
        assert result["routing"]["auto_routed"] is False
        assert result["routing"]["queue"] == "MANUAL_TRIAGE"

    def test_assignment_is_sticky_per_client(self, service):
        versions = {
            service.triage(TriageInput("hello", client_id="same"))["model_version"]
            for _ in range(20)
        }
        assert len(versions) == 1

    def test_both_arms_receive_traffic(self, service):
        served = {
            service.triage(TriageInput("hello", client_id=f"c{i}"))["model_version"]
            for i in range(200)
        }
        assert served == {"control-v1", "treatment-v1"}

    def test_request_without_identifier_goes_to_control(self, service):
        result = service.triage(TriageInput("my card was stolen"))
        assert result["model_version"] == "control-v1"
        assert "no unit identifier" in result["ab"]["reason"]

    def test_force_version_bypasses_the_split(self, service):
        result = service.triage(
            TriageInput("hello", client_id="c1"), force_version="treatment-v1"
        )
        assert result["model_version"] == "treatment-v1"
        assert result["ab"]["reason"] == "forced by caller"

    def test_batch_matches_single(self, service):
        items = [TriageInput("my card was stolen", client_id=f"c{i}") for i in range(10)]
        batch = service.triage_many(items)
        singles = [service.triage(item) for item in items]
        assert [r["model_version"] for r in batch] == [r["model_version"] for r in singles]
        assert [r["classification"]["intent"] for r in batch] == \
               [r["classification"]["intent"] for r in singles]

    def test_empty_batch(self, service):
        assert service.triage_many([]) == []

    def test_every_record_lands_in_the_monitoring_window(self, service):
        service.triage_many([TriageInput("hi", client_id=f"c{i}") for i in range(25)])
        assert len(service.store) == 25

    def test_the_injected_store_is_the_one_written_to(self, stub_registry, ab_payload):
        """An empty PredictionStore is falsy (it defines __len__), so a
        `store or default` in the constructor would silently drop it and the
        caller would watch a window that never fills."""
        store = PredictionStore(capacity=500)
        service = TriageService(
            registry=stub_registry, ab_config=ABConfig(ab_payload), store=store
        )
        service.triage_many([TriageInput("hi", client_id=f"c{i}") for i in range(7)])
        assert service.store is store
        assert len(store) == 7
        assert len(store.snapshot()) == 7

    def test_metrics_separate_the_two_versions(self, service):
        service.triage_many([TriageInput("hi", client_id=f"c{i}") for i in range(100)])
        rendered = service.metrics.render().decode()
        assert 'version="control-v1"' in rendered
        assert 'version="treatment-v1"' in rendered

    def test_replayed_ground_truth_is_stored_but_not_returned(self, service):
        result = service.triage(
            TriageInput("my card was stolen", client_id="c1",
                        true_intent="lost_or_stolen_card")
        )
        assert "true_intent" not in result
        assert service.store.snapshot()[-1].true_intent == "lost_or_stolen_card"


class TestWeightUpdates:
    def test_weights_can_be_shifted_at_runtime(self, service):
        service.set_weights({"control-v1": 1.0, "treatment-v1": 0.0})
        served = {
            service.triage(TriageInput("hi", client_id=f"c{i}"))["model_version"]
            for i in range(100)
        }
        assert served == {"control-v1"}

    def test_unknown_version_is_rejected(self, service):
        with pytest.raises(ValueError, match="not registered"):
            service.set_weights({"control-v1": 0.5, "ghost-v1": 0.5})

    def test_invalid_weights_are_rejected(self, service):
        with pytest.raises(ValueError, match="sum to 1.0"):
            service.set_weights({"control-v1": 0.2, "treatment-v1": 0.2})

    def test_ramping_keeps_existing_assignments_stable(self, service):
        """Shifting weights must not reshuffle customers who stay in the arm."""
        before = {
            f"c{i}": service.triage(TriageInput("hi", client_id=f"c{i}"))["model_version"]
            for i in range(300)
        }
        service.set_weights({"control-v1": 0.75, "treatment-v1": 0.25})
        after = {
            f"c{i}": service.triage(TriageInput("hi", client_id=f"c{i}"))["model_version"]
            for i in range(300)
        }
        kept = [c for c, v in before.items() if v == "control-v1"]
        assert all(after[c] == "control-v1" for c in kept), (
            "growing the control arm moved customers who were already in it"
        )
