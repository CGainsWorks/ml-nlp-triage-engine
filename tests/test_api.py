"""HTTP surface. Stub models, so no trained artefacts are needed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.service import TriageService
from src.monitoring.metrics import TriageMetrics
from src.monitoring.store import PredictionStore
from src.routing.ab import ABConfig
from src.routing.rules import RoutingPolicy


@pytest.fixture
def client(monkeypatch, stub_registry, ab_payload, taxonomy):
    def fake_build_service():
        return TriageService(
            registry=stub_registry,
            ab_config=ABConfig(ab_payload),
            taxonomy=taxonomy,
            policy=RoutingPolicy(),
            metrics=TriageMetrics(),
            store=PredictionStore(capacity=5000),
        )

    monkeypatch.setattr(app_module, "build_service", fake_build_service)
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def unready_client(monkeypatch):
    monkeypatch.setattr(app_module, "build_service", lambda: None)
    monkeypatch.setitem(app_module._STATE, "error", "no trained models found")
    with TestClient(app_module.app) as test_client:
        yield test_client


class TestHealth:
    def test_healthz_is_ok(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_is_ready(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_root_lists_endpoints(self, client):
        assert "/v1/triage" in " ".join(client.get("/").json()["endpoints"])


class TestUnready:
    def test_liveness_still_answers_without_models(self, unready_client):
        assert unready_client.get("/healthz").status_code == 200

    def test_readiness_fails_with_an_actionable_message(self, unready_client):
        response = unready_client.get("/readyz")
        assert response.status_code == 503
        assert "no trained models" in response.json()["detail"]

    def test_triage_returns_503_not_a_stack_trace(self, unready_client):
        response = unready_client.post("/v1/triage", json={"text": "hello"})
        assert response.status_code == 503
        assert "detail" in response.json()

    def test_metrics_still_serves(self, unready_client):
        assert unready_client.get("/metrics").status_code == 200


class TestTriageEndpoint:
    def test_happy_path(self, client):
        response = client.post(
            "/v1/triage", json={"text": "my card was stolen", "client_id": "c1"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["classification"]["intent"] == "lost_or_stolen_card"
        assert body["routing"]["queue"] == "FRAUD_SECURITY"
        assert body["routing"]["priority"] == "P1"
        assert body["sentiment"]["is_calibrated"] is False

    def test_response_never_echoes_masked_pii(self, client):
        response = client.post("/v1/triage", json={
            "text": "card 4111 1111 1111 1111 was stolen", "client_id": "c1"
        })
        assert "4111" not in response.text

    def test_forced_version_is_honoured(self, client):
        response = client.post(
            "/v1/triage?version=treatment-v1",
            json={"text": "hello", "client_id": "c1"},
        )
        assert response.json()["model_version"] == "treatment-v1"

    def test_unknown_forced_version_is_a_400(self, client):
        response = client.post(
            "/v1/triage?version=ghost", json={"text": "hello", "client_id": "c1"}
        )
        assert response.status_code == 400

    def test_empty_text_is_rejected(self, client):
        assert client.post("/v1/triage", json={"text": ""}).status_code == 422

    def test_oversized_text_is_rejected(self, client):
        assert client.post(
            "/v1/triage", json={"text": "x" * 9000}
        ).status_code == 422

    def test_batch(self, client):
        response = client.post("/v1/triage/batch", json={"items": [
            {"text": "my card was stolen", "client_id": "a"},
            {"text": "what is the exchange rate", "client_id": "b"},
        ]})
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert body["results"][1]["classification"]["intent"] == "exchange_rate"

    def test_empty_batch_is_rejected(self, client):
        assert client.post("/v1/triage/batch", json={"items": []}).status_code == 422


class TestModelsAndAB:
    def test_lists_registered_models_with_weights(self, client):
        body = client.get("/v1/models").json()
        assert {m["version"] for m in body["models"]} == {"control-v1", "treatment-v1"}
        assert all(m["traffic_weight"] == 0.5 for m in body["models"])
        assert body["control_version"] == "control-v1"

    def test_ab_endpoint_describes_the_split(self, client):
        body = client.get("/v1/ab").json()
        assert body["config"]["weights"] == {"control-v1": 0.5, "treatment-v1": 0.5}
        assert body["assignment"]["sticky"] is True

    def test_weights_can_be_updated(self, client):
        response = client.put(
            "/v1/ab/weights",
            json={"weights": {"control-v1": 0.9, "treatment-v1": 0.1}},
        )
        assert response.status_code == 200
        assert response.json()["config"]["weights"]["control-v1"] == 0.9

    def test_weight_update_is_not_persisted_to_disk(self, client):
        """An incident-time ramp must not silently become the committed default."""
        response = client.put(
            "/v1/ab/weights", json={"weights": {"control-v1": 1.0, "treatment-v1": 0.0}}
        )
        assert response.json()["persisted"] is False

    def test_bad_weights_are_a_400(self, client):
        response = client.put(
            "/v1/ab/weights", json={"weights": {"control-v1": 0.3, "treatment-v1": 0.3}}
        )
        assert response.status_code == 400


class TestMonitoringEndpoints:
    def test_insights_reflect_served_traffic(self, client):
        for i in range(30):
            client.post("/v1/triage", json={
                "text": "my card was stolen", "client_id": f"c{i}"
            })
        body = client.get("/v1/insights").json()
        assert body["overall"]["n_records"] == 30
        assert set(body["versions_seen"]) <= {"control-v1", "treatment-v1"}
        assert body["overall"]["escalation_rate"] == 1.0

    def test_insights_can_be_filtered_to_one_version(self, client):
        for i in range(40):
            client.post("/v1/triage", json={"text": "hello", "client_id": f"c{i}"})
        body = client.get("/v1/insights?version=control-v1").json()
        assert body["versions_seen"] == ["control-v1"]

    def test_insights_declare_the_store_is_not_durable(self, client):
        assert client.get("/v1/insights").json()["store"]["durable"] is False

    def test_drift_endpoint_responds_without_references(self, client):
        body = client.get("/v1/monitoring/drift").json()
        assert "by_version" in body
        assert body["has_input_reference"] in (True, False)

    def test_metrics_are_labelled_by_version(self, client):
        client.post("/v1/triage", json={"text": "my card was stolen", "client_id": "c1"})
        text = client.get("/metrics").text
        assert "triage_requests_total" in text
        assert "triage_ab_traffic_weight" in text
        assert 'version="' in text


def test_openapi_schema_builds(client):
    assert client.get("/openapi.json").status_code == 200
