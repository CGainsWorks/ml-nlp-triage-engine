"""FastAPI application.

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

The service starts even when no models are trained yet. It reports itself as
live but not ready, and the triage endpoints return 503 with an instruction
rather than a stack trace. A container that crash-loops because an artefact is
missing tells an operator much less than one that says what is missing.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from src import __version__
from src.api import schemas
from src.api.service import TriageInput, TriageService
from src.logging_setup import configure
from src.models.registry import ModelRegistry, UnknownVersionError
from src.monitoring import insights
from src.monitoring.drift import DriftMonitor
from src.monitoring.metrics import CONTENT_TYPE_LATEST, TriageMetrics
from src.monitoring.store import PredictionStore
from src.routing.ab import load_ab_config
from src.routing.rules import RoutingPolicy
from src.taxonomy import load_taxonomy

LOGGER = logging.getLogger("api")

_STATE: dict = {"service": None, "error": None, "started_at": None}


def build_service() -> Optional[TriageService]:
    """Assemble the service, or return None with the reason recorded."""
    registry = ModelRegistry()
    if not registry.versions:
        _STATE["error"] = (
            "no trained models found. Run: python -m src.etl.build_dataset && "
            "python -m src.models.baseline.train"
        )
        LOGGER.error(_STATE["error"])
        return None

    ab_config = load_ab_config()
    missing = set(ab_config.active_versions) - set(registry.versions)
    if missing:
        LOGGER.warning(
            "A/B config references versions that are not trained: %s. "
            "Falling back to serving only what exists.", sorted(missing)
        )
        available = {v: w for v, w in ab_config.weights.items() if v in registry.versions}
        if not available or sum(available.values()) <= 0:
            _STATE["error"] = (
                f"none of the A/B versions {sorted(ab_config.weights)} are trained; "
                f"available: {registry.versions}"
            )
            LOGGER.error(_STATE["error"])
            return None
        total = sum(available.values())
        payload = ab_config.as_dict()
        payload["salt"] = ab_config.salt
        payload["weights"] = {v: w / total for v, w in available.items()}
        if ab_config.control_version not in available:
            payload["control_version"] = max(available, key=available.get)
        from src.routing.ab import ABConfig

        ab_config = ABConfig(payload)

    registry.preload(ab_config.active_versions)
    service = TriageService(
        registry=registry,
        ab_config=ab_config,
        taxonomy=load_taxonomy(),
        policy=RoutingPolicy.from_file(),
        metrics=TriageMetrics(),
        store=PredictionStore(
            capacity=int(os.environ.get("TRIAGE_WINDOW_CAPACITY", "20000"))
        ),
        drift=DriftMonitor.from_artifacts(registry.versions),
    )
    _STATE["error"] = None
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure()
    _STATE["started_at"] = time.time()
    try:
        _STATE["service"] = build_service()
    except Exception as exc:  # noqa: BLE001 - surface it on /readyz, do not crash-loop
        _STATE["error"] = f"{type(exc).__name__}: {exc}"
        _STATE["service"] = None
        LOGGER.exception("failed to build the triage service")
    yield
    _STATE["service"] = None


app = FastAPI(
    title="ml-nlp-triage-engine",
    version=__version__,
    description=(
        "Text triage: intent classification, sentiment and intent extraction, "
        "rule-based routing, and A/B assignment between model versions with "
        "per-version monitoring."
    ),
    lifespan=lifespan,
)


def get_service() -> TriageService:
    service = _STATE["service"]
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=_STATE["error"] or "service is not ready",
        )
    return service


# -- health ---------------------------------------------------------------


@app.get("/healthz", response_model=schemas.HealthResponse, tags=["health"])
def healthz() -> schemas.HealthResponse:
    """Liveness. The process is up; it may not have models."""
    service = _STATE["service"]
    return schemas.HealthResponse(
        status="ok",
        versions_loaded=service.registry.loaded_versions if service else [],
        versions_available=service.registry.versions if service else [],
        detail=None if service else _STATE["error"],
    )


@app.get("/readyz", response_model=schemas.HealthResponse, tags=["health"])
def readyz() -> Response:
    """Readiness. Fails with 503 until at least one model is servable."""
    service = _STATE["service"]
    if service is None:
        return JSONResponse(
            status_code=503,
            content=schemas.HealthResponse(
                status="not_ready", versions_loaded=[], versions_available=[],
                detail=_STATE["error"],
            ).model_dump(),
        )
    return JSONResponse(
        status_code=200,
        content=schemas.HealthResponse(
            status="ready",
            versions_loaded=service.registry.loaded_versions,
            versions_available=service.registry.versions,
        ).model_dump(),
    )


# -- triage ---------------------------------------------------------------


@app.post("/v1/triage", response_model=schemas.TriageResponse, tags=["triage"])
def triage(
    request: schemas.TriageRequest,
    version: Optional[str] = Query(
        None, description="Bypass A/B and force a model version. For debugging."
    ),
    service: TriageService = Depends(get_service),
) -> schemas.TriageResponse:
    try:
        result = service.triage(
            TriageInput(
                text=request.text,
                client_id=request.client_id,
                request_id=request.request_id,
                true_intent=request.true_intent,
            ),
            force_version=version,
        )
    except (KeyError, UnknownVersionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.TriageResponse(**result)


@app.post("/v1/triage/batch", response_model=schemas.BatchTriageResponse, tags=["triage"])
def triage_batch(
    request: schemas.BatchTriageRequest,
    version: Optional[str] = Query(None),
    service: TriageService = Depends(get_service),
) -> schemas.BatchTriageResponse:
    started = time.perf_counter()
    try:
        results = service.triage_many(
            [
                TriageInput(
                    text=item.text,
                    client_id=item.client_id,
                    request_id=item.request_id,
                    true_intent=item.true_intent,
                )
                for item in request.items
            ],
            force_version=version,
        )
    except (KeyError, UnknownVersionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return schemas.BatchTriageResponse(
        results=[schemas.TriageResponse(**r) for r in results],
        count=len(results),
        latency_ms=round(1000 * (time.perf_counter() - started), 3),
    )


# -- models and A/B -------------------------------------------------------


@app.get("/v1/models", response_model=schemas.ModelsResponse, tags=["models"])
def list_models(service: TriageService = Depends(get_service)) -> schemas.ModelsResponse:
    described = service.registry.describe()
    for entry in described:
        entry["traffic_weight"] = service.ab_config.weights.get(entry["version"], 0.0)
    return schemas.ModelsResponse(
        models=[schemas.ModelInfo(**entry) for entry in described],
        experiment_id=service.ab_config.experiment_id,
        control_version=service.ab_config.control_version,
    )


@app.get("/v1/ab", tags=["models"])
def ab_config(service: TriageService = Depends(get_service)) -> dict:
    return {
        "config": service.ab_config.as_dict(),
        "assignment": {
            "method": "sha256(experiment_id:salt:unit_id) -> [0,1) -> cumulative weights",
            "sticky": True,
            "stateless": True,
        },
        "served_counts": {
            version: len(service.store.snapshot(version=version))
            for version in service.ab_config.versions
        },
    }


@app.put("/v1/ab/weights", tags=["models"])
def update_weights(
    update: schemas.WeightUpdate, service: TriageService = Depends(get_service)
) -> dict:
    """Shift traffic between versions without a redeploy.

    Deliberately does not persist to `config/ab_config.json`: a ramp made
    during an incident should not silently become the committed default.
    """
    try:
        updated = service.set_weights(update.weights, update.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"config": updated.as_dict(), "persisted": False}


# -- monitoring -----------------------------------------------------------


@app.get("/v1/insights", tags=["monitoring"])
def get_insights(
    window_seconds: Optional[float] = Query(
        None, ge=1, description="Only include records from the last N seconds."
    ),
    version: Optional[str] = Query(None, description="Restrict to one model version."),
    service: TriageService = Depends(get_service),
) -> dict:
    records = service.store.snapshot(version=version, since_seconds=window_seconds)
    return {
        "store": service.store.stats,
        "filter": {"version": version, "window_seconds": window_seconds},
        **insights.report(records, window_seconds),
    }


@app.get("/v1/monitoring/drift", tags=["monitoring"])
def get_drift(
    window_seconds: Optional[float] = Query(None, ge=1),
    service: TriageService = Depends(get_service),
) -> dict:
    return service.drift_report(window_seconds)


@app.get("/metrics", tags=["monitoring"])
def metrics() -> Response:
    """Prometheus exposition. Every series carries a `version` label."""
    service = _STATE["service"]
    if service is None:
        return Response(content="", media_type=CONTENT_TYPE_LATEST)
    return Response(content=service.metrics.render(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "service": "ml-nlp-triage-engine",
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            "POST /v1/triage", "POST /v1/triage/batch", "GET /v1/models",
            "GET /v1/ab", "PUT /v1/ab/weights", "GET /v1/insights",
            "GET /v1/monitoring/drift", "GET /metrics",
            "GET /healthz", "GET /readyz",
        ],
    }
