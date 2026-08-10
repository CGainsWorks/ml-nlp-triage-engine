"""Request and response models for the triage API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

MAX_TEXT_CHARS = 8000
MAX_BATCH_SIZE = 256


class TriageItem(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS,
                      description="Raw customer message.")
    client_id: Optional[str] = Field(
        None,
        description="Stable identifier for the customer. Drives A/B assignment, "
                    "so the same customer always sees the same model version. "
                    "If omitted, request_id is used, and failing that the "
                    "assignment falls back to the control version.",
    )
    request_id: Optional[str] = Field(None, description="Caller's correlation id.")
    true_intent: Optional[str] = Field(
        None,
        description="Ground-truth intent. Offline evaluation only — set by "
                    "experiments/run_ab_experiment.py when replaying a labelled "
                    "split so per-version accuracy can be computed through the "
                    "same code path the service uses. Never set by live traffic.",
    )


class TriageRequest(TriageItem):
    pass


class BatchTriageRequest(BaseModel):
    items: List[TriageItem] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class TopIntent(BaseModel):
    intent: str
    score: float


class Classification(BaseModel):
    intent: str
    confidence: float
    margin: float
    category: str
    top_k: List[TopIntent]
    alternative_categories: List[str]
    speech_act: str
    urgency: float
    urgency_markers: List[str]
    is_ambiguous: bool


class Sentiment(BaseModel):
    compound: float
    label: str
    positive: float
    neutral: float
    negative: float
    method: str
    is_calibrated: bool = Field(
        ...,
        description="Always false. The sentiment lexicon has never been "
                    "evaluated against labelled data for this domain.",
    )
    matched_domain_terms: List[str]


class Routing(BaseModel):
    queue: str
    priority: str
    priority_level: int
    base_priority: str
    sla_minutes: int
    category: str
    escalate: bool
    needs_human_review: bool
    auto_routed: bool
    reasons: List[str]


class ABInfo(BaseModel):
    experiment_id: str
    version: str
    bucket: float
    reason: str
    unit_id: Optional[str]


class TriageResponse(BaseModel):
    request_id: Optional[str]
    model_version: str
    ab: ABInfo
    cleaned_text: str
    classification: Classification
    sentiment: Sentiment
    routing: Routing
    latency_ms: float


class BatchTriageResponse(BaseModel):
    results: List[TriageResponse]
    count: int
    latency_ms: float


class ModelInfo(BaseModel):
    version: str
    family: Optional[str] = None
    created_utc: Optional[str] = None
    n_labels: Optional[int] = None
    algorithm: Optional[str] = None
    loaded: bool = False
    test_intent_macro_f1: Optional[float] = None
    test_category_macro_f1: Optional[float] = None
    traffic_weight: float = 0.0


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    experiment_id: str
    control_version: str


class WeightUpdate(BaseModel):
    weights: Dict[str, float] = Field(
        ..., description="Version -> share of traffic. Must sum to 1.0."
    )
    enabled: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    versions_loaded: List[str]
    versions_available: List[str]
    detail: Optional[str] = None
