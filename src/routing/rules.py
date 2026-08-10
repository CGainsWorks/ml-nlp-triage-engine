"""Triage rules: from a classified record to a queue, a priority and an SLA.

The rules are ordinary code rather than a learned policy, for the same reason
the speech-act rules are: there are no routing labels in the dataset, so a
learned router could not be evaluated. What *can* be done honestly is to make
the policy explicit, make it configurable, and test it.

Two invariants the tests enforce:

* **Priority is never lowered below the category's base.** Rules escalate; they
  do not de-escalate. A model that is wrong about sentiment can make something
  more urgent than it needed to be, which is recoverable. The reverse is not.
* **Security-critical categories are never parked.** A low-confidence record
  that looks like fraud still goes to the fraud queue, flagged for a human. It
  does not sit in a manual-triage backlog waiting for someone to pick it up.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from src.config import DEFAULT_CONFIDENCE_THRESHOLD, ROUTING_POLICY_PATH
from src.extraction.intent import IntentSignals
from src.extraction.sentiment import SentimentResult
from src.taxonomy import Taxonomy

LOGGER = logging.getLogger(__name__)

#: Where records go when the model is not confident enough to route them.
REVIEW_QUEUE = "MANUAL_TRIAGE"

#: Priorities run 1 (most urgent) to 4. Nothing escalates past 1.
MAX_PRIORITY = 1
MIN_PRIORITY = 4


@dataclass(frozen=True)
class RoutingPolicy:
    """Thresholds the rules fire on. Kept in one object so it can be tuned."""

    #: Fallback for a model version with no entry in ``version_thresholds``.
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    #: Per-version auto-route thresholds. These are **not** interchangeable: a
    #: 0.75 top-1 probability means something different from a TF-IDF logistic
    #: regression than from a fine-tuned transformer, whose mean confidence on
    #: the same data is 0.96. Sharing one number would either hold back most of
    #: the linear model's correct predictions or wave through most of the
    #: transformer's wrong ones. Values come from
    #: experiments/tune_confidence_threshold.py and live in
    #: config/routing_policy.json.
    version_thresholds: Mapping[str, float] = field(default_factory=dict)
    strong_negative_sentiment: float = -0.5
    high_urgency: float = 0.5
    #: Escalate at most this many priority steps above the category base, so a
    #: single noisy signal cannot drag routine traffic into the P1 queue.
    max_escalation_steps: int = 2
    review_queue: str = REVIEW_QUEUE

    def threshold_for(self, version: Optional[str]) -> float:
        if version is None:
            return self.confidence_threshold
        return self.version_thresholds.get(version, self.confidence_threshold)

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "RoutingPolicy":
        """Load from ``config/routing_policy.json``, falling back to defaults."""
        target = Path(path) if path is not None else ROUTING_POLICY_PATH
        if not target.exists():
            LOGGER.warning(
                "%s not found; using built-in defaults. Regenerate it with "
                "`python -m experiments.tune_confidence_threshold --write-config`",
                target,
            )
            return cls()

        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(
            confidence_threshold=float(
                payload.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
            ),
            version_thresholds={
                str(k): float(v)
                for k, v in dict(payload.get("version_thresholds", {})).items()
            },
            strong_negative_sentiment=float(payload.get("strong_negative_sentiment", -0.5)),
            high_urgency=float(payload.get("high_urgency", 0.5)),
            max_escalation_steps=int(payload.get("max_escalation_steps", 2)),
            review_queue=str(payload.get("review_queue", REVIEW_QUEUE)),
        )


@dataclass
class TriageDecision:
    queue: str
    priority: int
    base_priority: int
    sla_minutes: int
    category: str
    escalate: bool = False
    needs_human_review: bool = False
    auto_routed: bool = True
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "queue": self.queue,
            "priority": f"P{self.priority}",
            "priority_level": self.priority,
            "base_priority": f"P{self.base_priority}",
            "sla_minutes": self.sla_minutes,
            "category": self.category,
            "escalate": self.escalate,
            "needs_human_review": self.needs_human_review,
            "auto_routed": self.auto_routed,
            "reasons": self.reasons,
        }


def effective_sla(base_sla: int, base_priority: int, priority: int) -> int:
    """Halve the SLA for each priority step the record was escalated by."""
    steps = max(0, base_priority - priority)
    return max(5, int(base_sla / (2 ** steps)))


def route(
    signals: IntentSignals,
    sentiment: SentimentResult,
    taxonomy: Taxonomy,
    policy: Optional[RoutingPolicy] = None,
    version: Optional[str] = None,
) -> TriageDecision:
    """Turn a classified record into a queue, a priority and an SLA.

    ``version`` selects the model's own auto-route threshold; omit it and the
    policy default applies.
    """
    policy = policy or RoutingPolicy()
    threshold = policy.threshold_for(version)
    category = taxonomy.categories[signals.category]

    priority = category.base_priority
    reasons: List[str] = [
        f"category '{category.key}' has base priority P{category.base_priority}"
    ]
    escalate = False
    steps = 0

    def bump(reason: str) -> None:
        nonlocal priority, steps
        if steps >= policy.max_escalation_steps or priority <= MAX_PRIORITY:
            reasons.append(f"{reason} (no further escalation available)")
            return
        priority -= 1
        steps += 1
        reasons.append(f"{reason} -> P{priority}")

    if category.security_critical:
        escalate = True
        priority = MAX_PRIORITY
        reasons.append(
            f"category '{category.key}' is security-critical -> P{MAX_PRIORITY}, escalated"
        )

    if sentiment.compound <= policy.strong_negative_sentiment:
        bump(f"strongly negative sentiment ({sentiment.compound:+.2f})")

    if signals.urgency >= policy.high_urgency:
        bump(f"urgency {signals.urgency:.2f} with markers {signals.urgency_markers}")

    if signals.speech_act == "complaint":
        bump("message reads as a complaint")

    priority = max(MAX_PRIORITY, min(MIN_PRIORITY, priority))

    queue = category.queue
    needs_review = False
    auto_routed = True

    if signals.confidence < threshold:
        needs_review = True
        reasons.append(
            f"confidence {signals.confidence:.2f} is below the "
            f"{threshold:.2f} auto-route threshold"
            + (f" for {version}" if version else "")
        )
        if category.security_critical:
            # Flag it, but do not park it. See the module docstring.
            reasons.append(
                "kept in the security queue despite low confidence because the "
                "cost of a delayed fraud report exceeds the cost of a false alarm"
            )
        else:
            queue = policy.review_queue
            auto_routed = False

    elif signals.is_ambiguous and signals.alternative_categories:
        needs_review = True
        reasons.append(
            f"top two intents are within {signals.margin:.2f} and point at "
            f"different queues ({', '.join(signals.alternative_categories)})"
        )

    return TriageDecision(
        queue=queue,
        priority=priority,
        base_priority=category.base_priority,
        sla_minutes=effective_sla(category.sla_minutes, category.base_priority, priority),
        category=category.key,
        escalate=escalate,
        needs_human_review=needs_review,
        auto_routed=auto_routed,
        reasons=reasons,
    )
