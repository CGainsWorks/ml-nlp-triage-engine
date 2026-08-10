"""Intent-side signals: the model's distribution plus rule-derived shape.

Two different things are called "intent" in this project and it is worth being
precise about which is which.

**Classified intent** — one of the 77 BANKING77 labels, produced by whichever
model version the A/B layer selected. This is learned, and its accuracy is
reported in the README.

**Speech act and urgency** — is the customer asking a question, reporting a
problem, or complaining, and how pressed are they? These are *rules*, listed
explicitly below. They are not learned and they are not evaluated, because the
dataset carries no labels for them. They exist because routing needs them: a
question about a fee and a complaint about the same fee belong in the same
queue but not at the same priority.

Keeping them as transparent rules rather than a second model is a deliberate
trade. An operations team can read this file and argue with it, which is worth
more here than a few points of an accuracy figure nobody can measure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.models.base import IntentPrediction
from src.taxonomy import Taxonomy

_WH_WORDS = re.compile(
    r"\b(what|why|how|when|where|which|who|can|could|do|does|did|is|are|will|would|should)\b",
    re.IGNORECASE,
)
# Complaint markers come in two strengths, and conflating them was a bug worth
# keeping the fix visible for. "unacceptable" is a complaint on its own. "weeks"
# is not — "my card should arrive in two weeks" is a neutral question — so the
# weak markers only count when something has also gone wrong.
_STRONG_COMPLAINT_TERMS = re.compile(
    r"\b(unacceptable|ridiculous|useless|terrible|awful|appalling|disgraceful|"
    r"angry|furious|livid|disappointed|disgusted|fed up|complain(?:t|ts|ing)?|"
    r"escalate|speak to (?:a |your )?manager)\b",
    re.IGNORECASE,
)
_WEAK_COMPLAINT_TERMS = re.compile(
    r"\b(still|again|never|nobody|no one|third time|second time|weeks?|days?)\b",
    re.IGNORECASE,
)
_REQUEST_TERMS = re.compile(
    r"\b(please|can you|could you|i (?:would like|want|need)|help me|let me know|"
    r"send me|i'd like)\b",
    re.IGNORECASE,
)
_PROBLEM_TERMS = re.compile(
    # "not work" needs its inflections: "still not worked" is as much a problem
    # report as "not working", and missing it sends complaints to the wrong act.
    r"\b(not work(?:ing|ed|s)?|doesn'?t work|failed|failing|error|problem|issue|"
    r"wrong|missing|stuck|declined|blocked|unable|can'?t|cannot|didn'?t)\b",
    re.IGNORECASE,
)
# Unambiguous urgency: these words mean "now" whatever else is in the sentence.
_URGENCY_TERMS = re.compile(
    r"\b(urgent(?:ly)?|asap|immediately|right now|emergency|straight away|"
    r"as soon as possible|stranded)\b",
    re.IGNORECASE,
)
# Situational markers that only imply urgency when something has gone wrong.
# "What rate do you use for purchases abroad?" is not an emergency; "my card is
# blocked and I'm abroad" is. Counting these unconditionally escalated ordinary
# questions, which is why they are gated on a problem term.
_URGENCY_CONTEXT_TERMS = re.compile(
    r"\b(abroad|overseas|travelling|traveling|on holiday|on a trip|stuck)\b",
    re.IGNORECASE,
)
_TIME_PRESSURE = re.compile(
    r"\b(still waiting|how (?:much )?long|when will|already been|over a week|"
    r"weeks? ago|days? ago)\b",
    re.IGNORECASE,
)

SPEECH_ACTS = ("question", "complaint", "problem_report", "request", "statement")


@dataclass
class IntentSignals:
    """Everything the routing layer needs to know about intent."""

    intent: str
    confidence: float
    margin: float
    category: str
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    alternative_categories: List[str] = field(default_factory=list)
    speech_act: str = "statement"
    urgency: float = 0.0
    urgency_markers: List[str] = field(default_factory=list)
    is_ambiguous: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "category": self.category,
            "top_k": [{"intent": i, "score": round(s, 4)} for i, s in self.top_k],
            "alternative_categories": self.alternative_categories,
            "speech_act": self.speech_act,
            "urgency": round(self.urgency, 4),
            "urgency_markers": self.urgency_markers,
            "is_ambiguous": self.is_ambiguous,
        }


def classify_speech_act(text: str) -> str:
    """First match wins; the order encodes which reading dominates.

    A message that both complains and asks a question is a complaint — the
    complaint is what changes how it should be handled.
    """
    if _STRONG_COMPLAINT_TERMS.search(text):
        return "complaint"
    if _WEAK_COMPLAINT_TERMS.search(text) and (
        _PROBLEM_TERMS.search(text) or "!" in text
    ):
        return "complaint"
    if _PROBLEM_TERMS.search(text):
        return "problem_report"
    if "?" in text or _WH_WORDS.match(text.strip()):
        return "question"
    if _REQUEST_TERMS.search(text):
        return "request"
    return "statement"


def score_urgency(text: str) -> Tuple[float, List[str]]:
    """Bounded 0-1 urgency from explicit markers. Additive and capped."""
    markers: List[str] = []
    score = 0.0

    explicit = _URGENCY_TERMS.findall(text)
    if explicit:
        score += 0.5
        markers.extend(sorted({m.lower() for m in explicit}))

    if _PROBLEM_TERMS.search(text):
        situational = _URGENCY_CONTEXT_TERMS.findall(text)
        if situational:
            score += 0.2
            markers.extend(sorted({m.lower() for m in situational}))

    pressure = _TIME_PRESSURE.findall(text)
    if pressure:
        score += 0.3
        markers.extend(sorted({m.lower() for m in pressure}))

    if text.count("!") >= 1:
        score += 0.1
        markers.append("exclamation")

    # Shouting, but only if there is enough text for it to be deliberate.
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        score += 0.2
        markers.append("shouting")

    return min(round(score, 4), 1.0), markers


AMBIGUITY_MARGIN = 0.15


def extract(
    text: str, prediction: IntentPrediction, taxonomy: Taxonomy
) -> IntentSignals:
    category = taxonomy.category_for_intent(prediction.intent).key

    # Categories the runner-up intents point at. When the alternatives sit in a
    # different queue, a wrong top-1 is expensive, so routing wants to know.
    alternatives: List[str] = []
    for intent, _ in prediction.top_k[1:]:
        key = taxonomy.category_for_intent(intent).key
        if key != category and key not in alternatives:
            alternatives.append(key)

    urgency, markers = score_urgency(text)
    return IntentSignals(
        intent=prediction.intent,
        confidence=prediction.confidence,
        margin=prediction.margin,
        category=category,
        top_k=prediction.top_k,
        alternative_categories=alternatives,
        speech_act=classify_speech_act(text),
        urgency=urgency,
        urgency_markers=markers,
        is_ambiguous=prediction.margin < AMBIGUITY_MARGIN,
    )
