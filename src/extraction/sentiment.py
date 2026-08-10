"""Sentiment scoring.

**Read this before quoting any sentiment number.**

BANKING77 has no sentiment labels. Nothing here is trained, and nothing here
has been evaluated against ground truth, because there is no ground truth to
evaluate against. This is a lexicon — VADER (Hutto & Gilbert, ICWSM 2014, MIT
licence) plus a small banking-specific extension — and it is used as a
*priority signal* in routing, never as a reported classifier.

What that means in practice:

* the service returns a sentiment score and the routing rules may bump a
  record's priority on the strength of it;
* the README reports no sentiment accuracy, precision or F1, because computing
  one would require inventing labels;
* the only sentiment evidence in this repository is a face-validity check
  (`experiments/sentiment_face_validity.py`) showing that categories which are
  complaints by construction score more negatively than categories which are
  questions by construction. That is weak evidence and it is labelled as such.

If this were going to production against real traffic, the honest next step is
to label a few thousand records and either calibrate this or replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Sequence

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Terms that carry obvious valence in a banking support context but are absent
# or neutral in the general-purpose VADER lexicon. Values are on VADER's scale,
# roughly -4 (most negative) to +4 (most positive).
BANKING_LEXICON: Dict[str, float] = {
    "unauthorised": -2.6,
    "unauthorized": -2.6,
    "fraudulent": -3.0,
    "fraudulently": -3.0,
    "scammed": -3.2,
    "overcharged": -2.4,
    "double-charged": -2.4,
    "declined": -1.6,
    "blocked": -1.8,
    "frozen": -1.8,
    "swallowed": -1.9,
    "unrecognised": -1.5,
    "unrecognized": -1.5,
    "chargeback": -1.2,
    "disputed": -1.4,
    "overdrawn": -2.0,
    "stranded": -2.6,
    "useless": -2.8,
    "ridiculous": -2.4,
    "unacceptable": -2.6,
    "resolved": 1.8,
    "refunded": 1.6,
    "reimbursed": 1.6,
    "sorted": 1.4,
}

# Boundaries on VADER's compound score. The 0.05 cut is VADER's own published
# convention; we widen the neutral band to 0.20 on the negative side because
# routing acts on negative sentiment and we would rather under-escalate than
# flood the fraud queue.
NEGATIVE_THRESHOLD = -0.20
POSITIVE_THRESHOLD = 0.05


@dataclass
class SentimentResult:
    """An uncalibrated sentiment reading. See the module docstring."""

    compound: float
    label: str
    positive: float
    neutral: float
    negative: float
    method: str = "vader+banking_lexicon"
    is_calibrated: bool = False
    matched_domain_terms: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "compound": self.compound,
            "label": self.label,
            "positive": self.positive,
            "neutral": self.neutral,
            "negative": self.negative,
            "method": self.method,
            "is_calibrated": self.is_calibrated,
            "matched_domain_terms": self.matched_domain_terms,
        }


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(BANKING_LEXICON)
    return analyzer


def label_for(compound: float) -> str:
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    return "neutral"


def score(text: str) -> SentimentResult:
    analyzer = _analyzer()
    scores = analyzer.polarity_scores(text or "")
    lowered = (text or "").lower()
    matched = sorted(term for term in BANKING_LEXICON if term in lowered)
    return SentimentResult(
        compound=round(float(scores["compound"]), 4),
        label=label_for(scores["compound"]),
        positive=round(float(scores["pos"]), 4),
        neutral=round(float(scores["neu"]), 4),
        negative=round(float(scores["neg"]), 4),
        matched_domain_terms=matched,
    )


def score_many(texts: Sequence[str]) -> List[SentimentResult]:
    return [score(t) for t in texts]
