"""The rolling window of served predictions that monitoring reads from.

This is an in-process, bounded, in-memory ring buffer. That is a real
limitation and it is stated rather than hidden: with more than one replica each
holds its own window, and everything is lost on restart. It is enough to
demonstrate per-version monitoring, drift and insight aggregation from a single
service instance, which is what this project claims.

The production shape is the same code writing each record to a queue or an
append-only table, with the aggregation running over that instead. The
:class:`PredictionStore` interface is deliberately narrow so that swap is a
small change.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional

DEFAULT_CAPACITY = 20_000


@dataclass
class PredictionRecord:
    """One served triage decision, flattened for aggregation."""

    timestamp: float
    version: str
    intent: str
    category: str
    queue: str
    priority: int
    confidence: float
    margin: float
    sentiment_compound: float
    sentiment_label: str
    urgency: float
    speech_act: str
    needs_human_review: bool
    auto_routed: bool
    escalate: bool
    n_chars: int
    n_words: int
    latency_ms: float
    request_id: Optional[str] = None
    unit_id: Optional[str] = None
    #: Only present in replay/backtest traffic, never in live serving.
    true_intent: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class PredictionStore:
    """Thread-safe bounded window over recent predictions."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self.capacity = capacity
        self._records: Deque[PredictionRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._total_seen = 0

    def append(self, record: PredictionRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._total_seen += 1

    def extend(self, records: List[PredictionRecord]) -> None:
        with self._lock:
            self._records.extend(records)
            self._total_seen += len(records)

    def snapshot(
        self,
        version: Optional[str] = None,
        since_seconds: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[PredictionRecord]:
        with self._lock:
            records = list(self._records)

        if version is not None:
            records = [r for r in records if r.version == version]
        if since_seconds is not None:
            cutoff = time.time() - since_seconds
            records = [r for r in records if r.timestamp >= cutoff]
        if limit is not None:
            records = records[-limit:]
        return records

    def versions(self) -> List[str]:
        with self._lock:
            return sorted({r.version for r in self._records})

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._total_seen = 0

    @property
    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "held": len(self._records),
                "total_seen": self._total_seen,
                "evicted": max(0, self._total_seen - len(self._records)),
                "durable": False,
                "scope": "single process",
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
