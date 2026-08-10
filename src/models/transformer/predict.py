"""Batched inference for the fine-tuned transformer.

The batching here is not incidental. Padding every record to a fixed 64 tokens
wastes most of the compute, because the median customer message is around 13
tokens. Two things fix that and both live in :func:`batched_predict_proba`:

* pad each batch to the longest member of *that batch*, not to a global maximum;
* sort by length before batching so similar-length records share a batch, then
  restore the caller's order afterwards.

On CPU this is worth roughly a 3x speedup on this corpus, which is the
difference between a transformer that can be retrained on a laptop and one that
cannot.
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path  # noqa: E402
from typing import List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.models.base import IntentClassifier, read_metadata  # noqa: E402


@torch.no_grad()
def batched_predict_proba(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    max_length: int = 64,
    batch_size: int = 64,
) -> np.ndarray:
    texts = list(texts)
    if not texts:
        return np.zeros((0, model.config.num_labels), dtype=np.float64)

    encoded = tokenizer(texts, truncation=True, max_length=max_length)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    was_training = model.training
    model.eval()
    probabilities = np.zeros((len(texts), model.config.num_labels), dtype=np.float64)

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer.pad(
            {
                "input_ids": [encoded["input_ids"][i] for i in idx],
                "attention_mask": [encoded["attention_mask"][i] for i in idx],
            },
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probabilities[idx] = torch.softmax(logits, dim=-1).cpu().numpy()

    if was_training:
        model.train()
    return probabilities


class TransformerClassifier(IntentClassifier):
    """Serves a checkpoint written by ``src.models.transformer.train``."""

    def __init__(
        self,
        directory: Path,
        batch_size: int = 64,
        device: Optional[str] = None,
        threads: int = 0,
    ) -> None:
        directory = Path(directory)
        metadata = read_metadata(directory)
        self.version = metadata["version"]
        self.family = metadata["family"]
        self.labels: List[str] = list(metadata["labels"])
        self.metadata = metadata
        self.max_length = int(metadata["train_config"].get("max_length", 64))
        self.batch_size = batch_size

        if threads:
            torch.set_num_threads(threads)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory)
        self.model.to(self.device).eval()

        if self.model.config.num_labels != len(self.labels):
            raise ValueError(
                f"{self.version}: checkpoint has {self.model.config.num_labels} "
                f"output units but metadata lists {len(self.labels)} labels"
            )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return batched_predict_proba(
            self.model, self.tokenizer, texts, self.device,
            max_length=self.max_length, batch_size=self.batch_size,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TransformerClassifier(version={self.version!r}, "
            f"device={self.device}, labels={len(self.labels)})"
        )
