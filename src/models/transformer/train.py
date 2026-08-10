"""Fine-tune a transformer encoder for intent classification.

    python -m src.models.transformer.train
    python -m src.models.transformer.train --smoke --epochs 1   # wiring check only

Deliberately a plain PyTorch loop rather than ``transformers.Trainer``: the
class-weighted loss, the per-epoch macro-F1 selection and the CPU thread
control are all clearer written out, and it drops the ``accelerate`` dependency.

Imbalance is handled the same way as the linear baseline — balanced class
weights — so the baseline/transformer comparison is not confounded by the two
models being given different treatment of the class distribution.

Two things make this trainable on a CPU in under an hour rather than three:
dynamic padding (pad to the longest member of each batch, not to a global 64)
and length-grouped batching (put similar-length records together so that
padding has little left to do). Both are in this file and in
``src/models/transformer/predict.py``.

Model selection is on validation macro-F1. The checkpoint written to disk is
the best epoch, not the last.
"""

from __future__ import annotations

import os

# Must precede the transformers import: transformers probes for TensorFlow at
# import time, and an unrelated broken TF install would take the process down.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from typing import Dict, Iterator, List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Sampler  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import MODELS_DIR, SEED, TRANSFORMER_VERSION, ensure_dirs  # noqa: E402
from src.etl.stats import class_weights  # noqa: E402
from src.logging_setup import configure  # noqa: E402
from src.models.base import write_metadata  # noqa: E402
from src.models.data import load_split, texts_and_labels  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.models.transformer.predict import batched_predict_proba  # noqa: E402
from src.taxonomy import load_taxonomy  # noqa: E402

LOGGER = logging.getLogger("transformer")

DEFAULT_CHECKPOINT = "distilbert-base-uncased"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EncodedDataset(Dataset):
    """Pre-tokenised, *unpadded* records. Padding happens in the collator."""

    def __init__(self, encodings, labels: np.ndarray) -> None:
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def lengths(self) -> List[int]:
        return [len(ids) for ids in self.input_ids]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": int(self.labels[idx]),
        }


def make_collate(tokenizer):
    def collate(batch: List[Dict[str, object]]):
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        padded = tokenizer.pad(
            [{k: v for k, v in item.items() if k != "labels"} for item in batch],
            return_tensors="pt",
        )
        padded["labels"] = labels
        return padded

    return collate


class LengthGroupedBatchSampler(Sampler):
    """Batch similar-length records together while staying close to random.

    Indices are shuffled, cut into mega-batches, sorted by length inside each
    mega-batch and then split into real batches whose order is shuffled again.
    Records still meet different neighbours every epoch, but a batch of short
    messages is no longer padded up to the length of one long outlier.
    """

    def __init__(
        self, lengths: List[int], batch_size: int, seed: int, mega_batch_factor: int = 50
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega = batch_size * mega_batch_factor
        self.epoch = 0
        self.seed = seed

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)

        batches: List[List[int]] = []
        for start in range(0, len(indices), self.mega):
            chunk = sorted(indices[start:start + self.mega], key=lambda i: self.lengths[i])
            batches.extend(
                chunk[i:i + self.batch_size]
                for i in range(0, len(chunk), self.batch_size)
            )
        rng.shuffle(batches)
        return iter(batches)


def train(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=0,
                        help="torch CPU threads; 0 leaves the default")
    parser.add_argument("--version", default=TRANSFORMER_VERSION)
    parser.add_argument("--smoke", action="store_true",
                        help="subsample to 800 training rows for a wiring check")
    args = parser.parse_args(argv)

    configure()
    ensure_dirs()
    set_seed(SEED)
    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s torch_threads=%d", device, torch.get_num_threads())

    taxonomy = load_taxonomy()
    labels = taxonomy.intents
    index = {label: i for i, label in enumerate(labels)}

    splits = {name: load_split(name) for name in ("train", "val", "test")}
    if args.smoke:
        splits["train"] = (
            splits["train"].sample(n=min(800, len(splits["train"])), random_state=SEED)
            .reset_index(drop=True)
        )
        LOGGER.warning("SMOKE RUN: training on %d rows; numbers are not reportable",
                       len(splits["train"]))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=index,
    ).to(device)

    train_texts, train_truth = texts_and_labels(splits["train"])
    encodings = tokenizer(train_texts, truncation=True, max_length=args.max_length)
    train_set = EncodedDataset(
        encodings, np.array([index[y] for y in train_truth])
    )

    untruncated = [len(ids) for ids in tokenizer(train_texts)["input_ids"]]
    cut_rate = round(float(np.mean([n > args.max_length for n in untruncated])), 4)
    LOGGER.info(
        "token lengths: median %d, p95 %d, max %d | %.2f%% truncated at %d",
        int(np.median(untruncated)), int(np.percentile(untruncated, 95)),
        max(untruncated), 100 * cut_rate, args.max_length,
    )

    sampler = LengthGroupedBatchSampler(train_set.lengths(), args.batch_size, SEED)
    train_loader = DataLoader(
        train_set, batch_sampler=sampler, collate_fn=make_collate(tokenizer)
    )

    weights = torch.tensor(
        class_weights(pd.Series(splits["train"]["intent"]), labels), device=device
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    decayed = [p for n, p in model.named_parameters()
               if not any(s in n for s in ("bias", "LayerNorm.weight"))]
    undecayed = [p for n, p in model.named_parameters()
                 if any(s in n for s in ("bias", "LayerNorm.weight"))]
    optimizer = torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": args.weight_decay},
            {"params": undecayed, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    total_steps = len(sampler) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    val_texts, y_val = texts_and_labels(splits["val"])
    history: List[Dict[str, float]] = []
    best_state, best_f1, best_epoch = None, -1.0, -1
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        running, seen = 0.0, 0
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * len(targets)
            seen += len(targets)
            if step % 50 == 0:
                LOGGER.info("epoch %d step %d/%d loss %.4f",
                            epoch, step, len(sampler), running / seen)

        probabilities = batched_predict_proba(
            model, tokenizer, val_texts, device,
            max_length=args.max_length, batch_size=args.eval_batch_size,
        )
        predicted = [labels[i] for i in probabilities.argmax(axis=1)]
        val_f1 = float(f1_score(y_val, predicted, average="macro", zero_division=0))
        elapsed = time.perf_counter() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 4),
            "val_macro_f1": round(val_f1, 4),
            "seconds": round(elapsed, 1),
        })
        LOGGER.info("epoch %d done in %.0fs | train loss %.4f | val macro-F1 %.4f",
                    epoch, elapsed, running / max(seen, 1), val_f1)

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            LOGGER.info("epoch %d is the best so far", epoch)

    total_seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    LOGGER.info("restored epoch %d (val macro-F1 %.4f)", best_epoch, best_f1)

    directory = MODELS_DIR / args.version
    directory.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    model.to(device)

    metrics = {}
    for name in ("val", "test"):
        texts, truth = texts_and_labels(splits[name])
        probabilities = batched_predict_proba(
            model, tokenizer, texts, device,
            max_length=args.max_length, batch_size=args.eval_batch_size,
        )
        predicted = [labels[i] for i in probabilities.argmax(axis=1)]
        metrics[name] = evaluate(truth, predicted, probabilities, labels, taxonomy, name)
        LOGGER.info("%-9s %-5s intent macro-F1 %.4f | category macro-F1 %.4f",
                    args.version, name,
                    metrics[name]["intent"]["macro_f1"],
                    metrics[name]["category"]["macro_f1"])

    config = {
        "algorithm": f"{args.checkpoint} fine-tuned for sequence classification",
        "imbalance_handling": "balanced class weights in CrossEntropyLoss",
        "checkpoint": args.checkpoint,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "batching": "length-grouped sampler + dynamic padding",
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "train_truncation_rate": cut_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW, linear decay with warmup, grad clip 1.0",
        "selection_metric": "validation macro-F1 over 77 intents",
        "trained_on": "train split only",
        "device": str(device),
        "seed": SEED,
        "smoke_run": bool(args.smoke),
        "train_seconds": round(total_seconds, 1),
        "history": history,
    }
    write_metadata(directory, args.version, "transformer", labels, config, metrics)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("saved -> %s (%.0fs total)", directory, total_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
