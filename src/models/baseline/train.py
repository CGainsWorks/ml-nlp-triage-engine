"""Train the scikit-learn baselines.

    python -m src.models.baseline.train --model linear
    python -m src.models.baseline.train --model gbdt
    python -m src.models.baseline.train --model both

Two families, both on TF-IDF features:

**linear** — word 1-2 grams unioned with character 3-5 grams, into multinomial
logistic regression. The character grams are not decoration: customer messages
are full of typos and the word view alone loses those records. Regularisation
strength is selected on the validation split.

**gbdt** — word TF-IDF reduced to a dense space with truncated SVD, into
XGBoost. Trees cannot use 50k sparse columns sensibly, so the projection is
mandatory rather than an optimisation.

Both handle class imbalance by weighting: ``class_weight="balanced"`` for the
linear model, the same weights passed as per-row ``sample_weight`` for XGBoost.
Hyperparameters are chosen on validation; the model that gets saved is trained
on the training split only, so validation numbers stay out of sample.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from xgboost import XGBClassifier

from src.config import (
    BASELINE_GBDT_VERSION,
    BASELINE_LINEAR_VERSION,
    MODELS_DIR,
    SEED,
    ensure_dirs,
)
from src.etl.stats import class_weights
from src.logging_setup import configure
from src.models.base import write_metadata
from src.models.data import load_split, texts_and_labels
from src.models.evaluate import evaluate
from src.taxonomy import load_taxonomy

LOGGER = logging.getLogger("baseline")

# Extended until the winner stopped being the largest value on offer. A grid
# whose optimum sits on its own edge has not finished searching.
C_GRID = (1.0, 4.0, 16.0, 64.0, 256.0)


def _feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )


def train_linear(
    x_train: List[str], y_train: List[str], x_val: List[str], y_val: List[str]
) -> Tuple[Pipeline, Dict[str, Any]]:
    features = _feature_union()
    LOGGER.info("fitting TF-IDF features on %d documents", len(x_train))
    matrix_train = features.fit_transform(x_train)
    matrix_val = features.transform(x_val)
    LOGGER.info("feature matrix: %s", matrix_train.shape)

    search: List[Dict[str, float]] = []
    best_c, best_score = None, -1.0
    for c in C_GRID:
        clf = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
            random_state=SEED,
        )
        clf.fit(matrix_train, y_train)
        score = f1_score(y_val, clf.predict(matrix_val), average="macro", zero_division=0)
        search.append({"C": c, "val_macro_f1": round(float(score), 4)})
        LOGGER.info("C=%-5s val macro-F1 =%.4f", c, score)
        if score > best_score:
            best_c, best_score = c, score

    LOGGER.info("selected C=%s (val macro-F1 %.4f)", best_c, best_score)
    pipeline = Pipeline(
        [
            ("features", _feature_union()),
            (
                "clf",
                LogisticRegression(
                    C=best_c,
                    class_weight="balanced",
                    max_iter=2000,
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    config = {
        "algorithm": "TF-IDF (word 1-2 + char_wb 3-5) -> LogisticRegression",
        "imbalance_handling": "class_weight='balanced'",
        "n_features": int(matrix_train.shape[1]),
        "selected_C": best_c,
        "selection_metric": "validation macro-F1 over 77 intents",
        "search": search,
        "trained_on": "train split only",
        "seed": SEED,
    }
    return pipeline, config


def train_gbdt(
    x_train: List[str], y_train: List[str], x_val: List[str], y_val: List[str],
    labels: List[str], n_components: int = 256, n_estimators: int = 600,
) -> Tuple[Pipeline, Dict[str, Any]]:
    vectoriser = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)

    LOGGER.info("fitting TF-IDF + SVD(%d)", n_components)
    sparse_train = vectoriser.fit_transform(x_train)
    dense_train = svd.fit_transform(sparse_train)
    dense_val = svd.transform(vectoriser.transform(x_val))
    explained = float(svd.explained_variance_ratio_.sum())
    LOGGER.info("SVD keeps %.1f%% of TF-IDF variance", 100 * explained)

    index = {label: i for i, label in enumerate(labels)}
    y_train_idx = np.array([index[y] for y in y_train])
    y_val_idx = np.array([index[y] for y in y_val])

    import pandas as pd  # local: only needed for the weight lookup

    weights = class_weights(pd.Series(y_train), labels)
    sample_weight = weights[y_train_idx]

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(labels),
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=SEED,
    )
    LOGGER.info("boosting up to %d rounds over %d classes", n_estimators, len(labels))
    model.fit(
        dense_train,
        y_train_idx,
        sample_weight=sample_weight,
        eval_set=[(dense_val, y_val_idx)],
        verbose=False,
    )
    best_rounds = int(getattr(model, "best_iteration", n_estimators - 1)) + 1
    LOGGER.info("early stopping chose %d rounds", best_rounds)

    pipeline = Pipeline([("tfidf", vectoriser), ("svd", svd), ("clf", model)])

    config = {
        "algorithm": f"TF-IDF (word 1-2) -> TruncatedSVD({n_components}) -> XGBoost",
        "imbalance_handling": "balanced sample_weight per row",
        "svd_components": n_components,
        "svd_explained_variance_ratio": round(explained, 4),
        "max_rounds": n_estimators,
        "best_rounds": best_rounds,
        "early_stopping_rounds": 30,
        "max_depth": 6,
        "learning_rate": 0.2,
        "selection_metric": "validation mlogloss (early stopping)",
        "trained_on": "train split only",
        "seed": SEED,
    }
    return pipeline, config


def _persist(
    pipeline: Pipeline, version: str, config: Dict[str, Any], labels: List[str],
    splits, taxonomy, elapsed: float,
) -> Dict[str, Any]:
    directory = MODELS_DIR / version
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, directory / "model.joblib", compress=3)

    # XGBoost's classes_ are integer indices; map back through `labels`.
    is_xgb = isinstance(pipeline.named_steps.get("clf"), XGBClassifier)
    proba_labels = labels if is_xgb else list(pipeline.named_steps["clf"].classes_)

    metrics = {}
    for split_name, frame in splits.items():
        texts, truth = texts_and_labels(frame)
        probabilities = pipeline.predict_proba(texts)
        predicted = [proba_labels[i] for i in probabilities.argmax(axis=1)]
        metrics[split_name] = evaluate(
            truth, predicted, probabilities, proba_labels, taxonomy, split_name
        )
        LOGGER.info(
            "%-9s %-5s intent macro-F1 %.4f | category macro-F1 %.4f",
            version, split_name,
            metrics[split_name]["intent"]["macro_f1"],
            metrics[split_name]["category"]["macro_f1"],
        )

    config = dict(config, train_seconds=round(elapsed, 1))
    write_metadata(directory, version, "baseline", proba_labels, config, metrics)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("linear", "gbdt", "both"), default="both")
    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=600)
    args = parser.parse_args(argv)

    configure()
    ensure_dirs()
    taxonomy = load_taxonomy()
    labels = taxonomy.intents

    splits = {name: load_split(name) for name in ("train", "val", "test")}
    x_train, y_train = texts_and_labels(splits["train"])
    x_val, y_val = texts_and_labels(splits["val"])

    if args.model in ("linear", "both"):
        start = time.perf_counter()
        pipeline, config = train_linear(x_train, y_train, x_val, y_val)
        _persist(pipeline, BASELINE_LINEAR_VERSION, config, labels, splits,
                 taxonomy, time.perf_counter() - start)

    if args.model in ("gbdt", "both"):
        start = time.perf_counter()
        pipeline, config = train_gbdt(
            x_train, y_train, x_val, y_val, labels,
            n_components=args.svd_components, n_estimators=args.n_estimators,
        )
        _persist(pipeline, BASELINE_GBDT_VERSION, config, labels, splits,
                 taxonomy, time.perf_counter() - start)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
