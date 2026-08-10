"""Classifier interface, evaluation metrics, and the trained artefacts."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.base import IntentClassifier, IntentPrediction
from src.models.evaluate import confidence_metrics, evaluate, top_k_accuracy
from src.models.registry import ModelRegistry, UnknownVersionError
from tests.conftest import StubClassifier, requires_models


class TestPredictionInterface:
    def test_top_k_is_ordered_and_the_right_length(self):
        model = StubClassifier(rules={"stolen": "lost_or_stolen_card"})
        prediction = model.predict(["my card was stolen"], top_k=3)[0]
        assert prediction.intent == "lost_or_stolen_card"
        assert len(prediction.top_k) == 3
        scores = [s for _, s in prediction.top_k]
        assert scores == sorted(scores, reverse=True)

    def test_confidence_matches_the_top_entry(self):
        model = StubClassifier(rules={"stolen": "lost_or_stolen_card"}, confidence=0.77)
        prediction = model.predict(["stolen card"])[0]
        assert prediction.confidence == pytest.approx(0.77)
        assert prediction.top_k[0][1] == pytest.approx(0.77)

    def test_margin_is_the_gap_to_second_place(self):
        prediction = IntentPrediction("a", 0.6, [("a", 0.6), ("b", 0.25), ("c", 0.15)])
        assert prediction.margin == pytest.approx(0.35)

    def test_margin_falls_back_to_confidence_for_a_single_label(self):
        assert IntentPrediction("a", 0.6, [("a", 0.6)]).margin == pytest.approx(0.6)

    def test_empty_input_returns_empty(self):
        assert StubClassifier().predict([]) == []

    def test_top_k_is_clamped_to_the_label_space(self):
        model = StubClassifier()
        assert len(model.predict(["x"], top_k=500)[0].top_k) == len(model.labels)

    def test_shape_mismatch_is_caught(self):
        class Broken(IntentClassifier):
            labels = ["a", "b", "c"]

            def predict_proba(self, texts):
                return np.zeros((len(texts), 2))

        with pytest.raises(ValueError, match="columns for 3 labels"):
            Broken().predict(["x"])


class TestEvaluate:
    def test_perfect_predictions_score_one(self, taxonomy):
        truth = ["card_arrival", "exchange_rate", "lost_or_stolen_card"]
        labels = taxonomy.intents
        probabilities = np.zeros((3, len(labels)))
        for row, intent in enumerate(truth):
            probabilities[row, labels.index(intent)] = 1.0

        result = evaluate(truth, truth, probabilities, labels, taxonomy, "test")
        assert result["intent"]["accuracy"] == 1.0
        assert result["intent"]["macro_f1"] == 1.0
        assert result["category"]["accuracy"] == 1.0
        assert result["top_k_accuracy"]["top_1"] == 1.0

    def test_category_metrics_forgive_within_category_confusion(self, taxonomy):
        """Two intents in the same queue: intent is wrong, routing is right."""
        labels = taxonomy.intents
        truth = ["lost_or_stolen_card"]
        predicted = ["compromised_card"]        # same category
        probabilities = np.zeros((1, len(labels)))
        probabilities[0, labels.index(predicted[0])] = 1.0

        result = evaluate(truth, predicted, probabilities, labels, taxonomy, "test")
        assert result["intent"]["accuracy"] == 0.0
        assert result["category"]["accuracy"] == 1.0

    def test_per_class_support_sums_to_the_row_count(self, taxonomy):
        labels = taxonomy.intents
        truth = ["card_arrival"] * 4 + ["exchange_rate"] * 6
        probabilities = np.full((10, len(labels)), 1.0 / len(labels))
        result = evaluate(truth, truth, probabilities, labels, taxonomy, "test")
        assert sum(v["support"] for v in result["intent"]["per_class"].values()) == 10

    def test_top_k_accuracy_rises_with_k(self, taxonomy):
        labels = taxonomy.intents
        rng = np.random.default_rng(0)
        probabilities = rng.random((200, len(labels)))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        truth = [labels[i % len(labels)] for i in range(200)]
        values = [top_k_accuracy(probabilities, truth, labels, k) for k in (1, 3, 5)]
        assert values[0] <= values[1] <= values[2]


class TestConfidenceMetrics:
    def test_coverage_falls_as_the_threshold_rises(self):
        probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.35, 0.65]])
        correct = np.array([True, True, False])
        result = confidence_metrics(probabilities, correct, [0.0, 0.5, 0.8])
        coverage = [row["coverage"] for row in result["sweep"]]
        assert coverage == sorted(coverage, reverse=True)

    def test_perfectly_calibrated_input_has_near_zero_ece(self):
        probabilities = np.column_stack([np.full(1000, 0.95), np.full(1000, 0.05)])
        correct = np.array([True] * 950 + [False] * 50)
        result = confidence_metrics(probabilities, correct, [0.5])
        assert result["expected_calibration_error"] < 0.01

    def test_overconfident_input_has_large_ece(self):
        probabilities = np.column_stack([np.full(100, 0.99), np.full(100, 0.01)])
        correct = np.array([True] * 50 + [False] * 50)
        result = confidence_metrics(probabilities, correct, [0.5])
        assert result["expected_calibration_error"] > 0.4


class TestRegistry:
    def test_unknown_version_raises_with_a_useful_message(self):
        registry = ModelRegistry()
        with pytest.raises(UnknownVersionError, match="available"):
            registry.metadata("does-not-exist-v9")

    @requires_models
    def test_trained_models_load_and_predict(self):
        registry = ModelRegistry()
        assert registry.versions
        for version in registry.versions:
            model = registry.get(version)
            assert len(model.labels) == 77
            probabilities = model.predict_proba(["my card has not arrived yet"])
            assert probabilities.shape == (1, 77)
            assert probabilities.sum() == pytest.approx(1.0, abs=1e-4)

    @requires_models
    def test_predictions_are_stable_across_calls(self):
        registry = ModelRegistry()
        model = registry.get(registry.versions[0])
        text = ["I was charged twice for the same thing"]
        assert model.predict(text)[0].intent == model.predict(text)[0].intent

    @requires_models
    def test_describe_exposes_test_metrics(self):
        described = ModelRegistry().describe()
        assert all("test_intent_macro_f1" in entry for entry in described)
