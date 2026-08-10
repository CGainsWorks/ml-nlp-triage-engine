"""Cleaning, deduplication, leakage control and split statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.etl.clean import clean_text, is_usable
from src.etl.split import deduplicate, remove_test_leakage, train_val_split
from src.etl.stats import CHAR_BINS, class_weights, histogram, length_reference


class TestClean:
    def test_masks_card_numbers(self):
        assert "<CARDNUM>" in clean_text("my card 4111 1111 1111 1111 was declined")

    def test_masks_sort_code_before_generic_numbers(self):
        assert "<SORTCODE>" in clean_text("sort code 20-00-00 please")

    def test_masks_amounts_with_symbol_and_with_word(self):
        assert "<AMOUNT>" in clean_text("you took £25.99 from me")
        assert "<AMOUNT>" in clean_text("you took 25.99 GBP from me")

    def test_masks_email_and_url(self):
        cleaned = clean_text("mail me at a.b@example.com or see https://x.test/y")
        assert "<EMAIL>" in cleaned and "<URL>" in cleaned
        assert "example.com" not in cleaned

    def test_bare_numbers_become_num(self):
        assert clean_text("I waited 14 days") == "I waited <NUM> days"

    def test_normalises_curly_quotes(self):
        assert clean_text("it hasn’t arrived") == "it hasn't arrived"

    def test_collapses_whitespace_and_control_characters(self):
        assert clean_text("  a\x0bb \n\n c ") == "a b c"

    def test_preserves_case(self):
        assert clean_text("Why Was I Charged") == "Why Was I Charged"

    def test_is_deterministic_and_idempotent_on_masked_output(self):
        once = clean_text("card 4111111111111111 charged £5")
        assert once == clean_text("card 4111111111111111 charged £5")
        # Re-cleaning must not mangle the mask tokens it just inserted.
        assert clean_text(once) == once

    def test_handles_none_and_non_strings(self):
        assert clean_text(None) == ""
        assert clean_text(12345) == "<NUM>"

    @pytest.mark.parametrize("text,expected", [
        ("hi there", True),
        ("", False),
        ("   ", False),
        ("<NUM> <AMOUNT>", False),   # nothing but masks carries no signal
        ("paid <AMOUNT>", True),
    ])
    def test_is_usable(self, text, expected):
        assert is_usable(text) is expected


class TestSplit:
    def test_drops_exact_duplicates(self):
        frame = pd.DataFrame({
            "text": ["a", "a", "b"],
            "intent": ["x", "x", "y"],
        })
        out, stats = deduplicate(frame)
        assert stats["exact_duplicates_dropped"] == 1
        assert len(out) == 2

    def test_counts_label_conflicts_and_keeps_one(self):
        frame = pd.DataFrame({
            "text": ["same", "same", "other"],
            "intent": ["x", "y", "z"],
        })
        out, stats = deduplicate(frame)
        assert stats["label_conflicting_texts"] == 1
        assert stats["label_conflicting_rows_dropped"] == 1
        assert len(out) == 2

    def test_leakage_is_removed_from_train_never_from_test(self):
        train = pd.DataFrame({"text": ["shared", "train only"], "intent": ["x", "y"]})
        test = pd.DataFrame({"text": ["shared", "test only"], "intent": ["x", "z"]})
        cleaned, stats = remove_test_leakage(train, test)
        assert stats["train_rows_overlapping_test"] == 1
        assert list(cleaned["text"]) == ["train only"]
        assert len(test) == 2

    def test_split_is_stratified_and_covers_every_class(self):
        frame = pd.DataFrame({
            "text": [f"t{i}" for i in range(200)],
            "intent": ["a", "b", "c", "d"] * 50,
        })
        train, val = train_val_split(frame, val_fraction=0.25, seed=42)
        assert len(train) == 150 and len(val) == 50
        assert set(train["intent"]) == set(val["intent"]) == {"a", "b", "c", "d"}
        # 50 validation rows over 4 classes cannot divide evenly; stratification
        # means the counts differ by at most one, not that they are identical.
        counts = val["intent"].value_counts()
        assert counts.max() - counts.min() <= 1

    def test_split_is_reproducible(self):
        frame = pd.DataFrame({
            "text": [f"t{i}" for i in range(100)],
            "intent": ["a", "b"] * 50,
        })
        first, _ = train_val_split(frame, 0.2, seed=7)
        second, _ = train_val_split(frame, 0.2, seed=7)
        assert list(first["text"]) == list(second["text"])

    def test_refuses_to_stratify_a_singleton_class(self):
        frame = pd.DataFrame({"text": ["a", "b", "c"], "intent": ["x", "x", "lonely"]})
        with pytest.raises(ValueError, match="fewer than 2 examples"):
            train_val_split(frame)


class TestStats:
    def test_histogram_sums_to_one(self):
        values = [5, 25, 35, 45, 55, 70, 90, 120, 150, 400]
        proportions = histogram(values, CHAR_BINS)
        assert len(proportions) == len(CHAR_BINS) - 1
        assert pytest.approx(sum(proportions), abs=1e-6) == 1.0

    def test_histogram_of_nothing_is_all_zero(self):
        assert histogram([], CHAR_BINS) == [0.0] * (len(CHAR_BINS) - 1)

    def test_class_weights_are_inverse_to_frequency(self):
        labels = pd.Series(["a"] * 90 + ["b"] * 10)
        weights = class_weights(labels, ["a", "b"])
        assert weights[1] > weights[0]
        # Balanced weighting: weight_i * count_i is constant across classes.
        assert np.isclose(weights[0] * 90, weights[1] * 10)

    def test_class_weights_cover_unseen_classes_without_dividing_by_zero(self):
        weights = class_weights(pd.Series(["a", "a"]), ["a", "b"])
        assert np.isfinite(weights).all()

    def test_length_reference_has_both_views(self):
        reference = length_reference(pd.Series(["short", "a much longer message here"]))
        assert set(reference) == {"n_rows", "chars", "words"}
        assert pytest.approx(sum(reference["chars"]["proportions"]), abs=1e-6) == 1.0
