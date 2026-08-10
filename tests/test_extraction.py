"""Sentiment lexicon and the rule-derived intent signals."""

from __future__ import annotations

import pytest

from src.extraction.intent import classify_speech_act, extract, score_urgency
from src.extraction.sentiment import BANKING_LEXICON, label_for, score
from src.models.base import IntentPrediction


class TestSentiment:
    def test_complaint_scores_negative(self):
        assert score("This is absolutely terrible and I am furious").compound < -0.4

    def test_thanks_scores_positive(self):
        assert score("Thanks, this is sorted and I am very happy").compound > 0.4

    def test_neutral_question_is_neutral(self):
        assert score("What is the exchange rate today").label == "neutral"

    def test_domain_terms_are_recognised_and_reported(self):
        result = score("there is an unauthorised fraudulent charge")
        assert result.compound < 0
        assert "unauthorised" in result.matched_domain_terms
        assert "fraudulent" in result.matched_domain_terms

    def test_domain_lexicon_shifts_a_sentence_vader_alone_calls_neutral(self):
        """The banking terms must actually change the score, not just be listed."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        text = "the payment was unauthorised"
        vanilla = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        assert vanilla == 0.0
        assert score(text).compound < vanilla

    def test_never_claims_to_be_calibrated(self):
        assert score("anything").is_calibrated is False

    def test_empty_input_is_safe(self):
        assert score("").compound == 0.0

    @pytest.mark.parametrize("compound,expected", [
        (-0.9, "negative"), (-0.20, "negative"), (-0.19, "neutral"),
        (0.0, "neutral"), (0.05, "positive"), (0.8, "positive"),
    ])
    def test_label_boundaries(self, compound, expected):
        assert label_for(compound) == expected

    def test_lexicon_entries_are_in_vader_range(self):
        assert all(-4.0 <= v <= 4.0 for v in BANKING_LEXICON.values())


class TestSpeechAct:
    @pytest.mark.parametrize("text,expected", [
        ("What is the exchange rate?", "question"),
        ("My card is not working", "problem_report"),
        ("I have been waiting weeks and this is unacceptable, it still doesn't work!",
         "complaint"),
        ("Please send me a new card", "request"),
        ("My address is in France", "statement"),
    ])
    def test_classification(self, text, expected):
        assert classify_speech_act(text) == expected

    def test_complaint_beats_question_when_both_present(self):
        text = "This is ridiculous, why has it still not worked?"
        assert classify_speech_act(text) == "complaint"

    def test_strong_marker_alone_is_enough(self):
        """No problem verb and no exclamation mark, but plainly a complaint."""
        text = ("I have been charged twice for the same transaction and nobody "
                "has replied in two weeks. This is unacceptable.")
        assert classify_speech_act(text) == "complaint"

    def test_weak_marker_alone_is_not_a_complaint(self):
        """'Weeks' on its own is a timeline question, not a grievance."""
        assert classify_speech_act("How many weeks does delivery take") == "question"

    def test_weak_marker_plus_a_problem_is_a_complaint(self):
        assert classify_speech_act(
            "I have been waiting weeks and it still doesn't work"
        ) == "complaint"


class TestUrgency:
    def test_no_markers_is_zero(self):
        assert score_urgency("what is the exchange rate")[0] == 0.0

    def test_explicit_urgency_registers(self):
        value, markers = score_urgency("I need this fixed urgently")
        assert value >= 0.5 and "urgently" in markers

    def test_situational_words_alone_are_not_urgent(self):
        """A neutral question about spending abroad must not escalate."""
        value, markers = score_urgency(
            "What exchange rate do you use for purchases abroad?"
        )
        assert value == 0.0
        assert markers == []

    def test_situational_words_count_when_something_is_broken(self):
        value, markers = score_urgency("My card is blocked and I am abroad")
        assert value > 0.0 and "abroad" in markers

    def test_time_pressure_registers(self):
        value, markers = score_urgency("I am still waiting for my card")
        assert value >= 0.3 and "still waiting" in markers

    def test_shouting_needs_enough_text(self):
        assert "shouting" not in score_urgency("HELP")[1]
        assert "shouting" in score_urgency("THIS IS COMPLETELY UNACCEPTABLE")[1]

    def test_is_bounded_at_one(self):
        value, _ = score_urgency(
            "URGENT ASAP I AM STRANDED ABROAD AND STILL WAITING, HOW LONG!!!"
        )
        assert value == 1.0


class TestExtract:
    def _prediction(self, intent, confidence, runner_up=None, runner_score=0.0):
        top_k = [(intent, confidence)]
        if runner_up:
            top_k.append((runner_up, runner_score))
        return IntentPrediction(intent=intent, confidence=confidence, top_k=top_k)

    def test_maps_intent_to_category(self, taxonomy):
        signals = extract(
            "my card was stolen", self._prediction("lost_or_stolen_card", 0.9), taxonomy
        )
        assert signals.category == "identity_security"

    def test_flags_ambiguity_when_runners_up_are_close(self, taxonomy):
        signals = extract(
            "I was charged twice",
            self._prediction("transaction_charged_twice", 0.40,
                             "card_payment_fee_charged", 0.36),
            taxonomy,
        )
        assert signals.is_ambiguous is True
        assert signals.alternative_categories == ["fees_fx"]

    def test_no_alternative_when_runner_up_shares_the_queue(self, taxonomy):
        signals = extract(
            "my card is gone",
            self._prediction("lost_or_stolen_card", 0.40, "compromised_card", 0.38),
            taxonomy,
        )
        assert signals.is_ambiguous is True
        assert signals.alternative_categories == []

    def test_margin_is_the_gap_to_the_runner_up(self, taxonomy):
        signals = extract(
            "hello", self._prediction("card_arrival", 0.7, "card_delivery_estimate", 0.2),
            taxonomy,
        )
        assert signals.margin == pytest.approx(0.5)
