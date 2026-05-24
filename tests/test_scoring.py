"""Unit tests for transcript scoring."""

import pytest

from asr_robustness.eval.scoring import aggregate, normalize, score_utterance


def test_normalize_strips_case_and_punctuation():
    out = normalize("Hello, World!")
    assert out == out.lower()
    assert "," not in out and "!" not in out
    assert normalize("  THE   QUICK  brown ") == "the quick brown"


def test_normalize_unifies_reference_and_hypothesis_formatting():
    # The point of normalization: a model writing "Mr." must not be penalised
    # against a reference that spells out "MISTER".
    assert normalize("MISTER QUILTER") == normalize("Mr. Quilter")


def test_perfect_match_is_zero_wer():
    s = score_utterance("THE QUICK BROWN FOX", "the quick brown fox")
    assert s["wer"] == 0.0
    assert s["substitutions"] == s["insertions"] == s["deletions"] == 0
    assert s["hits"] == 4


def test_single_substitution():
    s = score_utterance("the quick brown fox", "the quick green fox")
    assert s["substitutions"] == 1
    assert s["wer"] == pytest.approx(0.25)  # 1 error / 4 reference words


def test_insertion_is_counted():
    s = score_utterance("the quick fox", "the quick brown fox")
    assert s["insertions"] == 1
    assert s["hyp_words"] == 4
    assert s["length_ratio"] == pytest.approx(4 / 3)


def test_deletion_is_counted():
    s = score_utterance("the quick brown fox", "the quick fox")
    assert s["deletions"] == 1
    assert s["length_ratio"] == pytest.approx(3 / 4)


def test_hallucination_shows_as_insertions_and_long_ratio():
    # A model that invents a fluent sentence on (effectively) noise.
    s = score_utterance("yes", "yes and then they all went home together happily")
    assert s["insertions"] >= 8
    assert s["length_ratio"] > 5.0


def test_aggregate_uses_word_weighted_corpus_wer():
    # Utterance A: 1 error / 10 words. Utterance B: 1 error / 2 words.
    long_utt = score_utterance("a b c d e f g h i j", "a b c d e f g h i X")
    short_utt = score_utterance("k l", "k X")
    agg = aggregate([long_utt, short_utt])
    # Corpus WER = 2 errors / 12 words -- NOT mean(0.1, 0.5) = 0.3.
    assert agg["wer"] == pytest.approx(2 / 12)
    assert agg["ref_words"] == 12
    assert agg["n_utterances"] == 2


def test_aggregate_empty_is_safe():
    agg = aggregate([])
    assert agg["n_utterances"] == 0
    assert agg["wer"] == 0.0
