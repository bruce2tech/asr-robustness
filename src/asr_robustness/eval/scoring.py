"""Transcript scoring: WER, CER, and error-type breakdowns.

Two design points matter for the research questions:

* **Text normalization.** Reference (LibriSpeech, upper-case, no punctuation)
  and hypothesis (Whisper, mixed-case, punctuated) must be put on equal footing
  before scoring, or WER measures formatting differences instead of recognition
  errors. :func:`normalize` handles casing, punctuation, and whitespace.

* **Error-type split and hallucination signals.** WER alone hides *how* a model
  fails. We keep substitutions / insertions / deletions separately, plus the
  hypothesis/reference length ratio. Under heavy noise, generative models
  (Whisper) tend to *hallucinate* -- emitting fluent but invented text -- which
  shows up as a spike in the **insertion rate** and a **length ratio** well
  above 1. Those are exactly the metrics this module surfaces.
"""

from __future__ import annotations

import re

import jiwer

_PUNCT = re.compile(r"[^\w\s']")  # keep apostrophes (contractions); drop other punctuation
_WHITESPACE = re.compile(r"\s+")


def _make_normalizer():
    """Build Whisper's English text normalizer, if available.

    Whisper's normalizer is the de-facto standard for English ASR evaluation:
    beyond casing/punctuation it expands abbreviations ("Mr." -> "mister") and
    standardizes numbers and contractions. Without it, WER conflates *formatting*
    differences between reference and hypothesis with genuine recognition
    errors. It is pure-regex (no torch), so importing it stays cheap.
    """
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        return EnglishTextNormalizer({})  # empty US/UK spelling map is fine here
    except Exception:
        return None


_WHISPER_NORMALIZER = _make_normalizer()


def _basic_normalize(text: str) -> str:
    """Fallback normalizer: casing, punctuation, whitespace only."""
    text = text.lower().replace("’", "'")  # curly -> straight apostrophe
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Normalize text for fair WER scoring (Whisper normalizer, else basic)."""
    if _WHISPER_NORMALIZER is not None:
        return _WHISPER_NORMALIZER(text)
    return _basic_normalize(text)


def score_utterance(reference: str, hypothesis: str) -> dict:
    """Score one (reference, hypothesis) pair.

    Returns WER, CER, the raw hit / substitution / insertion / deletion counts,
    word counts, and the hypothesis/reference length ratio.
    """
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    ref_words = len(ref.split())
    hyp_words = len(hyp.split())

    # jiwer needs non-empty strings; a single space scores as an empty utterance.
    out = jiwer.process_words(ref or " ", hyp or " ")
    return {
        "wer": float(out.wer),
        "cer": float(jiwer.cer(ref or " ", hyp or " ")),
        "hits": int(out.hits),
        "substitutions": int(out.substitutions),
        "insertions": int(out.insertions),
        "deletions": int(out.deletions),
        "ref_words": ref_words,
        "hyp_words": hyp_words,
        "length_ratio": (hyp_words / ref_words) if ref_words else 0.0,
    }


def aggregate(scored: list[dict]) -> dict:
    """Aggregate per-utterance scores into corpus-level metrics.

    Corpus WER is ``total_errors / total_reference_words`` -- the standard
    word-weighted definition, **not** the mean of per-utterance WERs (which
    over-weights short utterances).
    """
    if not scored:
        return {"n_utterances": 0, "wer": 0.0, "ref_words": 0}

    subs = sum(s["substitutions"] for s in scored)
    ins = sum(s["insertions"] for s in scored)
    dels = sum(s["deletions"] for s in scored)
    ref_words = sum(s["ref_words"] for s in scored)
    hyp_words = sum(s["hyp_words"] for s in scored)

    return {
        "n_utterances": len(scored),
        "ref_words": ref_words,
        "wer": (subs + ins + dels) / ref_words if ref_words else 0.0,
        "substitution_rate": subs / ref_words if ref_words else 0.0,
        "insertion_rate": ins / ref_words if ref_words else 0.0,  # hallucination signal
        "deletion_rate": dels / ref_words if ref_words else 0.0,
        "length_ratio": hyp_words / ref_words if ref_words else 0.0,  # >1 => over-generation
        "mean_cer": sum(s["cer"] for s in scored) / len(scored),
    }
