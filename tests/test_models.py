"""Unit tests for the model registry and stub model."""

import numpy as np
import pytest

from asr_robustness.models import available, create
from asr_robustness.models.base import ASRModel


def test_stub_is_registered():
    assert "stub" in available()


def test_create_stub_returns_asr_model():
    model = create("stub", transcript="hello world")
    assert isinstance(model, ASRModel)
    assert model.transcribe(np.zeros(16000, dtype=np.float32), 16000) == "hello world"


def test_stub_per_call_responses():
    model = create("stub", responses=["first", "second"])
    audio = np.zeros(1600, dtype=np.float32)
    assert model.transcribe(audio, 16000) == "first"
    assert model.transcribe(audio, 16000) == "second"
    assert model.transcribe(audio, 16000) == "first"  # wraps around


def test_transcribe_batch_default():
    model = create("stub", responses=["a", "b", "c"])
    audios = [np.zeros(1600, dtype=np.float32)] * 3
    assert model.transcribe_batch(audios, 16000) == ["a", "b", "c"]


def test_create_unknown_model_raises():
    with pytest.raises(KeyError):
        create("no-such-model")
