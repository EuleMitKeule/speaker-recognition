"""Tests for backend training behavior."""

from __future__ import annotations

import numpy as np

from speaker_recognition.models import AudioInput, Config, TrainingRequest, VoiceSample
from speaker_recognition.recognizer import SpeakerRecognizer


class FakeEncoder:
    """Encoder test double returning the prepared waveform as an embedding."""

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        """Return a deterministic embedding for tests."""
        return wav


def _recognizer(tmp_path) -> SpeakerRecognizer:
    recognizer = SpeakerRecognizer.__new__(SpeakerRecognizer)
    recognizer._encoder = FakeEncoder()
    recognizer._reference_embeddings = {}
    recognizer._is_trained = False
    recognizer._config = Config(embeddings_directory=str(tmp_path))
    recognizer._embeddings_directory = tmp_path

    def process_audio_input(audio_input: AudioInput) -> np.ndarray:
        values = {
            "alice-1": np.array([1.0, 0.0], dtype=np.float32),
            "alice-2": np.array([0.0, 1.0], dtype=np.float32),
            "bob-1": np.array([-1.0, 0.0], dtype=np.float32),
        }
        return values[audio_input.audio_data]

    recognizer.process_audio_input = process_audio_input
    return recognizer


def test_train_averages_multiple_samples_per_user(tmp_path) -> None:
    """Multiple samples for one user are averaged into one reference embedding."""
    recognizer = _recognizer(tmp_path)

    result = recognizer.train(
        TrainingRequest(
            voice_samples=[
                VoiceSample(
                    user="alice",
                    audio=AudioInput(audio_data="alice-1", sample_rate=16000),
                ),
                VoiceSample(
                    user="alice",
                    audio=AudioInput(audio_data="alice-2", sample_rate=16000),
                ),
                VoiceSample(
                    user="bob",
                    audio=AudioInput(audio_data="bob-1", sample_rate=16000),
                ),
            ]
        )
    )

    assert result.status == "success"
    assert result.count == 2
    assert result.trained_users == ["alice", "bob"]
    np.testing.assert_allclose(
        recognizer._reference_embeddings["alice"],
        np.array([0.70710677, 0.70710677], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "alice_embedding.npy"),
        recognizer._reference_embeddings["alice"],
    )


def test_train_removes_stale_cached_embeddings(tmp_path) -> None:
    """Retraining replaces old cached embeddings with the current trained users."""
    np.save(tmp_path / "old_user_embedding.npy", np.array([1.0, 0.0], dtype=np.float32))
    recognizer = _recognizer(tmp_path)

    recognizer.train(
        TrainingRequest(
            voice_samples=[
                VoiceSample(
                    user="alice",
                    audio=AudioInput(audio_data="alice-1", sample_rate=16000),
                )
            ]
        )
    )

    assert not (tmp_path / "old_user_embedding.npy").exists()
    assert (tmp_path / "alice_embedding.npy").exists()


def test_load_embeddings_restores_training_state(tmp_path) -> None:
    """Cached embeddings can restore recognizer state after restart."""
    np.save(tmp_path / "alice_embedding.npy", np.array([1.0, 0.0], dtype=np.float32))
    np.save(tmp_path / "bob_embedding.npy", np.array([0.0, 1.0], dtype=np.float32))
    recognizer = _recognizer(tmp_path)

    assert recognizer.load_embeddings() is True

    assert recognizer.is_trained is True
    assert sorted(recognizer._reference_embeddings) == ["alice", "bob"]
    np.testing.assert_allclose(
        recognizer._reference_embeddings["alice"],
        np.array([1.0, 0.0], dtype=np.float32),
    )


def test_status_reports_loaded_embeddings(tmp_path) -> None:
    """Status response exposes trained users and embedding count."""
    np.save(tmp_path / "alice_embedding.npy", np.array([1.0, 0.0], dtype=np.float32))
    recognizer = _recognizer(tmp_path)
    recognizer.load_embeddings()

    status = recognizer.status()

    assert status.trained is True
    assert status.users == ["alice"]
    assert status.embeddings_count == 1
    assert status.embeddings_directory == str(tmp_path)


def test_load_embeddings_handles_missing_directory(tmp_path) -> None:
    """Missing cache directory leaves recognizer untrained."""
    recognizer = _recognizer(tmp_path / "missing")

    assert recognizer.load_embeddings() is False
    assert recognizer.is_trained is False
    assert recognizer.status().embeddings_count == 0
