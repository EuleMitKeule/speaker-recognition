"""Speaker recognition logic."""

import base64
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore[import-untyped]

from speaker_recognition.models import (
    AudioInput,
    Config,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
    config,
)

_LOGGER = logging.getLogger(__name__)


class SpeakerRecognizer:
    """Handle speaker recognition operations."""

    def __init__(self, config: Config) -> None:
        """Initialize the speaker recognizer.

        Args:
            config: Application configuration
        """
        self._encoder: VoiceEncoder = VoiceEncoder()
        self._reference_embeddings: dict[str, NDArray[np.float32]] = {}
        self._is_trained = False
        self._config = config
        self._embeddings_directory = Path(config.embeddings_directory)

    @property
    def is_trained(self) -> bool:
        """Check if the model is trained."""
        return self._is_trained

    @property
    def embeddings_directory(self) -> Path:
        """Get the embeddings directory."""
        return self._embeddings_directory

    @embeddings_directory.setter
    def embeddings_directory(self, value: str) -> None:
        """Set the embeddings directory.

        Args:
            value: New embeddings directory path
        """
        self._config.embeddings_directory = value
        self._embeddings_directory = Path(value)

    def process_audio_input(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Process audio input from base64 encoded data.

        Args:
            audio_input: Audio input containing base64 encoded audio

        Returns:
            Preprocessed audio waveform
        """
        audio_bytes = base64.b64decode(audio_input.audio_data)
        audio_array_int16 = np.frombuffer(audio_bytes, dtype=np.int16).copy()

        if audio_array_int16.size == 0:
            raise ValueError("Empty audio data")

        audio_array_float32 = audio_array_int16.astype(np.float32) / 32768.0
        result: NDArray[np.float32] = preprocess_wav(
            audio_array_float32, source_sr=audio_input.sample_rate
        )
        return result

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Train the speaker recognition model.

        Args:
            request: Training request with voice samples

        Returns:
            TrainingResult with status, trained users and count
        """
        if not request.voice_samples:
            raise ValueError("No voice samples provided")

        self._embeddings_directory.mkdir(parents=True, exist_ok=True)

        self._reference_embeddings = {}
        _LOGGER.info(f"Training with {len(request.voice_samples)} voice samples")

        for sample in request.voice_samples:
            user_id = sample.user
            audio_input = sample.audio

            _LOGGER.info(f"Processing voice sample for user: {user_id}")

            try:
                embedding: NDArray[np.float32]
                embedding_path = self._embeddings_directory / f"{user_id}_embedding.npy"

                if embedding_path.exists():
                    _LOGGER.debug(f"Loading cached embedding from {embedding_path}")
                    loaded_data = np.load(embedding_path, allow_pickle=False)
                    embedding = np.asarray(loaded_data)
                else:
                    _LOGGER.debug("Creating embedding from audio input")
                    wav = self.process_audio_input(audio_input)
                    embedding = np.asarray(self._encoder.embed_utterance(wav))

                    np.save(embedding_path, embedding)
                    _LOGGER.debug(f"Embedding cached to {embedding_path}")

                self._reference_embeddings[user_id] = embedding
                _LOGGER.info(f"Successfully trained voice sample for user: {user_id}")

            except Exception as error:
                _LOGGER.error(
                    f"Error processing voice sample for user {user_id}: {error}"
                )
                continue

        if self._reference_embeddings:
            self._is_trained = True
            _LOGGER.info(
                f"Training completed for {len(self._reference_embeddings)} users"
            )
            return TrainingResult(
                status="success",
                trained_users=list(self._reference_embeddings.keys()),
                count=len(self._reference_embeddings),
            )
        else:
            self._is_trained = False
            raise ValueError("No valid voice samples processed")

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        """Recognize speaker from audio data.

        Args:
            request: Recognition request with audio input

        Returns:
            RecognitionResult with user_id, confidence, and all scores
        """
        if not self._is_trained or not self._reference_embeddings:
            raise RuntimeError("Model not trained")

        wav = self.process_audio_input(request.audio)
        chunk_embedding = self._encoder.embed_utterance(wav)

        scores: dict[str, float] = {}
        for user_id, reference_embedding in self._reference_embeddings.items():
            similarity = float(np.dot(reference_embedding, chunk_embedding))
            scores[user_id] = similarity

        if not scores:
            raise RuntimeError("No scores calculated")

        best_user = max(scores, key=lambda user: scores[user])
        best_score = scores[best_user]

        _LOGGER.debug(f"Recognition scores: {scores}")

        return RecognitionResult(
            user_id=best_user, confidence=best_score, all_scores=scores
        )


recognizer = SpeakerRecognizer(config=config)
