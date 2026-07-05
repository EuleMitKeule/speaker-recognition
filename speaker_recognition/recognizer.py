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
    StatusResponse,
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

    def load_embeddings(self) -> bool:
        """Load cached embeddings from disk.

        Returns:
            True if at least one embedding was loaded, False otherwise.
        """
        loaded_embeddings: dict[str, NDArray[np.float32]] = {}

        if not self._embeddings_directory.exists():
            self._reference_embeddings = {}
            self._is_trained = False
            return False

        suffix = "_embedding.npy"
        for embedding_path in sorted(self._embeddings_directory.glob(f"*{suffix}")):
            user_id = embedding_path.name[: -len(suffix)]
            if not user_id:
                _LOGGER.warning("Skipping embedding with empty user id: %s", embedding_path)
                continue

            try:
                loaded_embeddings[user_id] = np.asarray(
                    np.load(embedding_path, allow_pickle=False), dtype=np.float32
                )
            except Exception as error:
                _LOGGER.warning("Could not load embedding %s: %s", embedding_path, error)

        self._reference_embeddings = loaded_embeddings
        self._is_trained = bool(loaded_embeddings)

        if self._is_trained:
            _LOGGER.info("Loaded %d cached embeddings", len(loaded_embeddings))
        else:
            _LOGGER.info("No cached embeddings found")

        return self._is_trained

    def status(self) -> StatusResponse:
        """Return current backend training status."""
        users = sorted(self._reference_embeddings)
        return StatusResponse(
            trained=self._is_trained and bool(users),
            users=users,
            embeddings_count=len(users),
            embeddings_directory=str(self._embeddings_directory),
        )

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

        samples_by_user: dict[str, list[AudioInput]] = {}
        for sample in request.voice_samples:
            samples_by_user.setdefault(sample.user, []).append(sample.audio)

        _LOGGER.info(
            "Training %d speakers from %d voice samples",
            len(samples_by_user),
            len(request.voice_samples),
        )

        trained_embeddings: dict[str, NDArray[np.float32]] = {}

        for user_id, audio_inputs in samples_by_user.items():
            user_embeddings: list[NDArray[np.float32]] = []
            _LOGGER.info(
                "Processing %d voice samples for user: %s",
                len(audio_inputs),
                user_id,
            )

            for index, audio_input in enumerate(audio_inputs, start=1):
                try:
                    _LOGGER.debug(
                        "Creating embedding from sample %d/%d for user: %s",
                        index,
                        len(audio_inputs),
                        user_id,
                    )
                    wav = self.process_audio_input(audio_input)
                    user_embeddings.append(
                        np.asarray(self._encoder.embed_utterance(wav))
                    )
                except Exception as error:
                    _LOGGER.error(
                        "Error processing sample %d/%d for user %s: %s",
                        index,
                        len(audio_inputs),
                        user_id,
                        error,
                    )
                    continue

            if not user_embeddings:
                _LOGGER.error("No valid voice samples for user: %s", user_id)
                continue

            embedding = np.mean(user_embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            trained_embeddings[user_id] = embedding
            _LOGGER.info(
                "Successfully trained user %s from %d/%d samples",
                user_id,
                len(user_embeddings),
                len(audio_inputs),
            )

        if trained_embeddings:
            self._embeddings_directory.mkdir(parents=True, exist_ok=True)

            for embedding_path in self._embeddings_directory.glob("*_embedding.npy"):
                try:
                    embedding_path.unlink()
                    _LOGGER.debug("Removed stale embedding: %s", embedding_path)
                except OSError as error:
                    _LOGGER.warning(
                        "Could not remove stale embedding %s: %s",
                        embedding_path,
                        error,
                    )
                    continue

            for user_id, embedding in trained_embeddings.items():
                embedding_path = self._embeddings_directory / f"{user_id}_embedding.npy"
                np.save(embedding_path, embedding)
                _LOGGER.debug("Embedding cached to %s", embedding_path)

            self._reference_embeddings = trained_embeddings
            self._is_trained = True
            _LOGGER.info("Training completed for %d users", len(trained_embeddings))
            return TrainingResult(
                status="success",
                trained_users=list(self._reference_embeddings.keys()),
                count=len(self._reference_embeddings),
            )
        else:
            self._reference_embeddings = {}
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
