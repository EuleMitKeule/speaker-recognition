"""Speaker recognition logic."""

import base64
import logging
from collections import defaultdict
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

# Minimum audio duration in seconds. Resemblyzer's embed_utterance uses a
# sliding window of ~1.6 s by default; anything shorter produces unreliable
# embeddings. Samples below this threshold are skipped with a warning.
MIN_AUDIO_DURATION_SEC = 1.0


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

    def load_embeddings(self) -> bool:
        """Load cached embeddings from disk (for startup without retraining).

        Returns:
            True if at least one embedding was loaded, False otherwise.
        """
        self._reference_embeddings = {}

        if not self._embeddings_directory.exists():
            return False

        for npy_path in sorted(self._embeddings_directory.glob("*_embedding.npy")):
            user_id = npy_path.stem.removesuffix("_embedding")
            try:
                loaded = np.load(npy_path, allow_pickle=False)
                self._reference_embeddings[user_id] = np.asarray(loaded)
                _LOGGER.info(f"Loaded cached embedding for user: {user_id}")
            except Exception as e:
                _LOGGER.error(f"Failed to load embedding {npy_path}: {e}")
                continue

        if self._reference_embeddings:
            self._is_trained = True
            _LOGGER.info(
                f"Loaded {len(self._reference_embeddings)} cached embeddings"
            )
            return True

        return False

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Train the speaker recognition model.

        Groups all voice samples by user, generates an embedding for each
        sample, then averages them into a single reference embedding per
        speaker. This replaces any cached embeddings on disk.

        Args:
            request: Training request with voice samples

        Returns:
            TrainingResult with status, trained users and count
        """
        if not request.voice_samples:
            raise ValueError("No voice samples provided")

        self._embeddings_directory.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Clear stale cached embeddings ---
        for old_npy in self._embeddings_directory.glob("*_embedding.npy"):
            old_npy.unlink()
            _LOGGER.debug(f"Removed stale embedding: {old_npy}")

        # --- Step 2: Group samples by user ---
        samples_by_user: dict[str, list[AudioInput]] = defaultdict(list)
        for sample in request.voice_samples:
            samples_by_user[sample.user].append(sample.audio)

        _LOGGER.info(
            f"Training {len(samples_by_user)} speakers from "
            f"{len(request.voice_samples)} total samples"
        )

        # --- Step 3: Generate averaged embedding per user ---
        self._reference_embeddings = {}

        for user_id, audio_inputs in samples_by_user.items():
            user_embeddings: list[NDArray[np.float32]] = []

            for i, audio_input in enumerate(audio_inputs, 1):
                try:
                    wav = self.process_audio_input(audio_input)

                    # Check minimum duration
                    duration_sec = len(wav) / 16000.0
                    if duration_sec < MIN_AUDIO_DURATION_SEC:
                        _LOGGER.warning(
                            f"Skipping sample {i}/{len(audio_inputs)} for "
                            f"{user_id}: too short ({duration_sec:.2f}s < "
                            f"{MIN_AUDIO_DURATION_SEC}s)"
                        )
                        continue

                    emb = np.asarray(self._encoder.embed_utterance(wav))
                    user_embeddings.append(emb)
                    _LOGGER.debug(
                        f"Embedded sample {i}/{len(audio_inputs)} for {user_id} "
                        f"({duration_sec:.1f}s)"
                    )

                except Exception as e:
                    _LOGGER.error(
                        f"Error processing sample {i}/{len(audio_inputs)} "
                        f"for {user_id}: {e}"
                    )
                    continue

            if not user_embeddings:
                _LOGGER.error(f"No valid embeddings for user {user_id}, skipping")
                continue

            # Average all embeddings for this user
            averaged = np.mean(user_embeddings, axis=0).astype(np.float32)
            # Normalize to unit length (cosine similarity via dot product)
            norm = np.linalg.norm(averaged)
            if norm > 0:
                averaged = averaged / norm

            self._reference_embeddings[user_id] = averaged

            # Cache the averaged embedding
            embedding_path = self._embeddings_directory / f"{user_id}_embedding.npy"
            np.save(embedding_path, averaged)

            _LOGGER.info(
                f"Trained {user_id}: averaged {len(user_embeddings)}/{len(audio_inputs)} "
                f"samples"
            )

        # --- Step 4: Report results ---
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
