"""Speaker recognition module."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


@dataclass
class SpeakerRecognitionResult:
    """Result of speaker recognition."""

    user_id: str
    confidence: float
    all_scores: dict[str, float]


class SpeakerRecognition:
    """Handle speaker recognition from audio data."""

    def __init__(self, hass: HomeAssistant, voice_samples: list[dict]) -> None:
        """Initialize speaker recognition.

        Args:
            hass: Home Assistant instance
            voice_samples: List of voice samples with user and audio file info
        """
        self.hass = hass
        self.voice_samples = voice_samples
        self._trained = False
        self._encoder = None
        self._reference_embeddings: dict[str, np.ndarray] = {}

    async def async_train(self) -> None:
        """Train the speaker recognition model with configured voice samples."""
        _LOGGER.debug(
            "Training speaker recognition with %d voice samples",
            len(self.voice_samples),
        )

        if not self.voice_samples:
            _LOGGER.warning("No voice samples configured for training")
            self._trained = False
            return

        # Import resemblyzer in executor to avoid blocking
        await self.hass.async_add_executor_job(self._train_sync)

        _LOGGER.info("Speaker recognition training completed")

    def _train_sync(self) -> None:
        """Synchronous training logic (runs in executor)."""
        # Initialize encoder if not already done
        if self._encoder is None:
            _LOGGER.info("Initializing voice encoder...")
            self._encoder = VoiceEncoder()

        self._reference_embeddings = {}

        for sample in self.voice_samples:
            user_id = sample["user"]
            # Get the media path from media_content_id
            # Format is typically: media-source://media_source/local/path/to/file.wav
            media_id = sample["samples"].get("media_content_id", "")

            # Extract actual file path from media_content_id
            # Remove the media-source prefix
            if media_id.startswith("media-source://media_source/local/"):
                # Extract the relative path (e.g., "speaker1_1.mp3")
                relative_path = media_id.replace(
                    "media-source://media_source/local/", ""
                )
                # Build full path: config/media/filename
                full_path = Path(self.hass.config.path("media")) / relative_path
            else:
                _LOGGER.warning("Unsupported media_content_id format: %s", media_id)
                continue
            embedding_path = full_path.parent / f"{full_path.stem}_embedding.npy"

            _LOGGER.info("Processing voice sample for user: %s", user_id)

            try:
                # Load cached embedding if exists
                if embedding_path.exists():
                    _LOGGER.debug("Loading cached embedding from %s", embedding_path)
                    embedding = np.load(embedding_path)
                else:
                    # Create new embedding
                    if not full_path.exists():
                        _LOGGER.error(
                            "Voice sample file not found for user %s: %s",
                            user_id,
                            full_path,
                        )
                        continue

                    _LOGGER.debug("Creating embedding from %s", full_path)
                    wav = preprocess_wav(full_path)
                    embedding = self._encoder.embed_utterance(wav)

                    # Cache the embedding
                    np.save(embedding_path, embedding)
                    _LOGGER.debug("Embedding cached to %s", embedding_path)

                self._reference_embeddings[user_id] = embedding
                _LOGGER.info("Successfully trained voice sample for user: %s", user_id)

            except Exception as e:
                _LOGGER.error(
                    "Error processing voice sample for user %s: %s", user_id, e
                )
                continue

        if self._reference_embeddings:
            self._trained = True
            _LOGGER.info(
                "Training completed for %d users", len(self._reference_embeddings)
            )
        else:
            self._trained = False
            _LOGGER.warning("No valid voice samples processed")

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> SpeakerRecognitionResult | None:
        """Recognize speaker from audio data.

        Args:
            audio_data: Raw audio data to analyze (PCM 16-bit)
            sample_rate: Audio sample rate

        Returns:
            SpeakerRecognitionResult if a speaker is recognized, None otherwise
        """
        if not self._trained:
            _LOGGER.debug("Speaker recognition not trained yet")
            return None

        if not self._reference_embeddings:
            _LOGGER.debug("No reference embeddings available")
            return None

        # Run recognition in executor to avoid blocking
        result = await self.hass.async_add_executor_job(
            self._recognize_sync, audio_data, sample_rate
        )
        return result

    def _recognize_sync(
        self, audio_data: bytes, sample_rate: int
    ) -> SpeakerRecognitionResult | None:
        """Synchronous recognition logic (runs in executor)."""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array_int16 = np.frombuffer(audio_data, dtype=np.int16).copy()

            if audio_array_int16.size == 0:
                return None

            # Convert to float32 normalized to [-1, 1]
            audio_array_float32 = audio_array_int16.astype(np.float32) / 32768.0

            # Preprocess the audio
            wav = preprocess_wav(audio_array_float32, source_sr=sample_rate)

            # Create embedding for the audio chunk
            chunk_embed = self._encoder.embed_utterance(wav)

            # Calculate similarity scores with all registered speakers
            scores: dict[str, float] = {}
            for user_id, ref_embed in self._reference_embeddings.items():
                similarity = float(np.dot(ref_embed, chunk_embed))
                scores[user_id] = similarity

            if not scores:
                return None

            # Find the best match
            best_user = max(scores, key=scores.get)
            best_score = scores[best_user]

            _LOGGER.debug("Recognition scores: %s", scores)

            return SpeakerRecognitionResult(
                user_id=best_user, confidence=best_score, all_scores=scores
            )

        except Exception as e:
            _LOGGER.error("Error during speaker recognition: %s", e)
            return None

    def update_voice_samples(self, voice_samples: list[dict]) -> None:
        """Update voice samples and mark as needing retraining.

        Args:
            voice_samples: New list of voice samples
        """
        self.voice_samples = voice_samples
        self._trained = False
        _LOGGER.info("Voice samples updated, retraining required")
