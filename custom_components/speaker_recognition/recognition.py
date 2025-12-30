"""Speaker recognition module."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.models import (
    AudioInput,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    VoiceSample,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

DEFAULT_ADDON_URL = "http://localhost:8099"


class SpeakerRecognition:
    """Handle speaker recognition from audio data."""

    def __init__(
        self,
        hass: HomeAssistant,
        voice_samples: list[dict],
        base_url: str = DEFAULT_ADDON_URL,
    ) -> None:
        """Initialize speaker recognition.

        Args:
            hass: Home Assistant instance
            voice_samples: List of voice samples with user and audio file info
            base_url: Base URL of the speaker recognition service
        """
        self.hass = hass
        self.voice_samples = voice_samples
        self._trained = False
        self._client = SpeakerRecognitionClient(base_url=base_url, timeout=300.0)

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

        try:
            voice_sample_models = []
            for sample in self.voice_samples:
                user_id = sample["user"]
                media_id = sample["samples"].get("media_content_id", "")

                if media_id.startswith("media-source://media_source/local/"):
                    relative_path = media_id.replace(
                        "media-source://media_source/local/", ""
                    )
                    full_path = Path(self.hass.config.path("media")) / relative_path

                    # Read the audio file
                    audio_data = await self.hass.async_add_executor_job(
                        full_path.read_bytes
                    )
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                    voice_sample_models.append(
                        VoiceSample(
                            user=user_id,
                            audio=AudioInput(
                                audio_data=audio_base64,
                                sample_rate=16000,
                            ),
                        )
                    )
                else:
                    _LOGGER.warning("Unsupported media_content_id format: %s", media_id)
                    continue

            if not voice_sample_models:
                _LOGGER.warning("No valid training samples prepared")
                self._trained = False
                return

            request = TrainingRequest(voice_samples=voice_sample_models)
            result = await self._client.train(request)

        except (OSError, ValueError, TypeError) as error:
            _LOGGER.error("Error during training: %s", error)
            self._trained = False
        else:
            self._trained = True
            _LOGGER.info(
                "Speaker recognition training completed: %d users trained",
                result.users_trained,
            )

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult | None:
        """Recognize speaker from audio data.

        Args:
            audio_data: Raw audio data to analyze (PCM 16-bit)
            sample_rate: Audio sample rate

        Returns:
            RecognitionResult if a speaker is recognized, None otherwise
        """
        if not self._trained:
            _LOGGER.debug("Speaker recognition not trained yet")
            return None

        try:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            request = RecognitionRequest(
                audio=AudioInput(
                    audio_data=audio_base64,
                    sample_rate=sample_rate,
                )
            )

            result = await self._client.recognize(request)

        except (OSError, ValueError, TypeError) as error:
            _LOGGER.error("Error during recognition: %s", error)
            return None
        else:
            _LOGGER.debug(
                "Recognition result: user=%s, confidence=%.2f",
                result.user_id,
                result.confidence,
            )

            return result

    def update_voice_samples(self, voice_samples: list[dict]) -> None:
        """Update voice samples and mark as needing retraining.

        Args:
            voice_samples: New list of voice samples
        """
        self.voice_samples = voice_samples
        self._trained = False
        _LOGGER.info("Voice samples updated, retraining required")
