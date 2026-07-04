"""Speaker recognition module."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.audio import (
    media_selector_content_ids,
    read_wav_pcm,
    resolve_local_media_path,
)
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
            media_dirs = [
                Path("/media"),
                Path(self.hass.config.path("media")),
            ]

            for sample in self.voice_samples:
                user_id = sample["user"]
                media_ids = media_selector_content_ids(sample.get("samples"))

                if not media_ids:
                    _LOGGER.warning("No media files configured for user: %s", user_id)
                    continue

                for media_id in media_ids:
                    try:
                        full_path = resolve_local_media_path(media_id, media_dirs)
                        if full_path is None:
                            _LOGGER.warning(
                                "Unsupported media_content_id format: %s", media_id
                            )
                            continue

                        prepared_audio = await self.hass.async_add_executor_job(
                            read_wav_pcm, full_path
                        )
                    except (OSError, ValueError) as error:
                        _LOGGER.warning(
                            "Skipping voice sample %s for user %s: %s",
                            media_id,
                            user_id,
                            error,
                        )
                        continue

                    voice_sample_models.append(
                        VoiceSample(
                            user=user_id,
                            audio=AudioInput(
                                audio_data=prepared_audio.audio_data,
                                sample_rate=prepared_audio.sample_rate,
                            ),
                        )
                    )

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
                result.count,
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
