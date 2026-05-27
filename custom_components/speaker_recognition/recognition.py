"""Speaker recognition module."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

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

LOCAL_MEDIA_PREFIX = "media-source://media_source/local/"


def _resolve_local_media_path(hass: HomeAssistant, media_content_id: str) -> Path:
    """Resolve a Home Assistant local media-source ID to a real filesystem path."""
    if not media_content_id.startswith(LOCAL_MEDIA_PREFIX):
        raise ValueError(f"Unsupported media_content_id format: {media_content_id}")

    relative_path = unquote(
        media_content_id.removeprefix(LOCAL_MEDIA_PREFIX)
    ).lstrip("/")

    if not relative_path:
        raise ValueError(f"Empty local media path in media_content_id: {media_content_id}")

    candidates = [
        Path("/media") / relative_path,
        Path(hass.config.path("media")) / relative_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find selected media file. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _extract_media_object(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Extract one media selector object from a configured voice sample.

    The intended shape is:

        {
            "user": "...",
            "samples": {
                "media_content_id": "...",
                "media_content_type": "...",
            },
        }

    Older/broken saved data may contain "samples" as a one-item list because
    the old MediaSelector used multiple=True. This helper accepts that shape
    defensively for migration/testing, but rejects zero or multiple media files.
    """
    media = sample.get("samples")

    if isinstance(media, list):
        if len(media) != 1:
            _LOGGER.warning(
                "Expected exactly one voice sample media object, got %d",
                len(media),
            )
            return None

        media = media[0]

    if not isinstance(media, dict):
        _LOGGER.warning("Invalid voice sample media object: %r", media)
        return None

    return media


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
            hass: Home Assistant instance.
            voice_samples: List of voice samples with user and audio file info.
            base_url: Base URL of the speaker recognition service.
        """
        self.hass = hass
        self.voice_samples = voice_samples
        self.base_url = base_url
        self._trained = False
        self._client = SpeakerRecognitionClient(base_url=base_url, timeout=300.0)


    async def _async_train_request(self, request: TrainingRequest):
        """Call the speaker-recognition backend training API off the HA event loop."""

        def _train_in_thread():
            client = SpeakerRecognitionClient(base_url=self.base_url, timeout=300.0)
            return asyncio.run(client.train(request))

        return await self.hass.async_add_executor_job(_train_in_thread)

    async def _async_recognize_request(self, request: RecognitionRequest):
        """Call the speaker-recognition backend recognition API off the HA event loop."""

        def _recognize_in_thread():
            client = SpeakerRecognitionClient(base_url=self.base_url, timeout=300.0)
            return asyncio.run(client.recognize(request))

        return await self.hass.async_add_executor_job(_recognize_in_thread)

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
                if not isinstance(sample, dict):
                    _LOGGER.warning("Invalid voice sample entry: %r", sample)
                    continue

                user_id = sample.get("user")
                if not isinstance(user_id, str) or not user_id:
                    _LOGGER.warning("Invalid voice sample user: %r", user_id)
                    continue

                media = _extract_media_object(sample)
                if media is None:
                    _LOGGER.warning(
                        "Skipping invalid voice sample media for user %s",
                        user_id,
                    )
                    continue

                media_id = media.get("media_content_id", "")
                if not isinstance(media_id, str):
                    _LOGGER.warning(
                        "Invalid media_content_id for user %s: %r",
                        user_id,
                        media_id,
                    )
                    continue

                if media_id.startswith(LOCAL_MEDIA_PREFIX):
                    full_path = _resolve_local_media_path(self.hass, media_id)

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
                    _LOGGER.warning(
                        "Unsupported media_content_id format for user %s: %s",
                        user_id,
                        media_id,
                    )
                    continue

            if not voice_sample_models:
                _LOGGER.warning("No valid training samples prepared")
                self._trained = False
                return

            request = TrainingRequest(voice_samples=voice_sample_models)
            result = await self._async_train_request(request)

        except (OSError, ValueError, TypeError) as error:
            _LOGGER.error("Error during training: %s", error)
            self._trained = False

        else:
            self._trained = True
            _LOGGER.info("Speaker recognition training completed")

    async def async_recognize(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> RecognitionResult | None:
        """Recognize speaker from audio data.

        Args:
            audio_data: Raw audio data to analyze.
            sample_rate: Audio sample rate.

        Returns:
            RecognitionResult if a speaker is recognized, None otherwise.
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
            result = await self._async_recognize_request(request)

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
            voice_samples: New list of voice samples.
        """
        self.voice_samples = voice_samples
        self._trained = False
        _LOGGER.info("Voice samples updated, retraining required")
