"""An unavailable backend must delay speaker recognition, never break speech-to-text."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from homeassistant.components.stt import SpeechResult, SpeechResultState
from homeassistant.exceptions import ConfigEntryNotReady

from custom_components import speaker_recognition as integration
from custom_components.speaker_recognition import stt as stt_module
from custom_components.speaker_recognition.const import CONF_VOICE_SAMPLES
from custom_components.speaker_recognition.recognition import SpeakerRecognition

pytestmark = pytest.mark.asyncio

BACKEND_ERRORS = [
    httpx.ConnectError("All connection attempts failed"),
    httpx.ReadTimeout("timed out"),
    httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock()),
]


def _main_entry():
    samples = [{"user": "u1", "samples": {"media_content_id": "media-source://media_source/local/s.mp3"}}]
    return MagicMock(data={}, options={CONF_VOICE_SAMPLES: samples}, spec_set=["data", "options", "runtime_data", "async_on_unload", "add_update_listener"])


def _recognition(hass, client):
    with patch("custom_components.speaker_recognition.recognition.SpeakerRecognitionClient", return_value=client):
        return SpeakerRecognition(hass, [], "http://backend")


@pytest.mark.parametrize("error", BACKEND_ERRORS)
async def test_setup_retries_when_backend_unavailable(error):
    with patch.object(integration, "SpeakerRecognition") as recognition_class:
        recognition_class.return_value.async_train = AsyncMock(side_effect=error)
        with pytest.raises(ConfigEntryNotReady):
            await integration.async_setup_main_entry(MagicMock(), _main_entry())


async def test_train_propagates_backend_error():
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(return_value=b"\x00")
    hass.config.path.return_value = "/media"
    client = MagicMock(train=AsyncMock(side_effect=httpx.ConnectError("down")))
    recognition = _recognition(hass, client)
    recognition.voice_samples = _main_entry().options[CONF_VOICE_SAMPLES]
    with pytest.raises(httpx.HTTPError):
        await recognition.async_train()


@pytest.mark.parametrize("error", BACKEND_ERRORS)
async def test_recognize_returns_none_when_backend_unavailable(error):
    client = MagicMock(recognize=AsyncMock(side_effect=error))
    recognition = _recognition(MagicMock(), client)
    recognition._trained = True
    assert await recognition.async_recognize(b"\x00\x01") is None


async def test_options_update_of_unloaded_entry_only_reloads():
    hass = MagicMock()
    hass.config_entries.async_reload = AsyncMock()
    entry = SimpleNamespace(entry_id="main", options={CONF_VOICE_SAMPLES: [{"user": "u1"}]})
    await integration.async_update_main_listener(hass, entry)
    hass.config_entries.async_reload.assert_awaited_once_with("main")


async def test_stt_returns_text_when_main_entry_not_loaded():
    entity = stt_module.SpeakerRecognitionSTTEntity.__new__(stt_module.SpeakerRecognitionSTTEntity)
    entity.hass = MagicMock()
    entity.hass.data = {}
    entity._stt_entity_id = "stt.test"
    entity._skip_without_text = False
    # A ConfigEntry whose setup failed has no runtime_data attribute at all.
    entity._main_entry = SimpleNamespace()

    source = MagicMock()

    async def fake_stt(metadata, stream):
        async for _ in stream:
            pass
        return SpeechResult("która godzina", SpeechResultState.SUCCESS)

    source.async_process_audio_stream = fake_stt

    async def audio():
        yield b"\x00\x01" * 100

    with patch.object(stt_module, "async_get_speech_to_text_entity", return_value=source):
        result = await entity.async_process_audio_stream(MagicMock(sample_rate=16000), audio())

    assert result.text == "która godzina"
