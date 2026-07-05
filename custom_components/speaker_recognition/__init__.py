"""The Speaker Recognition integration."""

from __future__ import annotations

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback

from .const import (
    CONF_BACKEND_URL,
    CONF_ENTRY_TYPE,
    CONF_VOICE_SAMPLES,
    DEFAULT_BACKEND_URL,
    ENTRY_TYPE_MAIN,
    ENTRY_TYPE_STT,
)
from .recognition import SpeakerRecognition

type SpeakerRecognitionConfigEntry = ConfigEntry[SpeakerRecognition]

TRAINING_NOTIFICATION_ID = "speaker_recognition_training"


@callback
def _async_update_training_notification(
    hass: HomeAssistant, recognition: SpeakerRecognition
) -> None:
    """Notify the user if the last training attempt failed; clear it otherwise.

    Training happens in the background, so without this a failure (backend
    unreachable, audio files not found, ...) would only appear in the logs.
    """
    if recognition.last_train_error:
        persistent_notification.async_create(
            hass,
            (
                "Speaker Recognition could not train the configured voices:\n\n"
                f"{recognition.last_train_error}"
            ),
            title="Speaker Recognition",
            notification_id=TRAINING_NOTIFICATION_ID,
        )
    else:
        persistent_notification.async_dismiss(hass, TRAINING_NOTIFICATION_ID)


def _get_main_entry(hass: HomeAssistant) -> ConfigEntry | None:
    """Get the main config entry."""
    entries = hass.config_entries.async_entries(__name__.rsplit(".", maxsplit=1)[-1])
    for entry in entries:
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_MAIN:
            return entry
    return None


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Speaker Recognition from a config entry."""
    entry_type = entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_MAIN)

    if entry_type == ENTRY_TYPE_MAIN:
        return await async_setup_main_entry(hass, entry)
    if entry_type == ENTRY_TYPE_STT:
        return await async_setup_stt_entry(hass, entry)
    return await async_setup_conversation_entry(hass, entry)


async def async_setup_main_entry(
    hass: HomeAssistant, entry: SpeakerRecognitionConfigEntry
) -> bool:
    """Set up main config entry."""
    backend_url = entry.data.get(CONF_BACKEND_URL, DEFAULT_BACKEND_URL)
    voice_samples = entry.options.get(CONF_VOICE_SAMPLES, [])

    recognition = SpeakerRecognition(hass, voice_samples, backend_url)

    if voice_samples:
        await recognition.async_train()
        _async_update_training_notification(hass, recognition)

    entry.runtime_data = recognition
    entry.async_on_unload(entry.add_update_listener(async_update_main_listener))

    return True


async def async_setup_stt_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up STT proxy entry."""
    main_entry = _get_main_entry(hass)
    if main_entry is None:
        return False

    await hass.config_entries.async_forward_entry_setups(entry, [Platform.STT])
    entry.async_on_unload(entry.add_update_listener(async_update_stt_listener))

    return True


async def async_setup_conversation_entry(
    hass: HomeAssistant, entry: ConfigEntry
) -> bool:
    """Set up Conversation proxy entry."""
    main_entry = _get_main_entry(hass)
    if main_entry is None:
        return False

    await hass.config_entries.async_forward_entry_setups(entry, [Platform.CONVERSATION])
    entry.async_on_unload(entry.add_update_listener(async_update_conversation_listener))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    entry_type = entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_MAIN)

    if entry_type == ENTRY_TYPE_MAIN:
        return True

    platforms = (
        [Platform.STT] if entry_type == ENTRY_TYPE_STT else [Platform.CONVERSATION]
    )
    return await hass.config_entries.async_unload_platforms(entry, platforms)


async def async_update_main_listener(
    hass: HomeAssistant, entry: SpeakerRecognitionConfigEntry
) -> None:
    """Handle main config options update."""
    voice_samples = entry.options.get(CONF_VOICE_SAMPLES, [])
    entry.runtime_data.update_voice_samples(voice_samples)

    if voice_samples:
        await entry.runtime_data.async_train()

    await hass.config_entries.async_reload(entry.entry_id)


async def async_update_stt_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle STT proxy options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_update_conversation_listener(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Handle Conversation proxy options update."""
    await hass.config_entries.async_reload(entry.entry_id)
