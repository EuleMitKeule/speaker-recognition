"""The Speaker Recognition integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

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
