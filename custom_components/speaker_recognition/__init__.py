"""The Speaker Recognition integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import CONF_HELPER_TYPE, CONF_VOICE_SAMPLES, HELPER_TYPE_STT
from .recognition import SpeakerRecognition

PLATFORMS: list[Platform] = [Platform.STT]

type SpeakerRecognitionConfigEntry = ConfigEntry[SpeakerRecognition]


async def async_setup_entry(
    hass: HomeAssistant, entry: SpeakerRecognitionConfigEntry
) -> bool:
    """Set up Speaker Recognition from a config entry."""
    helper_type = entry.data.get(CONF_HELPER_TYPE, HELPER_TYPE_STT)

    if helper_type == HELPER_TYPE_STT:
        # Get voice samples from options
        voice_samples = entry.options.get(CONF_VOICE_SAMPLES, [])

        # Create speaker recognition instance
        recognition = SpeakerRecognition(hass, voice_samples)

        # Train the model if voice samples are configured
        if voice_samples:
            await recognition.async_train()

        # Store in runtime_data for access by STT entity
        entry.runtime_data = recognition

        await hass.config_entries.async_forward_entry_setups(entry, [Platform.STT])
    else:
        # Conversation helper - no recognition instance needed
        await hass.config_entries.async_forward_entry_setups(
            entry, [Platform.CONVERSATION]
        )

    entry.async_on_unload(entry.add_update_listener(async_update_listener))
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: SpeakerRecognitionConfigEntry
) -> bool:
    """Unload a config entry."""
    helper_type = entry.data.get(CONF_HELPER_TYPE, HELPER_TYPE_STT)
    platforms = [Platform.STT] if helper_type == HELPER_TYPE_STT else ["conversation"]
    return await hass.config_entries.async_unload_platforms(entry, platforms)


async def async_update_listener(
    hass: HomeAssistant, entry: SpeakerRecognitionConfigEntry
) -> None:
    """Handle options update."""
    helper_type = entry.data.get(CONF_HELPER_TYPE, HELPER_TYPE_STT)

    if helper_type == HELPER_TYPE_STT:
        # Update voice samples in the recognition instance
        voice_samples = entry.options.get(CONF_VOICE_SAMPLES, [])
        entry.runtime_data.update_voice_samples(voice_samples)

        # Retrain if needed
        if voice_samples:
            await entry.runtime_data.async_train()

    # Reload to update the entity
    await hass.config_entries.async_reload(entry.entry_id)
