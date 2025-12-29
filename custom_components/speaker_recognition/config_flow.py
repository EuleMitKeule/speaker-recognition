"""Config flow for Speaker Recognition integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import selector

from .const import (
    CONF_CONVERSATION_ENTITY,
    CONF_HELPER_TYPE,
    CONF_MIN_CONFIDENCE,
    CONF_SAMPLES,
    CONF_STT_ENTITY,
    CONF_USER,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    HELPER_TYPE_CONVERSATION,
    HELPER_TYPE_STT,
)


async def _build_voice_samples_schema(
    hass: HomeAssistant, default_samples: list | None = None
) -> selector.ObjectSelector:
    """Build the voice samples selector schema."""
    # Get list of users
    users = await hass.auth.async_get_users()
    user_options = [
        selector.SelectOptionDict(value=user.id, label=user.name or user.id)
        for user in users
        if not user.system_generated
    ]

    return selector.ObjectSelector(
        selector.ObjectSelectorConfig(
            fields={
                CONF_USER: {
                    "required": True,
                    "selector": selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=user_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                },
                CONF_SAMPLES: {
                    "required": True,
                    "selector": selector.MediaSelector(
                        selector.MediaSelectorConfig(
                            multiple=True, accept=["audio/wav", "audio/mpeg"]
                        )
                    ),
                },
            },
            multiple=True,
            label_field=CONF_USER,
        )
    )


async def _build_stt_config_schema(
    hass: HomeAssistant,
    stt_entity: str | None = None,
    voice_samples: list | None = None,
) -> vol.Schema:
    """Build the STT configuration schema."""
    voice_samples_selector = await _build_voice_samples_schema(hass, voice_samples)

    return vol.Schema(
        {
            vol.Required(CONF_STT_ENTITY, default=stt_entity): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=Platform.STT,
                ),
            ),
            vol.Optional(
                CONF_VOICE_SAMPLES, default=voice_samples or []
            ): voice_samples_selector,
        }
    )


async def _build_conversation_config_schema(
    hass: HomeAssistant,
    conversation_entity: str | None = None,
    min_confidence: float = 0.7,
) -> vol.Schema:
    """Build the Conversation configuration schema."""
    return vol.Schema(
        {
            vol.Required(
                CONF_CONVERSATION_ENTITY, default=conversation_entity
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="conversation",
                ),
            ),
            vol.Required(
                CONF_MIN_CONFIDENCE, default=min_confidence
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    mode=selector.NumberSelectorMode.SLIDER,
                )
            ),
        }
    )


class SpeakerRecognitionConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Speaker Recognition."""

    VERSION = 1
    MINOR_VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._helper_type: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step - select helper type."""
        if user_input is not None:
            self._helper_type = user_input[CONF_HELPER_TYPE]
            
            if self._helper_type == HELPER_TYPE_STT:
                return await self.async_step_stt()
            return await self.async_step_conversation()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_HELPER_TYPE): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(
                                    value=HELPER_TYPE_STT, label="STT proxy"
                                ),
                                selector.SelectOptionDict(
                                    value=HELPER_TYPE_CONVERSATION,
                                    label="Conversation proxy",
                                ),
                            ],
                            mode=selector.SelectSelectorMode.LIST,
                        )
                    ),
                }
            ),
        )

    async def async_step_stt(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle STT helper configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate that the selected entity is actually an STT entity
            if not user_input[CONF_STT_ENTITY].startswith("stt."):
                errors["base"] = "not_stt_entity"
            else:
                # Create the config entry with both data and options
                return self.async_create_entry(
                    title="Speaker Recognition (STT)",
                    data={
                        CONF_HELPER_TYPE: HELPER_TYPE_STT,
                        CONF_STT_ENTITY: user_input[CONF_STT_ENTITY],
                    },
                    options={
                        CONF_VOICE_SAMPLES: user_input.get(CONF_VOICE_SAMPLES, [])
                    },
                )

        data_schema = await _build_stt_config_schema(self.hass)

        return self.async_show_form(
            step_id="stt",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_conversation(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle Conversation helper configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate that the selected entity is actually a conversation entity
            if not user_input[CONF_CONVERSATION_ENTITY].startswith("conversation."):
                errors["base"] = "not_conversation_entity"
            else:
                # Create the config entry
                return self.async_create_entry(
                    title="Speaker Recognition (Conversation)",
                    data={
                        CONF_HELPER_TYPE: HELPER_TYPE_CONVERSATION,
                        CONF_CONVERSATION_ENTITY: user_input[CONF_CONVERSATION_ENTITY],
                        CONF_MIN_CONFIDENCE: user_input[CONF_MIN_CONFIDENCE],
                    },
                )

        data_schema = await _build_conversation_config_schema(self.hass)

        return self.async_show_form(
            step_id="conversation",
            data_schema=data_schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> SpeakerRecognitionOptionsFlow:
        """Get the options flow for this handler."""
        return SpeakerRecognitionOptionsFlow()


class SpeakerRecognitionOptionsFlow(OptionsFlow):
    """Handle options flow for Speaker Recognition."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        # Check helper type to show appropriate form
        helper_type = self.config_entry.data.get(CONF_HELPER_TYPE, HELPER_TYPE_STT)
        
        if helper_type == HELPER_TYPE_STT:
            return await self.async_step_stt_options(user_input)
        return await self.async_step_conversation_options(user_input)

    async def async_step_stt_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage STT helper options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate that the selected entity is actually an STT entity
            if not user_input[CONF_STT_ENTITY].startswith("stt."):
                errors["base"] = "not_stt_entity"
            else:
                # Simply save whatever voice samples are provided
                # The UI handles the list, we just store it
                return self.async_create_entry(
                    title="",
                    data={
                        CONF_STT_ENTITY: user_input[CONF_STT_ENTITY],
                        CONF_VOICE_SAMPLES: user_input.get(CONF_VOICE_SAMPLES, []),
                    },
                )

        # Get current value from config entry data or options
        current_stt_entity = self.config_entry.options.get(
            CONF_STT_ENTITY, self.config_entry.data.get(CONF_STT_ENTITY)
        )

        # Get existing voice samples
        current_voice_samples = self.config_entry.options.get(CONF_VOICE_SAMPLES, [])

        data_schema = await _build_stt_config_schema(
            self.hass, current_stt_entity, current_voice_samples
        )

        return self.async_show_form(
            step_id="stt_options",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_conversation_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage Conversation helper options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate that the selected entity is actually a conversation entity
            if not user_input[CONF_CONVERSATION_ENTITY].startswith("conversation."):
                errors["base"] = "not_conversation_entity"
            else:
                return self.async_create_entry(
                    title="",
                    data={
                        CONF_CONVERSATION_ENTITY: user_input[CONF_CONVERSATION_ENTITY],
                        CONF_MIN_CONFIDENCE: user_input[CONF_MIN_CONFIDENCE],
                    },
                )

        # Get current values
        current_conversation_entity = self.config_entry.data.get(
            CONF_CONVERSATION_ENTITY
        )
        current_min_confidence = self.config_entry.data.get(CONF_MIN_CONFIDENCE, 0.7)

        data_schema = await _build_conversation_config_schema(
            self.hass, current_conversation_entity, current_min_confidence
        )

        return self.async_show_form(
            step_id="conversation_options",
            data_schema=data_schema,
            errors=errors,
        )
