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
    CONF_BACKEND_URL,
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_MIN_CONFIDENCE,
    CONF_SAMPLES,
    CONF_STT_ENTITY,
    CONF_USER,
    CONF_VOICE_SAMPLES,
    DEFAULT_BACKEND_URL,
    DEFAULT_MIN_CONFIDENCE,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_MAIN,
    ENTRY_TYPE_STT,
)


async def _build_voice_samples_schema(
    hass: HomeAssistant, default_samples: list | None = None
) -> selector.ObjectSelector:
    """Build the voice samples selector schema."""
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


def _get_main_config_entry(hass: HomeAssistant) -> ConfigEntry | None:
    """Get the main config entry if it exists."""
    entries = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_MAIN:
            return entry
    return None


class SpeakerRecognitionConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Speaker Recognition."""

    VERSION = 2
    MINOR_VERSION = 0

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        main_entry = _get_main_config_entry(self.hass)

        if main_entry is None:
            return await self.async_step_main(user_input)

        return await self.async_step_menu()

    async def async_step_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show menu to add STT or Conversation proxy."""
        return self.async_show_menu(
            step_id="menu",
            menu_options=["add_stt", "add_conversation"],
        )

    async def async_step_main(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle main configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            await self.async_set_unique_id(ENTRY_TYPE_MAIN)
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title="Speaker Recognition",
                data={
                    CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
                    CONF_BACKEND_URL: user_input[CONF_BACKEND_URL],
                },
                options={
                    CONF_VOICE_SAMPLES: user_input.get(CONF_VOICE_SAMPLES, []),
                },
            )

        voice_samples_selector = await _build_voice_samples_schema(self.hass)

        return self.async_show_form(
            step_id="main",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_BACKEND_URL, default=DEFAULT_BACKEND_URL
                    ): selector.TextSelector(),
                    vol.Optional(
                        CONF_VOICE_SAMPLES, default=[]
                    ): voice_samples_selector,
                }
            ),
            errors=errors,
        )

    async def async_step_add_stt(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add STT proxy entity."""
        errors: dict[str, str] = {}

        if user_input is not None:
            if not user_input[CONF_STT_ENTITY].startswith("stt."):
                errors["base"] = "not_stt_entity"
            else:
                stt_entity = user_input[CONF_STT_ENTITY]
                await self.async_set_unique_id(f"{ENTRY_TYPE_STT}_{stt_entity}")
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=f"STT: {stt_entity.split('.', 1)[-1]}",
                    data={
                        CONF_ENTRY_TYPE: ENTRY_TYPE_STT,
                        CONF_STT_ENTITY: stt_entity,
                    },
                )

        return self.async_show_form(
            step_id="add_stt",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_STT_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=Platform.STT,
                        ),
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_add_conversation(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add Conversation proxy entity."""
        errors: dict[str, str] = {}

        if user_input is not None:
            if not user_input[CONF_CONVERSATION_ENTITY].startswith("conversation."):
                errors["base"] = "not_conversation_entity"
            else:
                conversation_entity = user_input[CONF_CONVERSATION_ENTITY]
                await self.async_set_unique_id(
                    f"{ENTRY_TYPE_CONVERSATION}_{conversation_entity}"
                )
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=f"Conversation: {conversation_entity.split('.', 1)[-1]}",
                    data={
                        CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION,
                        CONF_CONVERSATION_ENTITY: conversation_entity,
                        CONF_MIN_CONFIDENCE: user_input[CONF_MIN_CONFIDENCE],
                    },
                )

        return self.async_show_form(
            step_id="add_conversation",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_CONVERSATION_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="conversation",
                        ),
                    ),
                    vol.Required(
                        CONF_MIN_CONFIDENCE, default=DEFAULT_MIN_CONFIDENCE
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
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
        entry_type = self.config_entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_MAIN)

        if entry_type == ENTRY_TYPE_MAIN:
            return await self.async_step_main_options(user_input)
        if entry_type == ENTRY_TYPE_STT:
            return await self.async_step_stt_options(user_input)
        return await self.async_step_conversation_options(user_input)

    async def async_step_main_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage main config options."""
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={
                    CONF_BACKEND_URL: user_input[CONF_BACKEND_URL],
                    CONF_VOICE_SAMPLES: user_input.get(CONF_VOICE_SAMPLES, []),
                },
            )

        current_url = self.config_entry.data.get(CONF_BACKEND_URL, DEFAULT_BACKEND_URL)
        current_voice_samples = self.config_entry.options.get(CONF_VOICE_SAMPLES, [])

        voice_samples_selector = await _build_voice_samples_schema(
            self.hass, current_voice_samples
        )

        return self.async_show_form(
            step_id="main_options",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_BACKEND_URL, default=current_url
                    ): selector.TextSelector(),
                    vol.Optional(
                        CONF_VOICE_SAMPLES, default=current_voice_samples
                    ): voice_samples_selector,
                }
            ),
        )

    async def async_step_stt_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage STT proxy options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            if not user_input[CONF_STT_ENTITY].startswith("stt."):
                errors["base"] = "not_stt_entity"
            else:
                return self.async_create_entry(
                    title="",
                    data={
                        CONF_STT_ENTITY: user_input[CONF_STT_ENTITY],
                    },
                )

        current_stt_entity = self.config_entry.data.get(CONF_STT_ENTITY)

        return self.async_show_form(
            step_id="stt_options",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_STT_ENTITY, default=current_stt_entity
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=Platform.STT,
                        ),
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_conversation_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage Conversation proxy options."""
        errors: dict[str, str] = {}

        if user_input is not None:
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

        current_conversation_entity = self.config_entry.data.get(
            CONF_CONVERSATION_ENTITY
        )
        current_min_confidence = self.config_entry.data.get(
            CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE
        )

        return self.async_show_form(
            step_id="conversation_options",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CONVERSATION_ENTITY, default=current_conversation_entity
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="conversation",
                        ),
                    ),
                    vol.Required(
                        CONF_MIN_CONFIDENCE, default=current_min_confidence
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
            errors=errors,
        )
