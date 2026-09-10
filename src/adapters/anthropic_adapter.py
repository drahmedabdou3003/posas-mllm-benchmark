"""
Anthropic adapter — handles Claude Sonnet 5.

Images are sent as base64 in an image content block with media_type="image/jpeg".
No native JSON mode — JSON is enforced via the prompt template alone.
Set omit_temperature: true in model config for models that reject the parameter
(claude-sonnet-5 rejects temperature).
"""

import logging
import os

import anthropic

from src.adapters.base_adapter import BaseAdapter

logger = logging.getLogger("posas")


class AnthropicAdapter(BaseAdapter):

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_string = model_config["model_string"]
        self.temperature = model_config.get("temperature", 0)
        self.max_tokens = model_config.get("max_tokens", 1500)
        self.omit_temperature = model_config.get("omit_temperature", False)
        self._client = None

    def authenticate(self) -> bool:
        api_key = os.environ.get(self.model_config["env_key"])
        if not api_key:
            raise ValueError(
                f"Missing API key: {self.model_config['env_key']} not set in environment"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        return True

    def prepare_payload(self, image_base64: str, prompt: str) -> dict:
        payload = {
            "model": self.model_string,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if not self.omit_temperature:
            payload["temperature"] = self.temperature
        return payload

    def send_request(self, payload: dict) -> dict:
        if self._client is None:
            self.authenticate()
        response = self._client.messages.create(**payload)
        return response

    def extract_response_text(self, raw_response) -> str:
        return raw_response.content[0].text

    def get_model_version(self) -> str:
        return self.model_string

    def get_token_usage(self, raw_response) -> dict:
        return {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
        }
