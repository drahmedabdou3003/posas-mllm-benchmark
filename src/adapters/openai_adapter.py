"""
OpenAI adapter — handles GPT-5.4.

Images are sent as base64 in an image_url content block with detail="high".
JSON mode is enforced via response_format={"type": "json_object"}.
Note: GPT-5.x requires max_completion_tokens (not the legacy max_tokens).
"""

import logging
import os

import openai

from src.adapters.base_adapter import BaseAdapter

logger = logging.getLogger("posas")


class OpenAIAdapter(BaseAdapter):

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_string = model_config["model_string"]
        self.temperature = model_config.get("temperature", 0)
        self.max_tokens = model_config.get("max_tokens", 1500)
        self.seed = model_config.get("seed", 42)
        self.image_detail = model_config.get("image_detail", "high")
        self._client = None

    def authenticate(self) -> bool:
        api_key = os.environ.get(self.model_config["env_key"])
        if not api_key:
            raise ValueError(
                f"Missing API key: {self.model_config['env_key']} not set in environment"
            )
        self._client = openai.OpenAI(api_key=api_key)
        return True

    def prepare_payload(self, image_base64: str, prompt: str) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": self.image_detail,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "model": self.model_string,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "seed": self.seed,
            "response_format": {"type": "json_object"},
        }

    def send_request(self, payload: dict) -> dict:
        if self._client is None:
            self.authenticate()
        response = self._client.chat.completions.create(**payload)
        return response

    def extract_response_text(self, raw_response) -> str:
        return raw_response.choices[0].message.content

    def get_model_version(self) -> str:
        return self.model_string

    def get_token_usage(self, raw_response) -> dict:
        usage = raw_response.usage
        return {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
