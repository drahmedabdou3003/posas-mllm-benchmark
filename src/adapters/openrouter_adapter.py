"""
OpenRouter adapter — handles Qwen3.7-Plus and GLM-5V-Turbo.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1.
Both Qwen and GLM are thinking/reasoning models: they generate an internal
chain-of-thought before writing the final JSON. The reasoning text is billed
as completion tokens and included in usage.completion_tokens_details.reasoning_tokens.

JSON mode (response_format) is deliberately omitted — thinking models on
OpenRouter do not reliably support it and the stage4 validator handles code
fences and brace extraction anyway.
"""

import logging
import os

import openai

from src.adapters.base_adapter import BaseAdapter

logger = logging.getLogger("posas")


class OpenRouterAdapter(BaseAdapter):

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_string = model_config["model_string"]
        self.temperature = model_config.get("temperature", 0)
        self.max_tokens = model_config.get("max_tokens", 2000)
        self._client = None

    def authenticate(self) -> bool:
        api_key = os.environ.get(self.model_config["env_key"])
        if not api_key:
            raise ValueError(
                f"Missing API key: {self.model_config['env_key']} not set in environment"
            )
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/posas-mllm-benchmark",
                "X-Title": "POSAS-MLLM Benchmark",
            },
        )
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
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "model": self.model_string,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0}
        result = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }
        details = getattr(usage, "completion_tokens_details", None)
        if details and getattr(details, "reasoning_tokens", None):
            result["reasoning_tokens"] = details.reasoning_tokens
        return result
