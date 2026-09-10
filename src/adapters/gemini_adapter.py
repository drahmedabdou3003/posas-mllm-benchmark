"""
Google Gemini adapter — handles Gemini 3.1 Pro.

Uses the current google-genai SDK (replaces deprecated google-generativeai).
Images are sent as inline bytes. JSON output is encouraged via
response_mime_type="application/json" in the generation config.
Google API errors are mapped to HTTP-style status codes for the retry handler.
"""

import base64
import logging
import os

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

from src.adapters.base_adapter import BaseAdapter

logger = logging.getLogger("posas")

_GOOGLE_ERROR_CODES = {
    google_exceptions.ResourceExhausted: 429,
    google_exceptions.InternalServerError: 500,
    google_exceptions.ServiceUnavailable: 503,
    google_exceptions.BadGateway: 502,
    google_exceptions.InvalidArgument: 400,
    google_exceptions.PermissionDenied: 403,
    google_exceptions.Unauthenticated: 401,
    google_exceptions.NotFound: 404,
}


class _GoogleError(Exception):
    """Wrapper that exposes a status_code for the retry handler."""
    def __init__(self, status_code: int, original: Exception):
        self.status_code = status_code
        super().__init__(str(original))


class GeminiAdapter(BaseAdapter):

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_string = model_config["model_string"]
        self.temperature = model_config.get("temperature", 0)
        self.max_tokens = model_config.get("max_tokens", 1500)
        self._client = None

    def authenticate(self) -> bool:
        api_key = os.environ.get(self.model_config["env_key"])
        if not api_key:
            raise ValueError(
                f"Missing API key: {self.model_config['env_key']} not set in environment"
            )
        self._client = genai.Client(api_key=api_key)
        return True

    def prepare_payload(self, image_base64: str, prompt: str) -> dict:
        return {
            "prompt": prompt,
            "image_base64": image_base64,
        }

    def send_request(self, payload: dict) -> object:
        if self._client is None:
            self.authenticate()

        image_bytes = base64.b64decode(payload["image_base64"])
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_string,
                contents=[payload["prompt"], image_part],
                config=config,
            )
        except tuple(_GOOGLE_ERROR_CODES.keys()) as e:
            for exc_type, code in _GOOGLE_ERROR_CODES.items():
                if isinstance(e, exc_type):
                    raise _GoogleError(code, e)
            raise _GoogleError(500, e)

        return response

    def extract_response_text(self, raw_response) -> str:
        try:
            candidate = raw_response.candidates[0]
            finish_reason = str(candidate.finish_reason)
            if "STOP" not in finish_reason:
                partial = raw_response.text or ""
                raise _GoogleError(
                    500,
                    Exception(
                        f"Generation stopped early (finish_reason={finish_reason}). "
                        f"Partial response ({len(partial)} chars): {partial[:120]!r}"
                    ),
                )
        except (AttributeError, IndexError):
            pass
        return raw_response.text

    def get_model_version(self) -> str:
        return self.model_string

    def get_token_usage(self, raw_response) -> dict:
        meta = raw_response.usage_metadata
        return {
            "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        }
