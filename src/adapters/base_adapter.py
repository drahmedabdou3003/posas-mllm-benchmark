"""
Abstract base class for all model adapters.

Every adapter must implement these six methods so Stage 3 can call
any model through a uniform interface.
"""

from abc import ABC, abstractmethod


class BaseAdapter(ABC):

    @abstractmethod
    def authenticate(self) -> bool:
        """Validate that the API key is present and the client initialises."""
        pass

    @abstractmethod
    def prepare_payload(self, image_base64: str, prompt: str) -> dict:
        """Format image + prompt per this API's specification."""
        pass

    @abstractmethod
    def send_request(self, payload: dict) -> dict:
        """Make the API call and return the raw response object as a dict."""
        pass

    @abstractmethod
    def extract_response_text(self, raw_response: dict) -> str:
        """Pull the text content out of the raw API response."""
        pass

    @abstractmethod
    def get_model_version(self) -> str:
        """Return the exact model string used (for the audit log)."""
        pass

    @abstractmethod
    def get_token_usage(self, raw_response: dict) -> dict:
        """Return {'input_tokens': int, 'output_tokens': int}."""
        pass
