"""LLM API client with token tracking."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import LLMConfig


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class TokenStats:
    """Accumulated token statistics."""

    total_usage: TokenUsage = field(default_factory=TokenUsage)
    request_count: int = 0
    start_time: float = field(default_factory=time.time)

    def add_usage(self, usage: TokenUsage) -> None:
        """Add token usage from a request."""
        self.total_usage = self.total_usage + usage
        self.request_count += 1

    def get_rate(self) -> float:
        """Get tokens per minute rate."""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0
        return (self.total_usage.total_tokens / elapsed) * 60

    def estimate_cost(self, input_price: float = 0.01, output_price: float = 0.03) -> float:
        """Estimate cost in USD (per 1K tokens pricing)."""
        input_cost = (self.total_usage.prompt_tokens / 1000) * input_price
        output_cost = (self.total_usage.completion_tokens / 1000) * output_price
        return input_cost + output_cost

    def to_dict(self) -> dict:
        return {
            "usage": self.total_usage.to_dict(),
            "request_count": self.request_count,
            "elapsed_seconds": time.time() - self.start_time,
            "tokens_per_minute": self.get_rate(),
            "estimated_cost_usd": self.estimate_cost(),
        }


@dataclass
class LLMResponse:
    """Response from LLM API."""

    content: str
    usage: TokenUsage
    model: str
    finish_reason: str


class LLMClient:
    """LLM API client with token tracking."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.stats = TokenStats()
        self._client: Optional[httpx.Client] = None

        # Initialize tokenizer for fallback counting
        try:
            self._tokenizer = tiktoken.encoding_for_model(config.model)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
        return self._client

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))

    def _create_retry_decorator(self):
        """Create retry decorator based on config."""
        return retry(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(multiplier=1, min=self.config.retry_delay, max=60),
        )

    def _make_request(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Make a request to the LLM API."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        response = self.client.post("/chat/completions", json=payload)

        # Handle HTTP errors with detailed information
        if response.status_code >= 400:
            error_detail = ""
            try:
                error_detail = response.text[:500]
            except Exception:
                pass
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}: {error_detail}",
                request=response.request,
                response=response,
            )

        data = response.json()

        # Check for error in response
        if "error" in data:
            error_msg = data.get("error", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            raise ValueError(f"API returned error: {error_msg}")

        # Safely extract the response content - handle multiple API formats
        choices = data.get("choices", [])
        if not choices:
            # Some APIs may use different response structures
            # Try to handle Gemini-style responses
            if "candidates" in data:
                candidates = data.get("candidates", [])
                if candidates:
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    if content_parts:
                        content = content_parts[0].get("text", "")
                        if content:
                            # Create mock choice for compatibility
                            choices = [{"message": {"content": content}, "finish_reason": "stop"}]
            
            if not choices:
                raise ValueError(f"API response missing 'choices' field: {str(data)[:500]}")
        
        first_choice = choices[0]
        
        # Handle different message formats (streaming vs non-streaming)
        message = first_choice.get("message", {})
        if not message:
            # Some streaming responses use 'delta' instead of 'message'
            message = first_choice.get("delta", {})
        
        content = message.get("content", "")
        
        # Some APIs return content directly in the choice
        if not content and "text" in first_choice:
            content = first_choice.get("text", "")
        
        if not content:
            # Empty content is often a transient issue - raise to trigger retry
            import logging
            logging.getLogger("qa_extractor").warning(
                f"API response has empty content. Response: {str(data)[:500]}"
            )
            raise ValueError(f"API returned empty content. Response: {str(data)[:500]}")

        # Extract usage information
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Fallback token counting if API doesn't provide usage
        if usage.total_tokens == 0:
            prompt_text = " ".join(m.get("content", "") for m in messages)
            usage = TokenUsage(
                prompt_tokens=self.count_tokens(prompt_text),
                completion_tokens=self.count_tokens(content),
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        # Update statistics
        self.stats.add_usage(usage)

        return LLMResponse(
            content=content,
            usage=usage,
            model=data.get("model", self.config.model),
            finish_reason=first_choice.get("finish_reason", "unknown"),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send chat completion request with retry logic."""
        retry_decorator = self._create_retry_decorator()
        make_request_with_retry = retry_decorator(self._make_request)
        return make_request_with_retry(messages, temperature, max_tokens)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Any, LLMResponse]:
        """Send chat request and parse JSON response."""
        response = self.chat(messages, temperature, max_tokens)

        # Try to extract JSON from response
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            parsed = json.loads(content)
            return parsed, response
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content[:500]}")

    def get_stats(self) -> TokenStats:
        """Get current token statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset token statistics."""
        self.stats = TokenStats()

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()
