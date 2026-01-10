"""
Multi-provider token counting for Cost Intelligence.
Uses tiktoken for OpenAI models and estimation for others.
"""

from typing import Optional

# Lazy import tiktoken to avoid startup overhead
_tiktoken = None
_tiktoken_encodings = {}


def _get_tiktoken():
    """Lazy load tiktoken module."""
    global _tiktoken
    if _tiktoken is None:
        try:
            import tiktoken

            _tiktoken = tiktoken
        except ImportError:
            raise RuntimeError(
                "tiktoken not installed. Run: pip install tiktoken"
            )
    return _tiktoken


# Model to tiktoken encoding mapping
TIKTOKEN_ENCODING_MAP = {
    # OpenAI models
    "gpt-4o": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "o1-preview": "o200k_base",
    "o1": "o200k_base",
    "o3-mini": "o200k_base",
    "o3": "o200k_base",
    # OpenAI-compatible models (approximate)
    "grok-2-latest": "cl100k_base",
    "grok": "cl100k_base",
    "llama-3.1-sonar-large-128k-online": "cl100k_base",
}

# Average characters per token by provider (for estimation)
CHARS_PER_TOKEN = {
    "anthropic": 3.5,  # Claude models
    "openai": 4.0,
    "google": 4.0,
    "xai": 4.0,
    "perplexity": 4.0,
    "default": 4.0,
}


class TokenCounter:
    """Multi-provider token counting."""

    def __init__(self):
        """Initialize token counter."""
        self._encodings = {}

    def count_tokens(
        self,
        text: str,
        model: str,
        provider: Optional[str] = None,
    ) -> int:
        """
        Count tokens for text using appropriate tokenizer.

        Args:
            text: Text to count tokens for
            model: Model name or ID
            provider: Provider name (optional, for disambiguation)

        Returns:
            Token count
        """
        if not text:
            return 0

        # Determine provider from model if not specified
        if not provider:
            provider = self._infer_provider(model)

        # Route to appropriate counter
        if provider == "openai" or model in TIKTOKEN_ENCODING_MAP:
            return self._count_tiktoken(text, model)
        elif provider == "anthropic":
            return self._count_anthropic(text, model)
        elif provider == "google":
            return self._count_estimate(text, provider)
        else:
            return self._count_estimate(text, provider or "default")

    def _infer_provider(self, model: str) -> str:
        """Infer provider from model name."""
        model_lower = model.lower()

        if "claude" in model_lower:
            return "anthropic"
        elif any(x in model_lower for x in ["gpt", "o1", "o3"]):
            return "openai"
        elif "gemini" in model_lower:
            return "google"
        elif "grok" in model_lower:
            return "xai"
        elif "sonar" in model_lower or "perplexity" in model_lower:
            return "perplexity"

        return "default"

    def _count_tiktoken(self, text: str, model: str) -> int:
        """Count tokens using tiktoken for OpenAI models."""
        try:
            tiktoken = _get_tiktoken()

            # Get encoding name for model
            encoding_name = TIKTOKEN_ENCODING_MAP.get(model, "cl100k_base")

            # Cache encodings for performance
            if encoding_name not in self._encodings:
                self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)

            encoding = self._encodings[encoding_name]
            return len(encoding.encode(text))

        except Exception as e:
            # Fall back to estimation if tiktoken fails
            print(f"Warning: tiktoken failed for {model}: {e}")
            return self._count_estimate(text, "openai")

    def _count_anthropic(self, text: str, model: str) -> int:
        """
        Count tokens for Anthropic Claude models.

        Note: Anthropic doesn't provide a public tokenizer, so we use
        a character-based estimation calibrated to Claude's tokenization.
        Claude uses a BPE tokenizer similar to GPT-4 but with different
        vocabulary.
        """
        # Claude's tokenization is roughly 3.5 chars per token
        # This is calibrated from empirical testing
        chars_per_token = CHARS_PER_TOKEN["anthropic"]
        return max(1, int(len(text) / chars_per_token))

    def _count_estimate(self, text: str, provider: str) -> int:
        """
        Estimate token count based on character count.

        Args:
            text: Text to estimate
            provider: Provider for calibration

        Returns:
            Estimated token count
        """
        chars_per_token = CHARS_PER_TOKEN.get(provider, CHARS_PER_TOKEN["default"])
        return max(1, int(len(text) / chars_per_token))

    def count_messages(
        self,
        messages: list,
        model: str,
        provider: Optional[str] = None,
    ) -> int:
        """
        Count tokens for a list of messages (chat format).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            provider: Provider name

        Returns:
            Total token count including message overhead
        """
        total = 0

        for message in messages:
            # Count content tokens
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content, model, provider)
            elif isinstance(content, list):
                # Handle multimodal content
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += self.count_tokens(part["text"], model, provider)

            # Add overhead for message structure
            # OpenAI: ~4 tokens per message for role/structure
            # Claude: ~3 tokens per message
            if provider == "anthropic":
                total += 3
            else:
                total += 4

        # Add overhead for conversation structure
        total += 3  # Every reply is primed with <|start|>assistant<|message|>

        return total

    def count_with_system(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        provider: Optional[str] = None,
    ) -> int:
        """
        Count tokens for prompt with optional system message.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            model: Model name
            provider: Provider name

        Returns:
            Total token count
        """
        total = self.count_tokens(prompt, model, provider)

        if system:
            total += self.count_tokens(system, model, provider)
            # Add overhead for system message
            total += 4

        return total


# Singleton instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get or create TokenCounter singleton."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter
