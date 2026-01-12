"""
Multi-provider LLM client management.
Handles connections to Claude, GPT, Gemini, Grok, Perplexity.
"""

import os
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Import clients
from anthropic import Anthropic
from openai import OpenAI

# Optional imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class LLMResponse:
    """Standardized response from any LLM."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class LLMClients:
    """
    Manages connections to multiple LLM providers.
    Provides unified interface for calling any model.
    """
    
    def __init__(self):
        self.clients: Dict[str, Any] = {}
        self._init_clients()
    
    def _init_clients(self):
        """Initialize available LLM clients based on API keys."""
        
        # Anthropic (Claude)
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.clients["anthropic"] = Anthropic()
        
        # OpenAI (GPT, o1, embeddings)
        if os.environ.get("OPENAI_API_KEY"):
            self.clients["openai"] = OpenAI()
        
        # Google (Gemini)
        if GEMINI_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.clients["google"] = genai.GenerativeModel("gemini-2.5-pro")
        
        # xAI (Grok) - uses OpenAI-compatible API
        if os.environ.get("XAI_API_KEY"):
            self.clients["xai"] = OpenAI(
                api_key=os.environ.get("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        
        # Perplexity - uses OpenAI-compatible API
        if os.environ.get("PERPLEXITY_API_KEY"):
            self.clients["perplexity"] = OpenAI(
                api_key=os.environ.get("PERPLEXITY_API_KEY"),
                base_url="https://api.perplexity.ai"
            )
    
    def get_available_providers(self) -> list:
        """Return list of available providers."""
        return list(self.clients.keys())
    
    def call(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> LLMResponse:
        """
        Call an LLM with unified interface.
        
        Args:
            prompt: User prompt
            model: Model name (claude, gpt4o, gemini, grok, perplexity, etc.)
            system: Optional system prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
        
        Returns:
            LLMResponse with content and metadata
        """
        
        # Map model names to providers and model IDs (January 2026)
        model_map = {
            # Anthropic - Claude 4.5 series
            "claude-opus": ("anthropic", "claude-opus-4-20250514"),
            "claude-sonnet": ("anthropic", "claude-sonnet-4-20250514"),
            "claude": ("anthropic", "claude-sonnet-4-20250514"),  # Legacy alias
            "claude-haiku": ("anthropic", "claude-haiku-4-5"),
            # OpenAI - GPT-5 series
            "gpt5": ("openai", "gpt-5.2"),
            "gpt-5": ("openai", "gpt-5.2"),
            "codex": ("openai", "gpt-5.2-codex"),
            "gpt5-codex": ("openai", "gpt-5.2-codex"),
            "o3": ("openai", "o3"),
            "o4-mini": ("openai", "o4-mini"),
            # Google - Gemini 2.5 series
            "gemini": ("google", "gemini-2.5-pro"),
            "gemini-pro": ("google", "gemini-2.5-pro"),
            "gemini-flash": ("google", "gemini-2.5-flash"),
            # xAI - Grok 4 series
            "grok": ("xai", "grok-4-1"),
            "grok-fast": ("xai", "grok-4-1-fast"),
            # Perplexity - Sonar series
            "perplexity": ("perplexity", "sonar-pro"),
            "sonar": ("perplexity", "sonar"),
        }
        
        if model not in model_map:
            raise ValueError(f"Unknown model: {model}. Available: {list(model_map.keys())}")
        
        provider, model_id = model_map[model]
        
        if provider not in self.clients:
            raise RuntimeError(f"Provider '{provider}' not available. Check API key.")
        
        # Route to appropriate handler
        if provider == "anthropic":
            return self._call_anthropic(prompt, model_id, system, max_tokens, temperature)
        elif provider == "openai":
            return self._call_openai(prompt, model_id, system, max_tokens, temperature)
        elif provider == "google":
            return self._call_google(prompt, model_id, system, max_tokens, temperature)
        elif provider == "xai":
            return self._call_openai_compatible(
                self.clients["xai"], prompt, model_id, system, max_tokens, temperature, "xai"
            )
        elif provider == "perplexity":
            return self._call_openai_compatible(
                self.clients["perplexity"], prompt, model_id, system, max_tokens, temperature, "perplexity"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_anthropic(
        self, prompt: str, model_id: str, system: Optional[str],
        max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Call Anthropic API."""
        client = self.clients["anthropic"]
        
        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if "opus" not in model_id:  # Opus doesn't support temperature well
            kwargs["temperature"] = temperature
        
        response = client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=model_id,
            provider="anthropic",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            raw_response=response
        )
    
    def _call_openai(
        self, prompt: str, model_id: str, system: Optional[str],
        max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Call OpenAI API."""
        client = self.clients["openai"]
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model_id,
            "messages": messages,
        }
        
        # o1/o3 models don't support temperature or max_tokens the same way
        if not model_id.startswith("o1") and not model_id.startswith("o3"):
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        
        response = client.chat.completions.create(**kwargs)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_id,
            provider="openai",
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            } if response.usage else None,
            raw_response=response
        )
    
    def _call_google(
        self, prompt: str, model_id: str, system: Optional[str],
        max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Call Google Gemini API."""
        client = self.clients["google"]
        
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        response = client.generate_content(full_prompt)
        
        return LLMResponse(
            content=response.text,
            model=model_id,
            provider="google",
            raw_response=response
        )
    
    def _call_openai_compatible(
        self, client: OpenAI, prompt: str, model_id: str,
        system: Optional[str], max_tokens: int, temperature: float,
        provider_name: str
    ) -> LLMResponse:
        """Call OpenAI-compatible API (Grok, Perplexity)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_id,
            provider=provider_name,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            } if response.usage else None,
            raw_response=response
        )
    
    def embed(self, text: str, dimensions: int = 1024) -> list:
        """Generate embedding using OpenAI.

        Args:
            text: Text to embed
            dimensions: Output dimensions (must match Pinecone index, default 1024)
        """
        if "openai" not in self.clients:
            raise RuntimeError("OpenAI client required for embeddings")

        response = self.clients["openai"].embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],  # Truncate to avoid token limits
            dimensions=dimensions,
        )
        return response.data[0].embedding


# Singleton instance
_clients: Optional[LLMClients] = None

def get_clients() -> LLMClients:
    """Get or create the LLM clients singleton."""
    global _clients
    if _clients is None:
        _clients = LLMClients()
    return _clients
