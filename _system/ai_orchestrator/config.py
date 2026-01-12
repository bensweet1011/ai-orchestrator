"""
Configuration management for AI Orchestrator.
Loads API keys from environment variables.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

# Workspace paths
WORKSPACE_ROOT = Path.home() / "orchestrator-workspace"
SYSTEM_DIR = WORKSPACE_ROOT / "_system"
TEMPLATES_DIR = WORKSPACE_ROOT / "_templates"
PROJECTS_DIR = WORKSPACE_ROOT / "projects"

# Ensure directories exist
WORKSPACE_ROOT.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)


@dataclass
class ModelInfo:
    """Information about an available LLM."""

    name: str
    provider: str
    model_id: str
    description: str
    available: bool = False


# All supported models (January 2026)
MODELS: Dict[str, ModelInfo] = {
    # Anthropic - Claude 4.5 series
    "claude-opus": ModelInfo(
        name="claude-opus",
        provider="anthropic",
        model_id="claude-opus-4-20250514",
        description="Claude Opus 4.5 - maximum reasoning, best for complex tasks (DEFAULT)",
    ),
    "claude-sonnet": ModelInfo(
        name="claude-sonnet",
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        description="Claude Sonnet 4.5 - balanced performance and cost",
    ),
    "claude-haiku": ModelInfo(
        name="claude-haiku",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        description="Claude Haiku 4.5 - fast and efficient for simple tasks",
    ),
    # OpenAI - GPT-5 series
    "gpt5": ModelInfo(
        name="gpt5",
        provider="openai",
        model_id="gpt-5.2",
        description="GPT-5.2 - flagship model, strong general purpose",
    ),
    "codex": ModelInfo(
        name="codex",
        provider="openai",
        model_id="gpt-5.2-codex",
        description="GPT-5.2 Codex - optimized for code generation",
    ),
    "o3": ModelInfo(
        name="o3",
        provider="openai",
        model_id="o3",
        description="o3 - advanced reasoning model",
    ),
    "o4-mini": ModelInfo(
        name="o4-mini",
        provider="openai",
        model_id="o4-mini",
        description="o4-mini - efficient reasoning, lower cost",
    ),
    # Google - Gemini 2.5 series
    "gemini": ModelInfo(
        name="gemini",
        provider="google",
        model_id="gemini-2.5-pro",
        description="Gemini 2.5 Pro - long context, multimodal",
    ),
    "gemini-flash": ModelInfo(
        name="gemini-flash",
        provider="google",
        model_id="gemini-2.5-flash",
        description="Gemini 2.5 Flash - fast and cost-effective",
    ),
    # xAI - Grok 4 series
    "grok": ModelInfo(
        name="grok",
        provider="xai",
        model_id="grok-4-1",
        description="Grok 4.1 - real-time information, 2M context",
    ),
    "grok-fast": ModelInfo(
        name="grok-fast",
        provider="xai",
        model_id="grok-4-1-fast",
        description="Grok 4.1 Fast - lower latency, lower cost",
    ),
    # Perplexity - Sonar series
    "perplexity": ModelInfo(
        name="perplexity",
        provider="perplexity",
        model_id="sonar-pro",
        description="Sonar Pro - web search with citations, detailed answers",
    ),
    "sonar": ModelInfo(
        name="sonar",
        provider="perplexity",
        model_id="sonar",
        description="Sonar - fast web search, concise answers",
    ),
}


def get_api_keys() -> Dict[str, Optional[str]]:
    """Get all API keys from environment."""
    return {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
        "google": os.environ.get("GOOGLE_API_KEY"),
        "xai": os.environ.get("XAI_API_KEY"),
        "perplexity": os.environ.get("PERPLEXITY_API_KEY"),
        "pinecone": os.environ.get("PINECONE_API_KEY"),
        "github": os.environ.get("GITHUB_TOKEN"),
        "vercel": os.environ.get("VERCEL_TOKEN"),
    }


def get_available_models() -> List[str]:
    """Get list of models that have valid API keys configured."""
    keys = get_api_keys()
    available = []

    provider_key_map = {
        "anthropic": "anthropic",
        "openai": "openai",
        "google": "google",
        "xai": "xai",
        "perplexity": "perplexity",
    }

    for model_name, model_info in MODELS.items():
        key_name = provider_key_map.get(model_info.provider)
        if key_name and keys.get(key_name):
            model_info.available = True
            available.append(model_name)

    return available


def check_required_keys() -> Dict[str, bool]:
    """Check which required services are configured."""
    keys = get_api_keys()
    return {
        "pinecone": bool(keys.get("pinecone")),
        "openai": bool(keys.get("openai")),  # Required for embeddings
        "any_llm": any(
            [
                keys.get("anthropic"),
                keys.get("openai"),
                keys.get("google"),
            ]
        ),
    }


# Pinecone settings
PINECONE_INDEX = "llm-logs"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1024  # Must match Pinecone index


# Integration settings
GOOGLE_CREDENTIALS_PATH = SYSTEM_DIR / "google_credentials.json"
GOOGLE_TOKEN_PATH = SYSTEM_DIR / "google_token.json"

# Browser automation settings
BROWSER_DATA_DIR = TEMPLATES_DIR / "browser"
BROWSER_SCREENSHOTS_DIR = BROWSER_DATA_DIR / "screenshots"
BROWSER_CREDENTIALS_DIR = BROWSER_DATA_DIR / "credentials"
BROWSER_QUEUE_DIR = BROWSER_DATA_DIR / "queue"

# Default rate limits (seconds) - enforces human-like delays
BROWSER_MIN_DELAY = 3.0
BROWSER_MAX_DELAY = 5.0

# Deploy settings
DEPLOY_REGISTRY_PATH = SYSTEM_DIR / "deploy_registry.json"
DEPLOY_TEMPLATES_DIR = TEMPLATES_DIR / "deploy"
DEPLOY_TEMPLATES_DIR.mkdir(exist_ok=True)


def get_integration_status() -> Dict[str, bool]:
    """Check which integrations are configured."""
    return {
        "notion": bool(os.environ.get("NOTION_API_KEY")),
        "google_docs": GOOGLE_CREDENTIALS_PATH.exists(),
        "browser": True,  # Always available (no external API needed)
        "browser_credentials": bool(os.environ.get("BROWSER_MASTER_KEY")),
        "github": bool(os.environ.get("GITHUB_TOKEN")),
        "vercel": bool(os.environ.get("VERCEL_TOKEN")),
        "streamlit_cloud": bool(os.environ.get("GITHUB_TOKEN")),  # Uses GitHub for deployment
    }
