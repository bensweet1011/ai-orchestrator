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


# All supported models
MODELS: Dict[str, ModelInfo] = {
    "claude": ModelInfo(
        name="claude",
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        description="Claude Sonnet - strong all-around, excellent for writing and analysis",
    ),
    "claude-opus": ModelInfo(
        name="claude-opus",
        provider="anthropic",
        model_id="claude-opus-4-20250514",
        description="Claude Opus - most capable, best for complex reasoning",
    ),
    "gpt4o": ModelInfo(
        name="gpt4o",
        provider="openai",
        model_id="gpt-4o",
        description="GPT-4o - fast, strong general purpose",
    ),
    "gpt4": ModelInfo(
        name="gpt4",
        provider="openai",
        model_id="gpt-4-turbo",
        description="GPT-4 Turbo - high capability",
    ),
    "o1": ModelInfo(
        name="o1",
        provider="openai",
        model_id="o1-preview",
        description="o1 - deep reasoning and analysis",
    ),
    "o3-mini": ModelInfo(
        name="o3-mini",
        provider="openai",
        model_id="o3-mini",
        description="o3-mini - efficient reasoning",
    ),
    "gemini": ModelInfo(
        name="gemini",
        provider="google",
        model_id="gemini-1.5-pro",
        description="Gemini 1.5 Pro - long context, multimodal",
    ),
    "grok": ModelInfo(
        name="grok",
        provider="xai",
        model_id="grok-2-latest",
        description="Grok - real-time information access",
    ),
    "perplexity": ModelInfo(
        name="perplexity",
        provider="perplexity",
        model_id="llama-3.1-sonar-large-128k-online",
        description="Perplexity - web search with citations",
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


def get_integration_status() -> Dict[str, bool]:
    """Check which integrations are configured."""
    return {
        "notion": bool(os.environ.get("NOTION_API_KEY")),
        "google_docs": GOOGLE_CREDENTIALS_PATH.exists(),
        "browser": True,  # Always available (no external API needed)
        "browser_credentials": bool(os.environ.get("BROWSER_MASTER_KEY")),
    }
