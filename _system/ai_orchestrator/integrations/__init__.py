"""
External integrations for AI Orchestrator.
Provides connectors for Notion, Google Docs, and other services.
"""

from .notion import (
    NotionClient,
    get_notion_client,
)

from .google_docs import (
    GoogleDocsClient,
    get_google_docs_client,
    is_google_docs_configured,
)

__all__ = [
    # Notion
    "NotionClient",
    "get_notion_client",
    # Google Docs
    "GoogleDocsClient",
    "get_google_docs_client",
    "is_google_docs_configured",
]
