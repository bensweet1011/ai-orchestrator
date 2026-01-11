"""
External integrations for AI Orchestrator.
Provides connectors for Notion, Google Docs, GitHub, Streamlit Cloud, Vercel, and other services.
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

from .github import (
    GitHubClient,
    get_github_client,
    reset_github_client,
    RepoInfo,
    CommitInfo,
    BranchInfo,
    WorkflowRun,
)

from .streamlit_cloud import (
    StreamlitCloudClient,
    get_streamlit_cloud_client,
    reset_streamlit_cloud_client,
    StreamlitAppConfig,
    StreamlitDeployment,
)

from .vercel import (
    VercelClient,
    get_vercel_client,
    reset_vercel_client,
    ProjectInfo,
    DeploymentInfo,
    DomainInfo,
)

__all__ = [
    # Notion
    "NotionClient",
    "get_notion_client",
    # Google Docs
    "GoogleDocsClient",
    "get_google_docs_client",
    "is_google_docs_configured",
    # GitHub
    "GitHubClient",
    "get_github_client",
    "reset_github_client",
    "RepoInfo",
    "CommitInfo",
    "BranchInfo",
    "WorkflowRun",
    # Streamlit Cloud
    "StreamlitCloudClient",
    "get_streamlit_cloud_client",
    "reset_streamlit_cloud_client",
    "StreamlitAppConfig",
    "StreamlitDeployment",
    # Vercel
    "VercelClient",
    "get_vercel_client",
    "reset_vercel_client",
    "ProjectInfo",
    "DeploymentInfo",
    "DomainInfo",
]
