"""
Vercel integration for AI Orchestrator.
Provides deployment capabilities for professional Next.js frontends.
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DEPLOY_REGISTRY_PATH


@dataclass
class ProjectInfo:
    """Information about a Vercel project."""

    id: str
    name: str
    framework: str
    url: str
    created_at: str
    updated_at: str
    env_vars: List[str] = field(default_factory=list)


@dataclass
class DeploymentInfo:
    """Information about a Vercel deployment."""

    id: str
    project_id: str
    url: str
    state: str  # QUEUED, BUILDING, READY, ERROR, CANCELED
    created_at: str
    ready_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class DomainInfo:
    """Information about a custom domain."""

    name: str
    verified: bool
    configured: bool


class VercelClient:
    """
    Client for Vercel API operations.

    Features:
    - Project creation and management
    - Deployment from GitHub repos
    - Environment variable management
    - Custom domain configuration
    """

    BASE_URL = "https://api.vercel.com"

    def __init__(self, token: Optional[str] = None, team_id: Optional[str] = None):
        """
        Initialize Vercel client.

        Args:
            token: Vercel API token (defaults to VERCEL_TOKEN env var)
            team_id: Team ID for team deployments (optional)
        """
        self.token = token or os.environ.get("VERCEL_TOKEN")
        if not self.token:
            raise ValueError(
                "Vercel token required. Set VERCEL_TOKEN environment variable."
            )

        self.team_id = team_id or os.environ.get("VERCEL_TEAM_ID")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make API request to Vercel."""
        url = f"{self.BASE_URL}{endpoint}"

        # Add team ID if available
        if self.team_id:
            params = params or {}
            params["teamId"] = self.team_id

        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            json=data,
            params=params,
        )

        if response.status_code >= 400:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", {}).get("message", response.text)
            raise Exception(f"Vercel API error ({response.status_code}): {error_msg}")

        return response.json() if response.content else {}

    # =========================================================================
    # Project Operations
    # =========================================================================

    def create_project(
        self,
        name: str,
        framework: str = "nextjs",
        github_repo: Optional[str] = None,
        build_command: Optional[str] = None,
        output_directory: Optional[str] = None,
        install_command: Optional[str] = None,
    ) -> ProjectInfo:
        """
        Create a new Vercel project.

        Args:
            name: Project name
            framework: Framework preset (nextjs, vite, etc.)
            github_repo: GitHub repo to link (owner/repo)
            build_command: Custom build command
            output_directory: Build output directory
            install_command: Custom install command

        Returns:
            Created project info
        """
        data: Dict[str, Any] = {
            "name": name,
            "framework": framework,
        }

        if build_command:
            data["buildCommand"] = build_command
        if output_directory:
            data["outputDirectory"] = output_directory
        if install_command:
            data["installCommand"] = install_command

        # Link to GitHub repo if provided
        if github_repo:
            parts = github_repo.split("/")
            if len(parts) == 2:
                data["gitRepository"] = {
                    "type": "github",
                    "repo": github_repo,
                }

        result = self._request("POST", "/v10/projects", data=data)

        return ProjectInfo(
            id=result["id"],
            name=result["name"],
            framework=result.get("framework", framework),
            url=f"https://{result['name']}.vercel.app",
            created_at=datetime.utcfromtimestamp(
                result.get("createdAt", 0) / 1000
            ).isoformat(),
            updated_at=datetime.utcfromtimestamp(
                result.get("updatedAt", 0) / 1000
            ).isoformat(),
        )

    def get_project(self, project_id_or_name: str) -> ProjectInfo:
        """
        Get project information.

        Args:
            project_id_or_name: Project ID or name

        Returns:
            Project info
        """
        result = self._request("GET", f"/v9/projects/{project_id_or_name}")

        env_vars = [
            env["key"] for env in result.get("env", [])
        ]

        return ProjectInfo(
            id=result["id"],
            name=result["name"],
            framework=result.get("framework", ""),
            url=f"https://{result['name']}.vercel.app",
            created_at=datetime.utcfromtimestamp(
                result.get("createdAt", 0) / 1000
            ).isoformat(),
            updated_at=datetime.utcfromtimestamp(
                result.get("updatedAt", 0) / 1000
            ).isoformat(),
            env_vars=env_vars,
        )

    def list_projects(self, limit: int = 20) -> List[ProjectInfo]:
        """
        List all projects.

        Args:
            limit: Maximum number of projects

        Returns:
            List of project info
        """
        result = self._request("GET", "/v9/projects", params={"limit": limit})

        projects = []
        for proj in result.get("projects", []):
            projects.append(ProjectInfo(
                id=proj["id"],
                name=proj["name"],
                framework=proj.get("framework", ""),
                url=f"https://{proj['name']}.vercel.app",
                created_at=datetime.utcfromtimestamp(
                    proj.get("createdAt", 0) / 1000
                ).isoformat(),
                updated_at=datetime.utcfromtimestamp(
                    proj.get("updatedAt", 0) / 1000
                ).isoformat(),
            ))

        return projects

    def delete_project(self, project_id_or_name: str) -> bool:
        """
        Delete a project.

        Args:
            project_id_or_name: Project ID or name

        Returns:
            True if deleted
        """
        self._request("DELETE", f"/v9/projects/{project_id_or_name}")
        return True

    # =========================================================================
    # Deployment Operations
    # =========================================================================

    def create_deployment(
        self,
        project_id: str,
        files: Dict[str, str],
        target: str = "production",
    ) -> DeploymentInfo:
        """
        Create a deployment from files.

        Args:
            project_id: Project ID
            files: Dict of {path: content}
            target: Deployment target (production, preview)

        Returns:
            Deployment info
        """
        # Prepare file entries
        file_entries = []
        for path, content in files.items():
            file_entries.append({
                "file": path,
                "data": content,
            })

        data = {
            "name": project_id,
            "files": file_entries,
            "target": target,
            "projectSettings": {
                "framework": "nextjs",
            },
        }

        result = self._request("POST", "/v13/deployments", data=data)

        return DeploymentInfo(
            id=result["id"],
            project_id=project_id,
            url=f"https://{result.get('url', '')}",
            state=result.get("readyState", "QUEUED"),
            created_at=datetime.utcfromtimestamp(
                result.get("createdAt", 0) / 1000
            ).isoformat(),
        )

    def deploy_from_github(
        self,
        project_id: str,
        ref: str = "main",
    ) -> DeploymentInfo:
        """
        Trigger deployment from linked GitHub repo.

        Args:
            project_id: Project ID (must have GitHub repo linked)
            ref: Git ref (branch, tag, commit)

        Returns:
            Deployment info
        """
        data = {
            "name": project_id,
            "gitSource": {
                "ref": ref,
                "type": "github",
            },
        }

        result = self._request("POST", "/v13/deployments", data=data)

        return DeploymentInfo(
            id=result["id"],
            project_id=project_id,
            url=f"https://{result.get('url', '')}",
            state=result.get("readyState", "QUEUED"),
            created_at=datetime.utcfromtimestamp(
                result.get("createdAt", 0) / 1000
            ).isoformat(),
        )

    def get_deployment(self, deployment_id: str) -> DeploymentInfo:
        """
        Get deployment status.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment info
        """
        result = self._request("GET", f"/v13/deployments/{deployment_id}")

        return DeploymentInfo(
            id=result["id"],
            project_id=result.get("projectId", ""),
            url=f"https://{result.get('url', '')}",
            state=result.get("readyState", "UNKNOWN"),
            created_at=datetime.utcfromtimestamp(
                result.get("createdAt", 0) / 1000
            ).isoformat(),
            ready_at=datetime.utcfromtimestamp(
                result["ready"] / 1000
            ).isoformat() if result.get("ready") else None,
            error_message=result.get("errorMessage"),
        )

    def wait_for_deployment(
        self,
        deployment_id: str,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> DeploymentInfo:
        """
        Wait for deployment to complete.

        Args:
            deployment_id: Deployment ID
            timeout: Max seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            Final deployment info

        Raises:
            TimeoutError: If deployment doesn't complete in time
            Exception: If deployment fails
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            deployment = self.get_deployment(deployment_id)

            if deployment.state == "READY":
                return deployment
            elif deployment.state in ("ERROR", "CANCELED"):
                raise Exception(
                    f"Deployment failed: {deployment.error_message or deployment.state}"
                )

            time.sleep(poll_interval)

        raise TimeoutError(f"Deployment did not complete within {timeout} seconds")

    def list_deployments(
        self,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[DeploymentInfo]:
        """
        List deployments.

        Args:
            project_id: Filter by project (optional)
            limit: Maximum deployments to return

        Returns:
            List of deployment info
        """
        params: Dict[str, Any] = {"limit": limit}
        if project_id:
            params["projectId"] = project_id

        result = self._request("GET", "/v6/deployments", params=params)

        deployments = []
        for dep in result.get("deployments", []):
            deployments.append(DeploymentInfo(
                id=dep["uid"],
                project_id=dep.get("projectId", ""),
                url=f"https://{dep.get('url', '')}",
                state=dep.get("state", "UNKNOWN"),
                created_at=datetime.utcfromtimestamp(
                    dep.get("createdAt", 0) / 1000
                ).isoformat(),
                ready_at=datetime.utcfromtimestamp(
                    dep["ready"] / 1000
                ).isoformat() if dep.get("ready") else None,
            ))

        return deployments

    # =========================================================================
    # Environment Variables
    # =========================================================================

    def set_env_vars(
        self,
        project_id: str,
        env_vars: Dict[str, str],
        target: List[str] = None,
    ) -> bool:
        """
        Set environment variables for a project.

        Args:
            project_id: Project ID
            env_vars: Dict of {key: value}
            target: Deployment targets (production, preview, development)

        Returns:
            True if successful
        """
        target = target or ["production", "preview", "development"]

        for key, value in env_vars.items():
            data = {
                "key": key,
                "value": value,
                "type": "encrypted",
                "target": target,
            }

            try:
                self._request("POST", f"/v10/projects/{project_id}/env", data=data)
            except Exception as e:
                # If variable exists, update it
                if "already exists" in str(e).lower():
                    self._request(
                        "PATCH",
                        f"/v9/projects/{project_id}/env/{key}",
                        data={"value": value, "target": target},
                    )
                else:
                    raise

        return True

    def get_env_vars(self, project_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get environment variables for a project.

        Args:
            project_id: Project ID

        Returns:
            Dict of {key: {value, target, type}}
        """
        result = self._request("GET", f"/v9/projects/{project_id}/env")

        env_vars = {}
        for env in result.get("envs", []):
            env_vars[env["key"]] = {
                "value": env.get("value", "[encrypted]"),
                "target": env.get("target", []),
                "type": env.get("type", ""),
            }

        return env_vars

    def delete_env_var(self, project_id: str, key: str) -> bool:
        """
        Delete an environment variable.

        Args:
            project_id: Project ID
            key: Variable key

        Returns:
            True if deleted
        """
        self._request("DELETE", f"/v9/projects/{project_id}/env/{key}")
        return True

    # =========================================================================
    # Domain Operations
    # =========================================================================

    def add_domain(self, project_id: str, domain: str) -> DomainInfo:
        """
        Add a custom domain to a project.

        Args:
            project_id: Project ID
            domain: Domain name

        Returns:
            Domain info
        """
        data = {"name": domain}
        result = self._request(
            "POST",
            f"/v10/projects/{project_id}/domains",
            data=data,
        )

        return DomainInfo(
            name=result.get("name", domain),
            verified=result.get("verified", False),
            configured=result.get("configured", False),
        )

    def list_domains(self, project_id: str) -> List[DomainInfo]:
        """
        List domains for a project.

        Args:
            project_id: Project ID

        Returns:
            List of domain info
        """
        result = self._request("GET", f"/v9/projects/{project_id}/domains")

        domains = []
        for dom in result.get("domains", []):
            domains.append(DomainInfo(
                name=dom.get("name", ""),
                verified=dom.get("verified", False),
                configured=dom.get("configured", False),
            ))

        return domains

    def remove_domain(self, project_id: str, domain: str) -> bool:
        """
        Remove a domain from a project.

        Args:
            project_id: Project ID
            domain: Domain name

        Returns:
            True if removed
        """
        self._request("DELETE", f"/v9/projects/{project_id}/domains/{domain}")
        return True


# Singleton instance
_vercel_client: Optional[VercelClient] = None


def get_vercel_client() -> VercelClient:
    """Get or create VercelClient singleton."""
    global _vercel_client
    if _vercel_client is None:
        _vercel_client = VercelClient()
    return _vercel_client


def reset_vercel_client() -> None:
    """Reset the singleton."""
    global _vercel_client
    _vercel_client = None
