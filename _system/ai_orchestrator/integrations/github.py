"""
GitHub integration for AI Orchestrator.
Provides repository management and deployment workflows.
"""

import os
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from github import Github, GithubException
from github.Repository import Repository
from github.ContentFile import ContentFile


@dataclass
class RepoInfo:
    """Information about a GitHub repository."""

    name: str
    full_name: str
    description: str
    url: str
    clone_url: str
    default_branch: str
    private: bool
    created_at: str
    updated_at: str


@dataclass
class CommitInfo:
    """Information about a commit."""

    sha: str
    message: str
    url: str
    author: str
    date: str


@dataclass
class BranchInfo:
    """Information about a branch."""

    name: str
    sha: str
    protected: bool


@dataclass
class WorkflowRun:
    """Information about a workflow run."""

    id: int
    name: str
    status: str
    conclusion: Optional[str]
    url: str
    created_at: str


class GitHubClient:
    """
    Client for GitHub API operations.

    Features:
    - Repository CRUD operations
    - File and branch management
    - Actions workflow triggers
    - Deployment preparation
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token (defaults to GITHUB_TOKEN env var)
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN environment variable."
            )

        self.client = Github(self.token)
        self.user = self.client.get_user()

    # =========================================================================
    # Repository Operations
    # =========================================================================

    def create_repository(
        self,
        name: str,
        description: str = "",
        private: bool = False,
        auto_init: bool = True,
        gitignore_template: Optional[str] = None,
    ) -> RepoInfo:
        """
        Create a new GitHub repository.

        Args:
            name: Repository name
            description: Repository description
            private: Whether repo should be private
            auto_init: Initialize with README
            gitignore_template: Gitignore template (e.g., "Python", "Node")

        Returns:
            Created repository info
        """
        repo = self.user.create_repo(
            name=name,
            description=description,
            private=private,
            auto_init=auto_init,
            gitignore_template=gitignore_template,
        )

        return self._repo_to_info(repo)

    def get_repository(self, repo_name: str) -> RepoInfo:
        """
        Get repository information.

        Args:
            repo_name: Repository name (can be "owner/repo" or just "repo" for own repos)

        Returns:
            Repository info
        """
        repo = self._get_repo(repo_name)
        return self._repo_to_info(repo)

    def list_repositories(
        self,
        visibility: str = "all",
        sort: str = "updated",
        limit: int = 100,
    ) -> List[RepoInfo]:
        """
        List user's repositories.

        Args:
            visibility: "all", "public", or "private"
            sort: "created", "updated", "pushed", "full_name"
            limit: Maximum number of repos to return

        Returns:
            List of repository info
        """
        repos = self.user.get_repos(visibility=visibility, sort=sort)

        result = []
        for i, repo in enumerate(repos):
            if i >= limit:
                break
            result.append(self._repo_to_info(repo))

        return result

    def delete_repository(self, repo_name: str) -> bool:
        """
        Delete a repository.

        Args:
            repo_name: Repository name

        Returns:
            True if deleted successfully
        """
        repo = self._get_repo(repo_name)
        repo.delete()
        return True

    # =========================================================================
    # File Operations
    # =========================================================================

    def get_file(self, repo_name: str, path: str, branch: str = "main") -> Dict[str, Any]:
        """
        Get a file from a repository.

        Args:
            repo_name: Repository name
            path: File path in repo
            branch: Branch name

        Returns:
            File content and metadata
        """
        repo = self._get_repo(repo_name)

        try:
            content = repo.get_contents(path, ref=branch)
            if isinstance(content, list):
                # It's a directory
                return {
                    "type": "directory",
                    "path": path,
                    "files": [f.path for f in content],
                }

            return {
                "type": "file",
                "path": content.path,
                "sha": content.sha,
                "size": content.size,
                "content": base64.b64decode(content.content).decode("utf-8"),
                "encoding": content.encoding,
            }
        except GithubException as e:
            if e.status == 404:
                return {"type": "not_found", "path": path}
            raise

    def push_file(
        self,
        repo_name: str,
        path: str,
        content: str,
        message: str,
        branch: str = "main",
    ) -> CommitInfo:
        """
        Create or update a file in a repository.

        Args:
            repo_name: Repository name
            path: File path in repo
            content: File content
            message: Commit message
            branch: Branch name

        Returns:
            Commit info
        """
        repo = self._get_repo(repo_name)

        # Check if file exists
        try:
            existing = repo.get_contents(path, ref=branch)
            if isinstance(existing, list):
                raise ValueError(f"Path {path} is a directory, not a file")

            # Update existing file
            result = repo.update_file(
                path=path,
                message=message,
                content=content,
                sha=existing.sha,
                branch=branch,
            )
        except GithubException as e:
            if e.status == 404:
                # Create new file
                result = repo.create_file(
                    path=path,
                    message=message,
                    content=content,
                    branch=branch,
                )
            else:
                raise

        commit = result["commit"]
        return CommitInfo(
            sha=commit.sha,
            message=commit.commit.message,
            url=commit.html_url,
            author=commit.commit.author.name,
            date=commit.commit.author.date.isoformat(),
        )

    def delete_file(
        self,
        repo_name: str,
        path: str,
        message: str,
        branch: str = "main",
    ) -> CommitInfo:
        """
        Delete a file from a repository.

        Args:
            repo_name: Repository name
            path: File path
            message: Commit message
            branch: Branch name

        Returns:
            Commit info
        """
        repo = self._get_repo(repo_name)
        content = repo.get_contents(path, ref=branch)

        if isinstance(content, list):
            raise ValueError(f"Path {path} is a directory")

        result = repo.delete_file(
            path=path,
            message=message,
            sha=content.sha,
            branch=branch,
        )

        commit = result["commit"]
        return CommitInfo(
            sha=commit.sha,
            message=commit.commit.message,
            url=commit.html_url,
            author=commit.commit.author.name,
            date=commit.commit.author.date.isoformat(),
        )

    def push_multiple_files(
        self,
        repo_name: str,
        files: Dict[str, str],
        message: str,
        branch: str = "main",
    ) -> CommitInfo:
        """
        Push multiple files in a single commit.

        Args:
            repo_name: Repository name
            files: Dict of {path: content}
            message: Commit message
            branch: Branch name

        Returns:
            Commit info
        """
        repo = self._get_repo(repo_name)

        # Get the branch reference
        ref = repo.get_git_ref(f"heads/{branch}")
        base_tree = repo.get_git_tree(ref.object.sha)

        # Create blob for each file
        blobs = []
        for path, content in files.items():
            blob = repo.create_git_blob(content, "utf-8")
            blobs.append({
                "path": path,
                "mode": "100644",
                "type": "blob",
                "sha": blob.sha,
            })

        # Create new tree
        new_tree = repo.create_git_tree(blobs, base_tree)

        # Create commit
        parent = repo.get_git_commit(ref.object.sha)
        commit = repo.create_git_commit(message, new_tree, [parent])

        # Update reference
        ref.edit(commit.sha)

        return CommitInfo(
            sha=commit.sha,
            message=message,
            url=commit.html_url,
            author=commit.author.name if commit.author else "Unknown",
            date=commit.author.date.isoformat() if commit.author else datetime.utcnow().isoformat(),
        )

    # =========================================================================
    # Branch Operations
    # =========================================================================

    def create_branch(
        self,
        repo_name: str,
        branch_name: str,
        from_branch: str = "main",
    ) -> BranchInfo:
        """
        Create a new branch.

        Args:
            repo_name: Repository name
            branch_name: New branch name
            from_branch: Source branch

        Returns:
            Branch info
        """
        repo = self._get_repo(repo_name)

        # Get source branch SHA
        source = repo.get_branch(from_branch)

        # Create new branch
        ref = repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=source.commit.sha,
        )

        return BranchInfo(
            name=branch_name,
            sha=ref.object.sha,
            protected=False,
        )

    def get_branch(self, repo_name: str, branch_name: str) -> BranchInfo:
        """
        Get branch information.

        Args:
            repo_name: Repository name
            branch_name: Branch name

        Returns:
            Branch info
        """
        repo = self._get_repo(repo_name)
        branch = repo.get_branch(branch_name)

        return BranchInfo(
            name=branch.name,
            sha=branch.commit.sha,
            protected=branch.protected,
        )

    def list_branches(self, repo_name: str) -> List[BranchInfo]:
        """
        List all branches in a repository.

        Args:
            repo_name: Repository name

        Returns:
            List of branch info
        """
        repo = self._get_repo(repo_name)
        branches = repo.get_branches()

        return [
            BranchInfo(
                name=b.name,
                sha=b.commit.sha,
                protected=b.protected,
            )
            for b in branches
        ]

    def delete_branch(self, repo_name: str, branch_name: str) -> bool:
        """
        Delete a branch.

        Args:
            repo_name: Repository name
            branch_name: Branch name

        Returns:
            True if deleted
        """
        repo = self._get_repo(repo_name)
        ref = repo.get_git_ref(f"heads/{branch_name}")
        ref.delete()
        return True

    # =========================================================================
    # Workflow Operations
    # =========================================================================

    def trigger_workflow(
        self,
        repo_name: str,
        workflow_id: str,
        ref: str = "main",
        inputs: Optional[Dict[str, str]] = None,
    ) -> WorkflowRun:
        """
        Trigger a GitHub Actions workflow.

        Args:
            repo_name: Repository name
            workflow_id: Workflow file name or ID
            ref: Branch/tag to run on
            inputs: Workflow inputs

        Returns:
            Workflow run info
        """
        repo = self._get_repo(repo_name)
        workflow = repo.get_workflow(workflow_id)

        # Dispatch workflow
        workflow.create_dispatch(ref=ref, inputs=inputs or {})

        # Get the latest run (just triggered)
        # Note: There's a small race condition here
        import time
        time.sleep(2)  # Wait for workflow to register

        runs = workflow.get_runs()
        latest_run = next(iter(runs), None)

        if latest_run:
            return WorkflowRun(
                id=latest_run.id,
                name=latest_run.name,
                status=latest_run.status,
                conclusion=latest_run.conclusion,
                url=latest_run.html_url,
                created_at=latest_run.created_at.isoformat(),
            )

        # Return placeholder if we can't get the run
        return WorkflowRun(
            id=0,
            name=workflow_id,
            status="queued",
            conclusion=None,
            url="",
            created_at=datetime.utcnow().isoformat(),
        )

    def get_workflow_run(self, repo_name: str, run_id: int) -> WorkflowRun:
        """
        Get workflow run status.

        Args:
            repo_name: Repository name
            run_id: Workflow run ID

        Returns:
            Workflow run info
        """
        repo = self._get_repo(repo_name)
        run = repo.get_workflow_run(run_id)

        return WorkflowRun(
            id=run.id,
            name=run.name,
            status=run.status,
            conclusion=run.conclusion,
            url=run.html_url,
            created_at=run.created_at.isoformat(),
        )

    def list_workflow_runs(
        self,
        repo_name: str,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> List[WorkflowRun]:
        """
        List workflow runs.

        Args:
            repo_name: Repository name
            workflow_id: Filter by workflow (optional)
            status: Filter by status (optional)
            limit: Max runs to return

        Returns:
            List of workflow runs
        """
        repo = self._get_repo(repo_name)

        if workflow_id:
            workflow = repo.get_workflow(workflow_id)
            runs = workflow.get_runs(status=status) if status else workflow.get_runs()
        else:
            runs = repo.get_workflow_runs(status=status) if status else repo.get_workflow_runs()

        result = []
        for i, run in enumerate(runs):
            if i >= limit:
                break
            result.append(WorkflowRun(
                id=run.id,
                name=run.name,
                status=run.status,
                conclusion=run.conclusion,
                url=run.html_url,
                created_at=run.created_at.isoformat(),
            ))

        return result

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_repo(self, repo_name: str) -> Repository:
        """Get repository object, handling both full names and short names."""
        if "/" in repo_name:
            return self.client.get_repo(repo_name)
        else:
            return self.client.get_repo(f"{self.user.login}/{repo_name}")

    def _repo_to_info(self, repo: Repository) -> RepoInfo:
        """Convert Repository object to RepoInfo."""
        return RepoInfo(
            name=repo.name,
            full_name=repo.full_name,
            description=repo.description or "",
            url=repo.html_url,
            clone_url=repo.clone_url,
            default_branch=repo.default_branch,
            private=repo.private,
            created_at=repo.created_at.isoformat() if repo.created_at else "",
            updated_at=repo.updated_at.isoformat() if repo.updated_at else "",
        )


# Singleton instance
_github_client: Optional[GitHubClient] = None


def get_github_client() -> GitHubClient:
    """Get or create GitHubClient singleton."""
    global _github_client
    if _github_client is None:
        _github_client = GitHubClient()
    return _github_client


def reset_github_client() -> None:
    """Reset the singleton (useful for testing or token refresh)."""
    global _github_client
    _github_client = None
