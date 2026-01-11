"""
Enhanced Pinecone store for cross-session memory persistence.
Extends core memory with pipeline-specific storage and namespace isolation.
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pinecone import Pinecone

from .state import (
    MemoryNamespace,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
    PipelineMemory,
)
from ..config import PROJECTS_DIR, TEMPLATES_DIR
from ..pipelines.state import PipelineResult


# Local cache for namespace mappings
NAMESPACE_CACHE_FILE = TEMPLATES_DIR / "memory_namespaces.json"


class MemoryStore:
    """
    Enhanced Pinecone store with project-scoped namespaces.

    Features:
    - Pipeline run logging with full output storage
    - Semantic search across pipeline outputs
    - Project namespace isolation
    - Cross-session persistence
    """

    def __init__(
        self,
        index_name: str = "llm-logs",
        default_project: str = "default",
    ):
        """
        Initialize memory store.

        Args:
            index_name: Pinecone index name
            default_project: Default project for unscoped operations
        """
        import os

        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        self.index_name = index_name

        # Session tracking
        self.session_id = self._generate_id("session")
        self.current_project = default_project

        # Namespace cache
        self._namespaces: Dict[str, MemoryNamespace] = {}
        self._load_namespace_cache()

        # LLM clients for embeddings
        self._llm_clients = None

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        if self._llm_clients is None:
            from ..core.llm_clients import get_clients

            self._llm_clients = get_clients()
        return self._llm_clients.embed(text)

    def _generate_id(self, prefix: str = "mem") -> str:
        """Generate unique ID."""
        ts = datetime.utcnow().isoformat()
        unique = hashlib.md5(f"{ts}{time.time_ns()}".encode()).hexdigest()[:8]
        return f"{prefix}_{unique}"

    def _get_namespace(self, project: str) -> str:
        """Get Pinecone namespace for project."""
        if project not in self._namespaces:
            self._create_namespace(project)
        return self._namespaces[project].full_namespace

    def _create_namespace(self, project: str) -> MemoryNamespace:
        """Create namespace for a project."""
        # Generate short hash for namespace
        prefix = hashlib.md5(project.encode()).hexdigest()[:8]

        namespace = MemoryNamespace(
            project=project,
            namespace_prefix=prefix,
            created_at=datetime.utcnow().isoformat(),
            settings={},
        )

        self._namespaces[project] = namespace
        self._save_namespace_cache()
        return namespace

    def _load_namespace_cache(self):
        """Load namespace mappings from disk."""
        if NAMESPACE_CACHE_FILE.exists():
            try:
                with open(NAMESPACE_CACHE_FILE, "r") as f:
                    data = json.load(f)
                for project, ns_data in data.items():
                    self._namespaces[project] = MemoryNamespace(
                        project=ns_data["project"],
                        namespace_prefix=ns_data["namespace_prefix"],
                        created_at=ns_data["created_at"],
                        settings=ns_data.get("settings", {}),
                    )
            except (json.JSONDecodeError, KeyError):
                self._namespaces = {}

    def _save_namespace_cache(self):
        """Save namespace mappings to disk."""
        NAMESPACE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {project: ns.to_dict() for project, ns in self._namespaces.items()}
        with open(NAMESPACE_CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def set_project(self, project: str):
        """Set current project context."""
        self.current_project = project

    def new_session(self) -> str:
        """Start a new session."""
        self.session_id = self._generate_id("session")
        return self.session_id

    # =========================================================================
    # Pipeline Memory Operations
    # =========================================================================

    def log_pipeline_run(
        self,
        result: PipelineResult,
        pipeline_name: str,
        input_text: str,
        project: Optional[str] = None,
    ) -> str:
        """
        Log a complete pipeline execution to memory.

        Args:
            result: PipelineResult from pipeline execution
            pipeline_name: Name of the pipeline
            input_text: Original input text
            project: Project scope (uses current if not specified)

        Returns:
            Memory ID for the logged run
        """
        project = project or self.current_project
        memory_id = self._generate_id("prun")
        timestamp = datetime.utcnow().isoformat()

        # Create pipeline memory record
        pipeline_memory = PipelineMemory(
            id=memory_id,
            pipeline_id=result.pipeline_id,
            pipeline_name=pipeline_name,
            project=project,
            session_id=self.session_id,
            timestamp=timestamp,
            input_text=input_text,
            final_output=result.final_output,
            node_outputs=result.node_outputs,
            success=result.success,
            total_latency_ms=result.total_latency_ms,
            total_tokens=result.total_tokens,
            errors=result.errors,
        )

        # Generate embedding text and vector
        embed_text = pipeline_memory.get_embedding_text()
        embedding = self._get_embedding(embed_text[:8000])

        # Prepare metadata for Pinecone
        metadata = {
            "text": embed_text[:8000],  # Limit for Pinecone
            "memory_type": MemoryType.PIPELINE_RUN.value,
            "scope": MemoryScope.PROJECT.value,
            "timestamp": timestamp,
            "project": project,
            "session_id": self.session_id,
            "pipeline_id": result.pipeline_id,
            "pipeline_name": pipeline_name,
            "success": result.success,
            "total_latency_ms": result.total_latency_ms,
            "input_preview": input_text[:500],
            "output_preview": result.final_output[:500],
            "node_count": len(result.node_outputs),
            "nodes": ",".join(result.node_outputs.keys()),
        }

        # Store in Pinecone with namespace
        namespace = self._get_namespace(project)
        self.index.upsert(
            vectors=[{"id": memory_id, "values": embedding, "metadata": metadata}],
            namespace=namespace,
        )

        # Also store full record locally for detailed retrieval
        self._store_full_record(memory_id, pipeline_memory.to_dict(), project)

        return memory_id

    def _store_full_record(self, memory_id: str, data: Dict[str, Any], project: str):
        """Store full record locally for detailed retrieval."""
        project_dir = PROJECTS_DIR / project / "memory"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Organize by date
        date_str = datetime.utcnow().strftime("%Y-%m")
        date_dir = project_dir / date_str
        date_dir.mkdir(exist_ok=True)

        file_path = date_dir / f"{memory_id}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_pipeline_run(
        self, memory_id: str, project: Optional[str] = None
    ) -> Optional[PipelineMemory]:
        """
        Retrieve a specific pipeline run by ID.

        Args:
            memory_id: Memory ID of the run
            project: Project to search in

        Returns:
            PipelineMemory or None if not found
        """
        project = project or self.current_project
        project_dir = PROJECTS_DIR / project / "memory"

        # Search in all month directories
        if project_dir.exists():
            for month_dir in project_dir.iterdir():
                if month_dir.is_dir():
                    file_path = month_dir / f"{memory_id}.json"
                    if file_path.exists():
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        return PipelineMemory.from_dict(data)

        return None

    def search_pipeline_outputs(
        self,
        query: str,
        project: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        success_only: bool = False,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[MemorySearchResult]:
        """
        Semantic search across pipeline outputs.

        Args:
            query: Search query text
            project: Project to search in (None = current project)
            pipeline_name: Filter by specific pipeline
            success_only: Only return successful runs
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        project = project or self.current_project
        namespace = self._get_namespace(project)

        # Build filter
        filter_dict: Dict[str, Any] = {
            "memory_type": {"$eq": MemoryType.PIPELINE_RUN.value}
        }

        if pipeline_name:
            filter_dict["pipeline_name"] = {"$eq": pipeline_name}

        if success_only:
            filter_dict["success"] = {"$eq": True}

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Query Pinecone with vector
        results = self.index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True,
            filter=filter_dict,
            namespace=namespace,
        )

        # Process results
        search_results: List[MemorySearchResult] = []
        for match in results.get("matches", []):
            if match["score"] >= min_score:
                meta = match.get("metadata", {})
                search_results.append(
                    MemorySearchResult(
                        id=match["id"],
                        score=match["score"],
                        memory_type=meta.get("memory_type", "unknown"),
                        timestamp=meta.get("timestamp", ""),
                        project=meta.get("project", project),
                        content_preview=meta.get("output_preview", "")[:200],
                        metadata={
                            "pipeline_name": meta.get("pipeline_name"),
                            "success": meta.get("success"),
                            "input_preview": meta.get("input_preview"),
                            "nodes": meta.get("nodes"),
                            "latency_ms": meta.get("total_latency_ms"),
                        },
                    )
                )

        return search_results

    def get_pipeline_history(
        self,
        pipeline_name: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 50,
        success_only: bool = False,
    ) -> List[PipelineMemory]:
        """
        Get chronological history of pipeline runs.

        Args:
            pipeline_name: Filter by pipeline name (None = all)
            project: Project to search in
            limit: Maximum results
            success_only: Only return successful runs

        Returns:
            List of PipelineMemory records (newest first)
        """
        project = project or self.current_project
        project_dir = PROJECTS_DIR / project / "memory"

        runs: List[PipelineMemory] = []

        if not project_dir.exists():
            return runs

        # Collect all run files
        for month_dir in sorted(project_dir.iterdir(), reverse=True):
            if not month_dir.is_dir():
                continue

            for file_path in sorted(month_dir.glob("prun_*.json"), reverse=True):
                if len(runs) >= limit:
                    break

                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    run = PipelineMemory.from_dict(data)

                    # Apply filters
                    if pipeline_name and run.pipeline_name != pipeline_name:
                        continue
                    if success_only and not run.success:
                        continue

                    runs.append(run)
                except (json.JSONDecodeError, KeyError):
                    continue

            if len(runs) >= limit:
                break

        return runs

    # =========================================================================
    # General Memory Operations
    # =========================================================================

    def store_entry(
        self,
        content: str,
        memory_type: MemoryType,
        scope: MemoryScope = MemoryScope.PROJECT,
        project: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a general memory entry.

        Args:
            content: Text content to store
            memory_type: Type of memory entry
            scope: Visibility scope
            project: Project context
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        project = project or self.current_project
        memory_id = self._generate_id("mem")
        timestamp = datetime.utcnow().isoformat()

        # Prepare Pinecone metadata
        pinecone_metadata = {
            "text": content[:8000],
            "memory_type": memory_type.value,
            "scope": scope.value,
            "timestamp": timestamp,
            "project": project,
            "session_id": self.session_id,
            "content_preview": content[:500],
        }

        if metadata:
            for k, v in metadata.items():
                pinecone_metadata[f"meta_{k}"] = str(v)[:500]

        # Generate embedding
        embedding = self._get_embedding(content[:8000])

        # Store with namespace
        namespace = self._get_namespace(project)
        self.index.upsert(
            vectors=[
                {"id": memory_id, "values": embedding, "metadata": pinecone_metadata}
            ],
            namespace=namespace,
        )

        return memory_id

    def search(
        self,
        query: str,
        project: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[MemorySearchResult]:
        """
        General semantic search across memory.

        Args:
            query: Search query
            project: Project to search
            memory_types: Filter by memory types
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        project = project or self.current_project
        namespace = self._get_namespace(project)

        # Build filter
        filter_dict: Dict[str, Any] = {}
        if memory_types:
            filter_dict["memory_type"] = {"$in": [mt.value for mt in memory_types]}

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Query with vector
        results = self.index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
            namespace=namespace,
        )

        # Process
        search_results: List[MemorySearchResult] = []
        for match in results.get("matches", []):
            if match["score"] >= min_score:
                meta = match.get("metadata", {})
                search_results.append(
                    MemorySearchResult(
                        id=match["id"],
                        score=match["score"],
                        memory_type=meta.get("memory_type", "unknown"),
                        timestamp=meta.get("timestamp", ""),
                        project=meta.get("project", project),
                        content_preview=meta.get("content_preview", "")[:200],
                        metadata={
                            k: v for k, v in meta.items() if k.startswith("meta_")
                        },
                    )
                )

        return search_results

    def delete_entry(self, memory_id: str, project: Optional[str] = None) -> bool:
        """
        Delete a memory entry.

        Args:
            memory_id: ID of entry to delete
            project: Project context

        Returns:
            True if deleted
        """
        project = project or self.current_project
        namespace = self._get_namespace(project)

        try:
            self.index.delete(ids=[memory_id], namespace=namespace)

            # Also delete local file if exists
            project_dir = PROJECTS_DIR / project / "memory"
            if project_dir.exists():
                for month_dir in project_dir.iterdir():
                    if month_dir.is_dir():
                        file_path = month_dir / f"{memory_id}.json"
                        if file_path.exists():
                            file_path.unlink()
                            break

            return True
        except Exception as e:
            import sys
            print(f"Warning: Failed to delete memory {memory_id}: {e}", file=sys.stderr)
            return False

    # =========================================================================
    # Namespace Management
    # =========================================================================

    def list_namespaces(self) -> List[MemoryNamespace]:
        """List all project namespaces."""
        return list(self._namespaces.values())

    def get_namespace_stats(self, project: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a project's memory namespace.

        Returns:
            Dictionary with namespace statistics
        """
        project = project or self.current_project
        namespace = self._get_namespace(project)

        try:
            # Get index stats for namespace
            stats = self.index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(namespace, {})

            # Count local files
            project_dir = PROJECTS_DIR / project / "memory"
            local_count = 0
            if project_dir.exists():
                local_count = sum(
                    1
                    for month_dir in project_dir.iterdir()
                    if month_dir.is_dir()
                    for _ in month_dir.glob("*.json")
                )

            return {
                "project": project,
                "namespace": namespace,
                "vector_count": ns_stats.get("vector_count", 0),
                "local_records": local_count,
            }
        except Exception as e:
            return {
                "project": project,
                "namespace": namespace,
                "error": str(e),
            }

    def clear_namespace(self, project: str) -> bool:
        """
        Clear all memory for a project.

        Args:
            project: Project to clear

        Returns:
            True if successful
        """
        namespace = self._get_namespace(project)

        try:
            # Delete all vectors in namespace
            self.index.delete(delete_all=True, namespace=namespace)

            # Clear local files
            project_dir = PROJECTS_DIR / project / "memory"
            if project_dir.exists():
                import shutil

                shutil.rmtree(project_dir)

            return True
        except Exception as e:
            import sys
            print(f"Warning: Failed to clear namespace {namespace}: {e}", file=sys.stderr)
            return False


# Singleton instance
_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Get or create MemoryStore singleton."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
