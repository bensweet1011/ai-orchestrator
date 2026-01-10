"""
Memory state types for cross-session persistence.
Defines structures for storing and retrieving pipeline outputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class MemoryType(Enum):
    """Types of memory entries."""

    INTERACTION = "interaction"  # Single LLM call
    PIPELINE_RUN = "pipeline_run"  # Complete pipeline execution
    SYNTHESIS = "synthesis"  # Combined output from multiple runs
    CHECKPOINT = "checkpoint"  # Execution checkpoint
    USER_NOTE = "user_note"  # Manual user annotation


class MemoryScope(Enum):
    """Scope of memory visibility."""

    GLOBAL = "global"  # Visible across all projects
    PROJECT = "project"  # Visible within project
    PIPELINE = "pipeline"  # Visible within specific pipeline
    SESSION = "session"  # Visible within session only


@dataclass
class MemoryEntry:
    """Base class for any memory entry."""

    id: str
    memory_type: MemoryType
    scope: MemoryScope
    timestamp: str
    project: str
    content: str  # Main text content for embedding
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "scope": self.scope.value,
            "timestamp": self.timestamp,
            "project": self.project,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            memory_type=MemoryType(data["memory_type"]),
            scope=MemoryScope(data["scope"]),
            timestamp=data["timestamp"],
            project=data["project"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PipelineMemory:
    """
    Complete record of a pipeline execution.
    Stores input, all node outputs, and execution metadata.
    """

    id: str
    pipeline_id: str
    pipeline_name: str
    project: str
    session_id: str
    timestamp: str

    # Execution data
    input_text: str
    final_output: str
    node_outputs: Dict[str, Dict[str, Any]]  # node_name -> output dict
    success: bool

    # Metrics
    total_latency_ms: int
    total_tokens: Dict[str, int]

    # Optional
    errors: List[Dict[str, Any]] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "project": self.project,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "input_text": self.input_text,
            "final_output": self.final_output,
            "node_outputs": self.node_outputs,
            "success": self.success,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineMemory":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            pipeline_id=data["pipeline_id"],
            pipeline_name=data["pipeline_name"],
            project=data["project"],
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            input_text=data["input_text"],
            final_output=data["final_output"],
            node_outputs=data["node_outputs"],
            success=data["success"],
            total_latency_ms=data["total_latency_ms"],
            total_tokens=data["total_tokens"],
            errors=data.get("errors", []),
            custom_metadata=data.get("custom_metadata", {}),
        )

    def get_embedding_text(self) -> str:
        """Generate text for embedding (input + output summary)."""
        node_summary = " | ".join(
            f"{name}: {output.get('content', '')[:200]}"
            for name, output in self.node_outputs.items()
        )
        return f"INPUT: {self.input_text[:1000]}\n\nOUTPUT: {self.final_output[:1000]}\n\nNODES: {node_summary}"


@dataclass
class SynthesisRequest:
    """Request to synthesize multiple pipeline outputs."""

    id: str
    project: str
    timestamp: str
    source_run_ids: List[str]  # Pipeline run IDs to synthesize
    synthesis_type: str  # "combine", "compare", "extract_patterns"
    instructions: Optional[str] = None  # Custom synthesis instructions


@dataclass
class SynthesisResult:
    """Result of synthesizing multiple pipeline outputs."""

    id: str
    request_id: str
    project: str
    timestamp: str

    # Source data
    source_run_ids: List[str]
    source_count: int

    # Result
    synthesized_output: str
    synthesis_type: str

    # Metadata
    model_used: str
    latency_ms: int
    token_usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "request_id": self.request_id,
            "project": self.project,
            "timestamp": self.timestamp,
            "source_run_ids": self.source_run_ids,
            "source_count": self.source_count,
            "synthesized_output": self.synthesized_output,
            "synthesis_type": self.synthesis_type,
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesisResult":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            request_id=data["request_id"],
            project=data["project"],
            timestamp=data["timestamp"],
            source_run_ids=data["source_run_ids"],
            source_count=data["source_count"],
            synthesized_output=data["synthesized_output"],
            synthesis_type=data["synthesis_type"],
            model_used=data["model_used"],
            latency_ms=data["latency_ms"],
            token_usage=data.get("token_usage", {}),
        )


@dataclass
class MemoryNamespace:
    """
    Configuration for project-scoped memory isolation.
    Each project gets its own namespace prefix in Pinecone.
    """

    project: str
    namespace_prefix: str
    created_at: str
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_namespace(self) -> str:
        """Get full namespace string for Pinecone."""
        return f"proj_{self.namespace_prefix}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project": self.project,
            "namespace_prefix": self.namespace_prefix,
            "created_at": self.created_at,
            "settings": self.settings,
        }


@dataclass
class ContextConfig:
    """Configuration for context injection into nodes."""

    enabled: bool = True
    max_results: int = 3  # Number of past results to inject
    min_score: float = 0.5  # Minimum similarity score
    scope: MemoryScope = MemoryScope.PROJECT  # Where to search
    include_node_outputs: bool = True  # Include specific node outputs
    recency_weight: float = 0.2  # Weight for recent results (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "max_results": self.max_results,
            "min_score": self.min_score,
            "scope": self.scope.value,
            "include_node_outputs": self.include_node_outputs,
            "recency_weight": self.recency_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            max_results=data.get("max_results", 3),
            min_score=data.get("min_score", 0.5),
            scope=MemoryScope(data.get("scope", "project")),
            include_node_outputs=data.get("include_node_outputs", True),
            recency_weight=data.get("recency_weight", 0.2),
        )


class MemorySearchResult(TypedDict):
    """Result from a memory search."""

    id: str
    score: float
    memory_type: str
    timestamp: str
    project: str
    content_preview: str
    metadata: Dict[str, Any]
