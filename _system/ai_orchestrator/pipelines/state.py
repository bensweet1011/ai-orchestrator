"""
Pipeline state management and schemas.
Defines the state structure for LangGraph pipelines.
"""

from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field


class NodeOutput(TypedDict):
    """Output from a single node execution."""

    content: str
    model: str
    provider: str
    latency_ms: int
    tokens_used: Optional[Dict[str, int]]


class PipelineState(TypedDict, total=False):
    """
    State object passed through the LangGraph pipeline.
    Uses TypedDict for LangGraph compatibility.
    """

    # Core state
    input: str  # Original user input
    outputs: Dict[str, NodeOutput]  # node_name -> output
    current_node: str  # Currently executing node
    final_output: str  # Final pipeline result

    # Routing state (for conditionals)
    route: Optional[str]  # Next route to take
    route_reason: Optional[str]  # Why this route was chosen

    # Metadata
    pipeline_id: str
    started_at: str
    errors: List[Dict[str, Any]]  # List of {node, error, timestamp}

    # Custom state (user-defined)
    custom: Dict[str, Any]


def create_initial_state(input_text: str, pipeline_id: str = "default") -> PipelineState:
    """Create initial pipeline state from user input."""
    return PipelineState(
        input=input_text,
        outputs={},
        current_node="__start__",
        final_output="",
        route=None,
        route_reason=None,
        pipeline_id=pipeline_id,
        started_at=datetime.utcnow().isoformat(),
        errors=[],
        custom={},
    )


# Pydantic models for schema validation


class NodeInputSchema(BaseModel):
    """Base schema for node inputs."""

    text: str = Field(..., description="Text input to process")
    context: Optional[str] = Field(None, description="Additional context")


class NodeOutputSchema(BaseModel):
    """Base schema for node outputs."""

    result: str = Field(..., description="Processing result")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SummaryOutputSchema(NodeOutputSchema):
    """Schema for summarization node outputs."""

    summary: str = Field(..., description="Summarized text")
    key_points: List[str] = Field(default_factory=list)


class CritiqueOutputSchema(NodeOutputSchema):
    """Schema for critique/review node outputs."""

    critique: str = Field(..., description="Critique of the input")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    score: Optional[float] = Field(None, ge=0.0, le=10.0)


class ClassificationOutputSchema(NodeOutputSchema):
    """Schema for classification node outputs."""

    category: str = Field(..., description="Classified category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    alternatives: List[Dict[str, float]] = Field(default_factory=list)


@dataclass
class PipelineResult:
    """Final result from pipeline execution."""

    success: bool
    final_output: str
    node_outputs: Dict[str, NodeOutput]
    total_latency_ms: int
    total_tokens: Dict[str, int]
    errors: List[Dict[str, Any]]
    pipeline_id: str
    started_at: str
    completed_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_output": self.final_output,
            "node_outputs": self.node_outputs,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "pipeline_id": self.pipeline_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
