"""
Execution state management for autonomous pipeline execution.
Extends PipelineState with retry, checkpoint, and escalation tracking.
"""

from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ExecutionMode(Enum):
    """Execution mode for pipeline runs."""

    MANUAL = "manual"  # No auto-retry, no checkpoints
    SUPERVISED = "supervised"  # Auto-retry, checkpoints require approval
    AUTONOMOUS = "autonomous"  # Full auto-retry, auto-debug, minimal intervention


class ErrorType(Enum):
    """Classification of errors for handling strategy."""

    TRANSIENT = "transient"  # Rate limit, timeout, network - retry with backoff
    FIXABLE = "fixable"  # Bad format, missing field - auto-debug can fix
    FATAL = "fatal"  # Auth failure, invalid model - escalate immediately
    UNKNOWN = "unknown"  # Unclassified - attempt debug then escalate


class RetryStrategy(Enum):
    """Strategy for retry attempts."""

    IMMEDIATE = "immediate"  # Retry immediately
    BACKOFF = "backoff"  # Exponential backoff
    FALLBACK_LLM = "fallback_llm"  # Try different LLM
    MODIFY_PROMPT = "modify_prompt"  # Adjust prompt based on error
    ABORT = "abort"  # Stop retrying


class FixType(Enum):
    """Types of fixes that can be applied."""

    PROMPT_MODIFICATION = "prompt_modification"
    TEMPERATURE_CHANGE = "temperature_change"
    LLM_SWITCH = "llm_switch"
    INPUT_TRANSFORM = "input_transform"
    MAX_TOKENS_INCREASE = "max_tokens_increase"
    RETRY_ONLY = "retry_only"  # No modification, just retry


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""

    attempt_number: int
    node_name: str
    timestamp: str
    error_type: ErrorType
    error_message: str
    fix_type: Optional[FixType]
    fix_details: Optional[str]
    success: bool
    latency_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "node_name": self.node_name,
            "timestamp": self.timestamp,
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "fix_type": self.fix_type.value if self.fix_type else None,
            "fix_details": self.fix_details,
            "success": self.success,
            "latency_ms": self.latency_ms,
        }


@dataclass
class Checkpoint:
    """Snapshot of execution state at a point in time."""

    checkpoint_id: str
    pipeline_id: str
    node_name: str
    timestamp: str
    checkpoint_type: str  # "auto" | "user_defined" | "error"
    state_snapshot: Dict[str, Any]
    requires_approval: bool
    approved: Optional[bool] = None
    approved_at: Optional[str] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "pipeline_id": self.pipeline_id,
            "node_name": self.node_name,
            "timestamp": self.timestamp,
            "checkpoint_type": self.checkpoint_type,
            "state_snapshot": self.state_snapshot,
            "requires_approval": self.requires_approval,
            "approved": self.approved,
            "approved_at": self.approved_at,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(**data)


@dataclass
class DebugAnalysis:
    """Result of LLM-powered error analysis."""

    root_cause: str
    fix_type: FixType
    specific_fix: str
    confidence: float  # 0.0 - 1.0
    reasoning: str
    alternative_fixes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "fix_type": self.fix_type.value,
            "specific_fix": self.specific_fix,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternative_fixes": self.alternative_fixes,
        }


@dataclass
class Escalation:
    """Escalation record when auto-debug fails."""

    escalation_id: str
    pipeline_id: str
    node_name: str
    timestamp: str
    reason: str
    error_summary: str
    attempted_fixes: List[RetryAttempt]
    suggested_actions: List[str]
    state_snapshot: Dict[str, Any]
    resolved: bool = False
    resolution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escalation_id": self.escalation_id,
            "pipeline_id": self.pipeline_id,
            "node_name": self.node_name,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "error_summary": self.error_summary,
            "attempted_fixes": [a.to_dict() for a in self.attempted_fixes],
            "suggested_actions": self.suggested_actions,
            "state_snapshot": self.state_snapshot,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


class ExecutionState(TypedDict, total=False):
    """
    Extended state for autonomous pipeline execution.
    Includes all PipelineState fields plus execution control.
    """

    # Original PipelineState fields
    input: str
    outputs: Dict[str, Any]
    current_node: str
    final_output: str
    route: Optional[str]
    route_reason: Optional[str]
    pipeline_id: str
    started_at: str
    errors: List[Dict[str, Any]]
    custom: Dict[str, Any]

    # Execution control
    execution_mode: str  # ExecutionMode value
    max_retries: int
    current_attempt: int

    # Auto-debugging
    retry_history: List[Dict[str, Any]]  # List of RetryAttempt dicts
    last_error_analysis: Optional[Dict[str, Any]]  # DebugAnalysis dict
    applied_fixes: List[str]
    total_retry_count: int

    # Checkpoints
    checkpoints: Dict[str, Dict[str, Any]]  # checkpoint_id -> Checkpoint dict
    pending_approval: Optional[str]  # Checkpoint ID awaiting approval
    approved_checkpoints: List[str]
    checkpoint_nodes: List[str]  # Nodes that require checkpoints

    # Escalation
    escalation_triggered: bool
    escalation_id: Optional[str]
    escalation_reason: Optional[str]

    # Execution trace
    trace_log: List[Dict[str, Any]]
    completed_nodes: List[str]


def create_execution_state(
    input_text: str,
    pipeline_id: str,
    execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS,
    max_retries: int = 3,
    checkpoint_nodes: Optional[List[str]] = None,
) -> ExecutionState:
    """Create initial execution state for autonomous pipeline run."""
    return ExecutionState(
        # PipelineState fields
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
        # Execution control
        execution_mode=execution_mode.value,
        max_retries=max_retries,
        current_attempt=0,
        # Auto-debugging
        retry_history=[],
        last_error_analysis=None,
        applied_fixes=[],
        total_retry_count=0,
        # Checkpoints
        checkpoints={},
        pending_approval=None,
        approved_checkpoints=[],
        checkpoint_nodes=checkpoint_nodes or [],
        # Escalation
        escalation_triggered=False,
        escalation_id=None,
        escalation_reason=None,
        # Trace
        trace_log=[],
        completed_nodes=[],
    )


@dataclass
class ExecutionResult:
    """Final result from autonomous execution."""

    success: bool
    final_output: str
    node_outputs: Dict[str, Any]
    total_latency_ms: int
    total_tokens: Dict[str, int]
    errors: List[Dict[str, Any]]
    retry_attempts: List[RetryAttempt]
    checkpoints_created: int
    escalations: List[Escalation]
    execution_mode: ExecutionMode
    pipeline_id: str
    started_at: str
    completed_at: str
    trace_log: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_output": self.final_output,
            "node_outputs": self.node_outputs,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "retry_attempts": [a.to_dict() for a in self.retry_attempts],
            "checkpoints_created": self.checkpoints_created,
            "escalations": [e.to_dict() for e in self.escalations],
            "execution_mode": self.execution_mode.value,
            "pipeline_id": self.pipeline_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "trace_log": self.trace_log,
        }

    @property
    def had_retries(self) -> bool:
        return len(self.retry_attempts) > 0

    @property
    def was_escalated(self) -> bool:
        return len(self.escalations) > 0
