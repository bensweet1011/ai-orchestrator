"""
Autonomous execution module for AI Orchestrator.
Provides auto-retry, debugging, checkpoints, and escalation handling.
"""

from .state import (
    ExecutionMode,
    ExecutionState,
    ExecutionResult,
    ErrorType,
    RetryStrategy,
    FixType,
    RetryAttempt,
    Checkpoint,
    DebugAnalysis,
    Escalation,
    create_execution_state,
)

from .error_handler import ErrorHandler

from .debug_engine import DebugEngine

from .checkpoints import CheckpointManager

from .escalation import EscalationManager

from .autonomous import AutonomousExecutor

__all__ = [
    # State types
    "ExecutionMode",
    "ExecutionState",
    "ExecutionResult",
    "ErrorType",
    "RetryStrategy",
    "FixType",
    "RetryAttempt",
    "Checkpoint",
    "DebugAnalysis",
    "Escalation",
    "create_execution_state",
    # Components
    "ErrorHandler",
    "DebugEngine",
    "CheckpointManager",
    "EscalationManager",
    # Main executor
    "AutonomousExecutor",
]
