"""
Error classification and retry logic for autonomous execution.
Determines how to handle different types of errors.
"""

import re
from typing import Any, Dict, Optional
from datetime import datetime

from .state import (
    ErrorType,
    RetryStrategy,
    RetryAttempt,
    ExecutionState,
    FixType,
)


# Error patterns for classification
TRANSIENT_PATTERNS = [
    r"rate.?limit",
    r"too.?many.?requests",
    r"429",
    r"timeout",
    r"timed?.?out",
    r"connection.?(error|refused|reset)",
    r"network",
    r"temporary",
    r"unavailable",
    r"503",
    r"502",
    r"overloaded",
    r"capacity",
    r"retry.?after",
]

FATAL_PATTERNS = [
    r"invalid.?api.?key",
    r"authentication",
    r"unauthorized",
    r"401",
    r"403",
    r"forbidden",
    r"invalid.?model",
    r"model.?not.?found",
    r"permission.?denied",
    r"quota.?exceeded",
    r"billing",
    r"account.?(suspended|disabled)",
]

FIXABLE_PATTERNS = [
    r"invalid.?(json|format|output)",
    r"parse.?error",
    r"missing.?(field|key|parameter)",
    r"schema.?validation",
    r"expected.*but.?got",
    r"content.?filter",
    r"safety",
    r"max.?tokens",
    r"context.?length",
    r"too.?long",
    r"truncat",
]


class ErrorHandler:
    """
    Classify errors and manage retry logic.
    Determines the appropriate response strategy for each error type.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._transient_re = [re.compile(p, re.IGNORECASE) for p in TRANSIENT_PATTERNS]
        self._fatal_re = [re.compile(p, re.IGNORECASE) for p in FATAL_PATTERNS]
        self._fixable_re = [re.compile(p, re.IGNORECASE) for p in FIXABLE_PATTERNS]

    def classify_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> ErrorType:
        """
        Classify an error into TRANSIENT, FIXABLE, FATAL, or UNKNOWN.

        Args:
            error: The exception that occurred
            context: Optional context about the error (node, input, etc.)

        Returns:
            ErrorType classification
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        combined = f"{error_str} {error_type_name}"

        # Check fatal patterns first (highest priority)
        for pattern in self._fatal_re:
            if pattern.search(combined):
                return ErrorType.FATAL

        # Check transient patterns
        for pattern in self._transient_re:
            if pattern.search(combined):
                return ErrorType.TRANSIENT

        # Check fixable patterns
        for pattern in self._fixable_re:
            if pattern.search(combined):
                return ErrorType.FIXABLE

        # Additional heuristics based on exception type
        if isinstance(error, (TimeoutError, ConnectionError)):
            return ErrorType.TRANSIENT

        if isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorType.FIXABLE

        if isinstance(error, PermissionError):
            return ErrorType.FATAL

        return ErrorType.UNKNOWN

    def should_retry(
        self, state: ExecutionState, error_type: ErrorType, node_name: str
    ) -> bool:
        """
        Determine if a retry should be attempted.

        Args:
            state: Current execution state
            error_type: Classification of the error
            node_name: Name of the failed node

        Returns:
            True if retry should be attempted
        """
        # Never retry fatal errors
        if error_type == ErrorType.FATAL:
            return False

        # Check retry count for this node
        node_attempts = sum(
            1
            for attempt in state.get("retry_history", [])
            if attempt.get("node_name") == node_name
        )

        return node_attempts < self.max_retries

    def get_retry_strategy(
        self, error_type: ErrorType, attempt_number: int, error_message: str
    ) -> RetryStrategy:
        """
        Determine the retry strategy based on error type and attempt number.

        Args:
            error_type: Classification of the error
            attempt_number: Which retry attempt this is (1-based)
            error_message: The error message for additional context

        Returns:
            RetryStrategy to use
        """
        if error_type == ErrorType.FATAL:
            return RetryStrategy.ABORT

        if error_type == ErrorType.TRANSIENT:
            # Use backoff for transient errors
            return RetryStrategy.BACKOFF

        if error_type == ErrorType.FIXABLE:
            # First attempt: try modifying prompt
            if attempt_number == 1:
                return RetryStrategy.MODIFY_PROMPT
            # Second attempt: try different LLM
            elif attempt_number == 2:
                return RetryStrategy.FALLBACK_LLM
            # Third attempt: one more prompt modification
            else:
                return RetryStrategy.MODIFY_PROMPT

        # Unknown errors: progressive strategy
        if attempt_number == 1:
            return RetryStrategy.IMMEDIATE
        elif attempt_number == 2:
            return RetryStrategy.MODIFY_PROMPT
        else:
            return RetryStrategy.FALLBACK_LLM

    def calculate_backoff(self, attempt_number: int, base_delay: float = 1.0) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt_number: Which attempt this is (1-based)
            base_delay: Base delay in seconds

        Returns:
            Delay in seconds before retry
        """
        # Exponential backoff with jitter
        import random

        delay = base_delay * (2 ** (attempt_number - 1))
        jitter = random.uniform(0, delay * 0.1)
        return min(delay + jitter, 60.0)  # Cap at 60 seconds

    def record_attempt(
        self,
        state: ExecutionState,
        node_name: str,
        error: Exception,
        error_type: ErrorType,
        fix_type: Optional[FixType] = None,
        fix_details: Optional[str] = None,
        success: bool = False,
        latency_ms: int = 0,
    ) -> RetryAttempt:
        """
        Record a retry attempt in the execution state.

        Args:
            state: Current execution state
            node_name: Name of the node being retried
            error: The exception that occurred
            error_type: Classification of the error
            fix_type: Type of fix applied (if any)
            fix_details: Details of the fix applied
            success: Whether the attempt succeeded
            latency_ms: Time taken for the attempt

        Returns:
            The created RetryAttempt record
        """
        # Count previous attempts for this node
        node_attempts = sum(
            1
            for attempt in state.get("retry_history", [])
            if attempt.get("node_name") == node_name
        )

        attempt = RetryAttempt(
            attempt_number=node_attempts + 1,
            node_name=node_name,
            timestamp=datetime.utcnow().isoformat(),
            error_type=error_type,
            error_message=str(error),
            fix_type=fix_type,
            fix_details=fix_details,
            success=success,
            latency_ms=latency_ms,
        )

        # Update state
        retry_history = list(state.get("retry_history", []))
        retry_history.append(attempt.to_dict())
        state["retry_history"] = retry_history
        state["total_retry_count"] = state.get("total_retry_count", 0) + 1

        return attempt

    def get_node_attempt_count(self, state: ExecutionState, node_name: str) -> int:
        """Get the number of retry attempts for a specific node."""
        return sum(
            1
            for attempt in state.get("retry_history", [])
            if attempt.get("node_name") == node_name
        )

    def get_error_summary(self, state: ExecutionState) -> Dict[str, Any]:
        """
        Generate a summary of all errors encountered during execution.

        Returns:
            Dictionary with error statistics and details
        """
        retry_history = state.get("retry_history", [])
        errors = state.get("errors", [])

        # Count by error type
        error_counts = {}
        for attempt in retry_history:
            etype = attempt.get("error_type", "unknown")
            error_counts[etype] = error_counts.get(etype, 0) + 1

        # Group by node
        by_node = {}
        for attempt in retry_history:
            node = attempt.get("node_name", "unknown")
            if node not in by_node:
                by_node[node] = []
            by_node[node].append(attempt)

        return {
            "total_attempts": len(retry_history),
            "successful_retries": sum(1 for a in retry_history if a.get("success")),
            "failed_retries": sum(1 for a in retry_history if not a.get("success")),
            "error_counts_by_type": error_counts,
            "attempts_by_node": {k: len(v) for k, v in by_node.items()},
            "final_errors": errors,
        }

    def format_error_for_debug(
        self,
        error: Exception,
        error_type: ErrorType,
        node_name: str,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format error information for the debug engine.

        Returns:
            Formatted string for LLM analysis
        """
        lines = [
            f"ERROR IN NODE: {node_name}",
            f"ERROR TYPE: {error_type.value}",
            f"EXCEPTION: {type(error).__name__}",
            f"MESSAGE: {str(error)}",
            "",
            "INPUT TEXT (first 500 chars):",
            input_text[:500] + ("..." if len(input_text) > 500 else ""),
        ]

        if context:
            lines.extend(["", "ADDITIONAL CONTEXT:"])
            for key, value in context.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)
