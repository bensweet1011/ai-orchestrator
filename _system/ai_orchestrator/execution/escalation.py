"""
Escalation system for autonomous execution.
Alerts users only when auto-debugging fails after all retry attempts.
"""

import json
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from .state import (
    Escalation,
    ExecutionState,
    RetryAttempt,
    ErrorType,
)
from ..config import TEMPLATES_DIR


# Default escalation storage location
ESCALATIONS_DIR = TEMPLATES_DIR / "escalations"


class EscalationManager:
    """
    Manage escalations when autonomous execution gets stuck.

    Escalations are created when:
    - All retry attempts are exhausted
    - A FATAL error is encountered
    - Debug engine has low confidence after multiple attempts
    - User-defined escalation triggers are met
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        notification_callback: Optional[Callable[[Escalation], None]] = None,
    ):
        """
        Initialize escalation manager.

        Args:
            storage_path: Directory for escalation storage
            notification_callback: Optional callback for real-time notifications
        """
        self.storage = storage_path or ESCALATIONS_DIR
        self.notification_callback = notification_callback
        self._ensure_storage()

    def _ensure_storage(self):
        """Ensure escalation storage directory exists."""
        self.storage.mkdir(parents=True, exist_ok=True)

    def _generate_escalation_id(self, pipeline_id: str, node_name: str) -> str:
        """Generate unique escalation ID."""
        ts = datetime.utcnow().isoformat()
        unique = hashlib.md5(f"{pipeline_id}{node_name}{ts}".encode()).hexdigest()[:8]
        return f"esc_{unique}"

    def should_escalate(
        self,
        state: ExecutionState,
        node_name: str,
        error_type: ErrorType,
        debug_confidence: Optional[float] = None,
    ) -> bool:
        """
        Determine if an escalation should be triggered.

        Args:
            state: Current execution state
            node_name: Name of the failing node
            error_type: Classification of the error
            debug_confidence: Confidence from debug engine (if available)

        Returns:
            True if escalation should be triggered
        """
        max_retries = state.get("max_retries", 3)

        # Count attempts for this node
        node_attempts = sum(
            1
            for attempt in state.get("retry_history", [])
            if attempt.get("node_name") == node_name
        )

        # Escalate if max retries exhausted
        if node_attempts >= max_retries:
            return True

        # Escalate immediately for FATAL errors
        if error_type == ErrorType.FATAL:
            return True

        # Escalate if debug engine has low confidence after 2+ attempts
        if (
            debug_confidence is not None
            and debug_confidence < 0.3
            and node_attempts >= 2
        ):
            return True

        # Check for user-defined escalation triggers
        escalation_triggers = state.get("custom", {}).get("escalation_triggers", [])
        for trigger in escalation_triggers:
            if self._check_trigger(trigger, state, node_name, error_type):
                return True

        return False

    def _check_trigger(
        self,
        trigger: Dict[str, Any],
        state: ExecutionState,
        node_name: str,
        error_type: ErrorType,
    ) -> bool:
        """Check if a user-defined trigger condition is met."""
        trigger_type = trigger.get("type")

        if trigger_type == "node":
            # Escalate when specific node fails
            return node_name in trigger.get("nodes", [])

        elif trigger_type == "error_type":
            # Escalate for specific error types
            return error_type.value in trigger.get("error_types", [])

        elif trigger_type == "total_retries":
            # Escalate after total retry threshold
            total = state.get("total_retry_count", 0)
            return total >= trigger.get("threshold", 10)

        return False

    def create_escalation(
        self,
        state: ExecutionState,
        node_name: str,
        reason: str,
        error_summary: str,
    ) -> Escalation:
        """
        Create an escalation record.

        Args:
            state: Current execution state
            node_name: Node that triggered escalation
            reason: Why escalation was triggered
            error_summary: Summary of the error

        Returns:
            Created Escalation object
        """
        pipeline_id = state.get("pipeline_id", "unknown")
        escalation_id = self._generate_escalation_id(pipeline_id, node_name)

        # Collect retry attempts for this node
        node_attempts = [
            RetryAttempt(
                attempt_number=a["attempt_number"],
                node_name=a["node_name"],
                timestamp=a["timestamp"],
                error_type=ErrorType(a["error_type"]),
                error_message=a["error_message"],
                fix_type=None,  # Will be populated if available
                fix_details=a.get("fix_details"),
                success=a["success"],
                latency_ms=a["latency_ms"],
            )
            for a in state.get("retry_history", [])
            if a.get("node_name") == node_name
        ]

        # Generate suggested actions
        suggested_actions = self._generate_suggestions(
            state, node_name, error_summary, node_attempts
        )

        escalation = Escalation(
            escalation_id=escalation_id,
            pipeline_id=pipeline_id,
            node_name=node_name,
            timestamp=datetime.utcnow().isoformat(),
            reason=reason,
            error_summary=error_summary,
            attempted_fixes=node_attempts,
            suggested_actions=suggested_actions,
            state_snapshot=self._sanitize_state_for_storage(state),
            resolved=False,
            resolution=None,
        )

        # Save escalation
        self._save_escalation(escalation)

        # Update state
        state["escalation_triggered"] = True
        state["escalation_id"] = escalation_id
        state["escalation_reason"] = reason

        # Notify if callback provided
        if self.notification_callback:
            try:
                self.notification_callback(escalation)
            except Exception:
                pass  # Don't let notification failure break execution

        return escalation

    def _generate_suggestions(
        self,
        state: ExecutionState,
        node_name: str,
        error_summary: str,
        attempts: List[RetryAttempt],
    ) -> List[str]:
        """Generate suggested actions for resolving the escalation."""
        suggestions = []
        error_lower = error_summary.lower()

        # Authentication issues
        if any(kw in error_lower for kw in ["auth", "key", "permission", "forbidden"]):
            suggestions.append("Check API key configuration in environment variables")
            suggestions.append("Verify API key has required permissions")

        # Rate limiting
        if any(kw in error_lower for kw in ["rate", "limit", "quota", "too many"]):
            suggestions.append("Wait a few minutes before retrying")
            suggestions.append("Consider switching to a different LLM provider")
            suggestions.append("Check API usage quotas in provider dashboard")

        # Context/token issues
        if any(kw in error_lower for kw in ["context", "token", "length", "too long"]):
            suggestions.append("Reduce input text length")
            suggestions.append("Use a model with larger context window")
            suggestions.append("Split the task into smaller chunks")

        # Output format issues
        if any(kw in error_lower for kw in ["format", "json", "parse", "invalid"]):
            suggestions.append("Review and clarify the system prompt")
            suggestions.append("Add explicit format instructions to the prompt")
            suggestions.append("Try a different LLM that follows instructions better")

        # Model issues
        if any(kw in error_lower for kw in ["model", "not found", "unavailable"]):
            suggestions.append("Check if the specified model ID is correct")
            suggestions.append("Verify the model is available in your region")
            suggestions.append("Try an alternative model")

        # Generic suggestions if none matched
        if not suggestions:
            suggestions.append("Review the error message and node configuration")
            suggestions.append("Try running with a different LLM")
            suggestions.append("Check the input data for issues")
            suggestions.append("Review the retry history for patterns")

        # Add resume option
        suggestions.append("Resume from last checkpoint after fixing the issue")

        return suggestions

    def _sanitize_state_for_storage(self, state: ExecutionState) -> Dict[str, Any]:
        """Prepare state for JSON serialization."""
        snapshot = deepcopy(dict(state))

        # Truncate large content
        if "outputs" in snapshot:
            for node_name, output in snapshot["outputs"].items():
                if isinstance(output, dict) and "content" in output:
                    content = output["content"]
                    if len(content) > 5000:
                        output["content"] = content[:5000] + "... [truncated]"

        if "input" in snapshot and len(snapshot["input"]) > 5000:
            snapshot["input"] = snapshot["input"][:5000] + "... [truncated]"

        return snapshot

    def _save_escalation(self, escalation: Escalation):
        """Save escalation to JSON file."""
        # Organize by date
        date_str = escalation.timestamp[:10]  # YYYY-MM-DD
        date_dir = self.storage / date_str
        date_dir.mkdir(exist_ok=True)

        file_path = date_dir / f"{escalation.escalation_id}.json"
        with open(file_path, "w") as f:
            json.dump(escalation.to_dict(), f, indent=2)

    def load_escalation(self, escalation_id: str) -> Optional[Escalation]:
        """Load an escalation by ID."""
        # Search in all date directories
        for date_dir in self.storage.iterdir():
            if date_dir.is_dir():
                file_path = date_dir / f"{escalation_id}.json"
                if file_path.exists():
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    return self._dict_to_escalation(data)

        return None

    def _dict_to_escalation(self, data: Dict[str, Any]) -> Escalation:
        """Convert dictionary to Escalation object."""
        # Reconstruct RetryAttempt objects
        attempted_fixes = []
        for attempt_data in data.get("attempted_fixes", []):
            attempt = RetryAttempt(
                attempt_number=attempt_data["attempt_number"],
                node_name=attempt_data["node_name"],
                timestamp=attempt_data["timestamp"],
                error_type=ErrorType(attempt_data["error_type"]),
                error_message=attempt_data["error_message"],
                fix_type=None,
                fix_details=attempt_data.get("fix_details"),
                success=attempt_data["success"],
                latency_ms=attempt_data["latency_ms"],
            )
            attempted_fixes.append(attempt)

        return Escalation(
            escalation_id=data["escalation_id"],
            pipeline_id=data["pipeline_id"],
            node_name=data["node_name"],
            timestamp=data["timestamp"],
            reason=data["reason"],
            error_summary=data["error_summary"],
            attempted_fixes=attempted_fixes,
            suggested_actions=data["suggested_actions"],
            state_snapshot=data["state_snapshot"],
            resolved=data.get("resolved", False),
            resolution=data.get("resolution"),
        )

    def resolve(
        self, escalation_id: str, resolution: str, resolver: str = "user"
    ) -> bool:
        """
        Mark an escalation as resolved.

        Args:
            escalation_id: Escalation to resolve
            resolution: How it was resolved
            resolver: Who resolved it

        Returns:
            True if resolved, False if not found
        """
        escalation = self.load_escalation(escalation_id)
        if not escalation:
            return False

        escalation.resolved = True
        escalation.resolution = resolution
        escalation.state_snapshot["resolved_by"] = resolver
        escalation.state_snapshot["resolved_at"] = datetime.utcnow().isoformat()

        self._save_escalation(escalation)
        return True

    def list_escalations(
        self,
        pipeline_id: Optional[str] = None,
        unresolved_only: bool = False,
        limit: int = 50,
    ) -> List[Escalation]:
        """
        List escalations with optional filtering.

        Args:
            pipeline_id: Filter by pipeline ID
            unresolved_only: Only show unresolved escalations
            limit: Maximum number to return

        Returns:
            List of Escalation objects
        """
        escalations = []

        # Search all date directories
        for date_dir in sorted(self.storage.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue

            for file_path in date_dir.glob("*.json"):
                if len(escalations) >= limit:
                    break

                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    escalation = self._dict_to_escalation(data)

                    # Apply filters
                    if pipeline_id and escalation.pipeline_id != pipeline_id:
                        continue
                    if unresolved_only and escalation.resolved:
                        continue

                    escalations.append(escalation)
                except (json.JSONDecodeError, IOError, KeyError):
                    continue

            if len(escalations) >= limit:
                break

        return escalations

    def get_escalation_summary(
        self, pipeline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for escalations.

        Returns:
            Dictionary with escalation statistics
        """
        escalations = self.list_escalations(pipeline_id=pipeline_id, limit=1000)

        # Count by resolution status
        resolved = sum(1 for e in escalations if e.resolved)
        unresolved = sum(1 for e in escalations if not e.resolved)

        # Count by node
        by_node: Dict[str, int] = {}
        for e in escalations:
            by_node[e.node_name] = by_node.get(e.node_name, 0) + 1

        # Most common reasons
        reasons: Dict[str, int] = {}
        for e in escalations:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1

        return {
            "total": len(escalations),
            "resolved": resolved,
            "unresolved": unresolved,
            "by_node": by_node,
            "common_reasons": dict(
                sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "latest_unresolved": next(
                (e.to_dict() for e in escalations if not e.resolved), None
            ),
        }

    def format_escalation_alert(self, escalation: Escalation) -> str:
        """
        Format an escalation for display/notification.

        Returns:
            Human-readable escalation summary
        """
        lines = [
            "=" * 60,
            "ESCALATION ALERT",
            "=" * 60,
            f"Pipeline: {escalation.pipeline_id}",
            f"Node: {escalation.node_name}",
            f"Time: {escalation.timestamp}",
            "",
            f"REASON: {escalation.reason}",
            "",
            "ERROR SUMMARY:",
            escalation.error_summary,
            "",
            f"ATTEMPTED FIXES: {len(escalation.attempted_fixes)}",
        ]

        for attempt in escalation.attempted_fixes[-3:]:  # Show last 3
            lines.append(
                f"  - Attempt {attempt.attempt_number}: "
                f"{attempt.fix_type.value if attempt.fix_type else 'none'} - "
                f"{'Success' if attempt.success else 'Failed'}"
            )

        lines.extend(["", "SUGGESTED ACTIONS:"])
        for i, action in enumerate(escalation.suggested_actions, 1):
            lines.append(f"  {i}. {action}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)
