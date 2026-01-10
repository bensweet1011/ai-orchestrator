"""
Checkpoint management for autonomous execution.
Save, load, and manage execution checkpoints for recovery and approval workflows.
"""

import json
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from .state import Checkpoint, ExecutionState
from ..config import TEMPLATES_DIR


# Default checkpoint storage location
CHECKPOINTS_DIR = TEMPLATES_DIR / "checkpoints"


class CheckpointManager:
    """
    Manage execution checkpoints for recovery and approval workflows.

    Checkpoints allow:
    - Recovery from failures (resume from last successful node)
    - User approval gates (pause at defined points)
    - Execution history and audit trails
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize checkpoint manager.

        Args:
            storage_path: Directory for checkpoint storage (default: _templates/checkpoints)
        """
        self.storage = storage_path or CHECKPOINTS_DIR
        self._ensure_storage()

    def _ensure_storage(self):
        """Ensure checkpoint storage directory exists."""
        self.storage.mkdir(parents=True, exist_ok=True)

    def _generate_checkpoint_id(self, pipeline_id: str, node_name: str) -> str:
        """Generate unique checkpoint ID."""
        ts = datetime.utcnow().isoformat()
        unique = hashlib.md5(f"{pipeline_id}{node_name}{ts}".encode()).hexdigest()[:8]
        return f"ckpt_{unique}"

    def save_checkpoint(
        self,
        state: ExecutionState,
        node_name: str,
        checkpoint_type: str = "auto",
        requires_approval: bool = False,
    ) -> Checkpoint:
        """
        Save execution state as a checkpoint after successful node execution.

        Args:
            state: Current execution state to snapshot
            node_name: Name of the node that just completed
            checkpoint_type: "auto" | "user_defined" | "error"
            requires_approval: Whether this checkpoint needs user approval

        Returns:
            Created Checkpoint object
        """
        pipeline_id = state.get("pipeline_id", "unknown")
        checkpoint_id = self._generate_checkpoint_id(pipeline_id, node_name)

        # Create checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            pipeline_id=pipeline_id,
            node_name=node_name,
            timestamp=datetime.utcnow().isoformat(),
            checkpoint_type=checkpoint_type,
            state_snapshot=self._sanitize_state_for_storage(state),
            requires_approval=requires_approval,
            approved=None if requires_approval else True,
            approved_at=None if requires_approval else datetime.utcnow().isoformat(),
        )

        # Save to file
        self._save_checkpoint_file(checkpoint)

        # Update state with checkpoint reference
        checkpoints = dict(state.get("checkpoints", {}))
        checkpoints[checkpoint_id] = checkpoint.to_dict()
        state["checkpoints"] = checkpoints

        if requires_approval:
            state["pending_approval"] = checkpoint_id

        return checkpoint

    def _sanitize_state_for_storage(self, state: ExecutionState) -> Dict[str, Any]:
        """
        Prepare state for JSON serialization.
        Remove non-serializable objects and truncate large content.
        """
        # Deep copy to avoid modifying original
        snapshot = deepcopy(dict(state))

        # Truncate large content in outputs
        if "outputs" in snapshot:
            for node_name, output in snapshot["outputs"].items():
                if isinstance(output, dict) and "content" in output:
                    content = output["content"]
                    if len(content) > 10000:
                        output["content"] = content[:10000] + "... [truncated]"

        # Truncate input if very large
        if "input" in snapshot and len(snapshot["input"]) > 10000:
            snapshot["input"] = snapshot["input"][:10000] + "... [truncated]"

        return snapshot

    def _save_checkpoint_file(self, checkpoint: Checkpoint):
        """Save checkpoint to JSON file."""
        # Organize by pipeline_id
        pipeline_dir = self.storage / checkpoint.pipeline_id
        pipeline_dir.mkdir(exist_ok=True)

        file_path = pipeline_dir / f"{checkpoint.checkpoint_id}.json"
        with open(file_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Load a checkpoint by ID.

        Args:
            checkpoint_id: Unique checkpoint identifier

        Returns:
            Checkpoint object or None if not found
        """
        # Search for checkpoint file
        for pipeline_dir in self.storage.iterdir():
            if pipeline_dir.is_dir():
                file_path = pipeline_dir / f"{checkpoint_id}.json"
                if file_path.exists():
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    return Checkpoint.from_dict(data)

        return None

    def restore_state(self, checkpoint_id: str) -> Optional[ExecutionState]:
        """
        Restore execution state from a checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore from

        Returns:
            ExecutionState ready for resumption, or None if not found
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return None

        # Restore state from snapshot
        state = ExecutionState(**checkpoint.state_snapshot)

        # Mark that we're resuming from checkpoint
        state["custom"]["resumed_from"] = checkpoint_id
        state["custom"]["resumed_at"] = datetime.utcnow().isoformat()

        return state

    def list_checkpoints(
        self,
        pipeline_id: Optional[str] = None,
        checkpoint_type: Optional[str] = None,
        pending_only: bool = False,
    ) -> List[Checkpoint]:
        """
        List checkpoints with optional filtering.

        Args:
            pipeline_id: Filter by pipeline ID
            checkpoint_type: Filter by type ("auto", "user_defined", "error")
            pending_only: Only show checkpoints pending approval

        Returns:
            List of matching Checkpoint objects
        """
        checkpoints = []

        # Determine which directories to search
        if pipeline_id:
            search_dirs = [self.storage / pipeline_id]
        else:
            search_dirs = [d for d in self.storage.iterdir() if d.is_dir()]

        for pipeline_dir in search_dirs:
            if not pipeline_dir.exists():
                continue

            for file_path in pipeline_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    checkpoint = Checkpoint.from_dict(data)

                    # Apply filters
                    if (
                        checkpoint_type
                        and checkpoint.checkpoint_type != checkpoint_type
                    ):
                        continue
                    if pending_only and checkpoint.approved is not None:
                        continue

                    checkpoints.append(checkpoint)
                except (json.JSONDecodeError, IOError, KeyError):
                    continue

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return checkpoints

    def request_approval(self, checkpoint_id: str, reason: str) -> Optional[Checkpoint]:
        """
        Mark a checkpoint as requiring approval.

        Args:
            checkpoint_id: Checkpoint to request approval for
            reason: Reason for the approval request

        Returns:
            Updated Checkpoint or None if not found
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return None

        checkpoint.requires_approval = True
        checkpoint.approved = None
        checkpoint.state_snapshot["approval_reason"] = reason

        self._save_checkpoint_file(checkpoint)
        return checkpoint

    def approve(self, checkpoint_id: str, approver: str = "user") -> bool:
        """
        Approve a checkpoint, allowing execution to continue.

        Args:
            checkpoint_id: Checkpoint to approve
            approver: Who approved (for audit)

        Returns:
            True if approved, False if checkpoint not found
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return False

        checkpoint.approved = True
        checkpoint.approved_at = datetime.utcnow().isoformat()
        checkpoint.state_snapshot["approved_by"] = approver

        self._save_checkpoint_file(checkpoint)
        return True

    def reject(self, checkpoint_id: str, reason: str, rejector: str = "user") -> bool:
        """
        Reject a checkpoint, signaling execution should not continue.

        Args:
            checkpoint_id: Checkpoint to reject
            reason: Reason for rejection
            rejector: Who rejected (for audit)

        Returns:
            True if rejected, False if checkpoint not found
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return False

        checkpoint.approved = False
        checkpoint.rejection_reason = reason
        checkpoint.state_snapshot["rejected_by"] = rejector
        checkpoint.state_snapshot["rejected_at"] = datetime.utcnow().isoformat()

        self._save_checkpoint_file(checkpoint)
        return True

    def is_approved(self, checkpoint_id: str) -> Optional[bool]:
        """
        Check if a checkpoint is approved.

        Returns:
            True if approved, False if rejected, None if pending
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return None
        return checkpoint.approved

    def get_latest_checkpoint(self, pipeline_id: str) -> Optional[Checkpoint]:
        """
        Get the most recent checkpoint for a pipeline.

        Args:
            pipeline_id: Pipeline to get checkpoint for

        Returns:
            Most recent Checkpoint or None
        """
        checkpoints = self.list_checkpoints(pipeline_id=pipeline_id)
        return checkpoints[0] if checkpoints else None

    def get_recovery_point(self, pipeline_id: str) -> Optional[Checkpoint]:
        """
        Get the best checkpoint to recover from (latest approved auto-checkpoint).

        Args:
            pipeline_id: Pipeline to find recovery point for

        Returns:
            Best Checkpoint for recovery or None
        """
        checkpoints = self.list_checkpoints(pipeline_id=pipeline_id)

        # Find latest approved checkpoint
        for checkpoint in checkpoints:
            if checkpoint.approved and checkpoint.checkpoint_type in [
                "auto",
                "user_defined",
            ]:
                return checkpoint

        return None

    def cleanup_old_checkpoints(
        self,
        pipeline_id: str,
        keep_count: int = 10,
        keep_approved: bool = True,
    ) -> int:
        """
        Remove old checkpoints to save storage.

        Args:
            pipeline_id: Pipeline to clean up
            keep_count: Number of recent checkpoints to keep
            keep_approved: Whether to always keep approved checkpoints

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints(pipeline_id=pipeline_id)

        # Separate checkpoints to keep and delete
        to_delete = checkpoints[keep_count:]

        deleted = 0
        for checkpoint in to_delete:
            if keep_approved and checkpoint.approved:
                continue

            # Delete file
            pipeline_dir = self.storage / pipeline_id
            file_path = pipeline_dir / f"{checkpoint.checkpoint_id}.json"
            if file_path.exists():
                file_path.unlink()
                deleted += 1

        return deleted

    def get_checkpoint_summary(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for checkpoints of a pipeline.

        Returns:
            Dictionary with checkpoint statistics
        """
        checkpoints = self.list_checkpoints(pipeline_id=pipeline_id)

        return {
            "total": len(checkpoints),
            "auto": sum(1 for c in checkpoints if c.checkpoint_type == "auto"),
            "user_defined": sum(
                1 for c in checkpoints if c.checkpoint_type == "user_defined"
            ),
            "error": sum(1 for c in checkpoints if c.checkpoint_type == "error"),
            "approved": sum(1 for c in checkpoints if c.approved is True),
            "rejected": sum(1 for c in checkpoints if c.approved is False),
            "pending": sum(1 for c in checkpoints if c.approved is None),
            "latest": checkpoints[0].to_dict() if checkpoints else None,
        }
