"""
Semi-automation queue for action approval workflow.
Manages pending actions that require human approval before execution.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from .state import (
    BrowserAction,
    ActionCategory,
    ApprovalStatus,
    BrowserActionType,
)


@dataclass
class ActionQueueEntry:
    """Entry in the action queue."""

    action: BrowserAction
    created_at: str
    context: Dict[str, Any] = field(default_factory=dict)
    preview_screenshot: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action": self.action.to_dict(),
            "created_at": self.created_at,
            "context": self.context,
            "preview_screenshot": self.preview_screenshot,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionQueueEntry":
        """Deserialize from dictionary."""
        return cls(
            action=BrowserAction.from_dict(data["action"]),
            created_at=data["created_at"],
            context=data.get("context", {}),
            preview_screenshot=data.get("preview_screenshot"),
            notes=data.get("notes", ""),
        )


class ActionQueue:
    """
    Manages queue of browser actions pending approval.

    Features:
    - Persistent queue storage (JSON)
    - Approval/rejection workflow
    - Auto-approval for read-only actions
    - Batch approval support
    - History tracking
    """

    def __init__(
        self,
        storage_path: Path,
        queue_id: Optional[str] = None,
        auto_approve_read_only: bool = True,
    ):
        """
        Initialize action queue.

        Args:
            storage_path: Directory for queue storage
            queue_id: Unique identifier for this queue instance
            auto_approve_read_only: Automatically approve read-only actions
        """
        self.storage_path = Path(storage_path)
        self.queue_id = queue_id or f"queue_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.auto_approve_read_only = auto_approve_read_only
        self.queue_file = self.storage_path / f"{self.queue_id}.json"

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._pending: List[ActionQueueEntry] = []
        self._approved: List[ActionQueueEntry] = []
        self._rejected: List[ActionQueueEntry] = []
        self._executed: List[ActionQueueEntry] = []

        # Load existing queue if present
        self._load()

    def _load(self):
        """Load queue from disk."""
        if self.queue_file.exists():
            try:
                data = json.loads(self.queue_file.read_text())
                self._pending = [
                    ActionQueueEntry.from_dict(e) for e in data.get("pending", [])
                ]
                self._approved = [
                    ActionQueueEntry.from_dict(e) for e in data.get("approved", [])
                ]
                self._rejected = [
                    ActionQueueEntry.from_dict(e) for e in data.get("rejected", [])
                ]
                self._executed = [
                    ActionQueueEntry.from_dict(e) for e in data.get("executed", [])
                ]
            except (json.JSONDecodeError, KeyError):
                pass  # Start fresh on parse errors

    def _save(self):
        """Save queue to disk."""
        data = {
            "queue_id": self.queue_id,
            "pending": [e.to_dict() for e in self._pending],
            "approved": [e.to_dict() for e in self._approved],
            "rejected": [e.to_dict() for e in self._rejected],
            "executed": [e.to_dict() for e in self._executed],
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.queue_file.write_text(json.dumps(data, indent=2))

    # -------------------------------------------------------------------------
    # Adding Actions
    # -------------------------------------------------------------------------

    def add_action(
        self,
        action: BrowserAction,
        context: Optional[Dict] = None,
        preview_screenshot: Optional[str] = None,
        notes: str = "",
    ) -> ActionQueueEntry:
        """
        Add action to queue.

        Auto-approves read-only actions if configured.

        Args:
            action: BrowserAction to queue
            context: Additional context (URL, page state, etc.)
            preview_screenshot: Path to preview screenshot
            notes: Human-readable notes about the action

        Returns:
            Created ActionQueueEntry
        """
        entry = ActionQueueEntry(
            action=action,
            created_at=datetime.utcnow().isoformat(),
            context=context or {},
            preview_screenshot=preview_screenshot,
            notes=notes,
        )

        # Auto-approve read-only actions
        if (
            self.auto_approve_read_only
            and action.category == ActionCategory.READ_ONLY
        ):
            action.approval_status = ApprovalStatus.AUTO_APPROVED
            action.approved_at = datetime.utcnow().isoformat()
            action.approved_by = "auto"
            self._approved.append(entry)
        else:
            action.requires_approval = True
            self._pending.append(entry)

        self._save()
        return entry

    def add_actions(
        self,
        actions: List[BrowserAction],
        context: Optional[Dict] = None,
    ) -> List[ActionQueueEntry]:
        """Add multiple actions to queue."""
        entries = []
        for action in actions:
            entry = self.add_action(action, context=context)
            entries.append(entry)
        return entries

    # -------------------------------------------------------------------------
    # Approval Workflow
    # -------------------------------------------------------------------------

    def approve_action(
        self,
        action_id: str,
        approver: str = "user",
        notes: str = "",
    ) -> bool:
        """
        Approve a pending action.

        Args:
            action_id: Action to approve
            approver: Who approved (default: "user")
            notes: Optional approval notes

        Returns:
            True if approved, False if not found
        """
        for i, entry in enumerate(self._pending):
            if entry.action.action_id == action_id:
                entry.action.approval_status = ApprovalStatus.APPROVED
                entry.action.approved_by = approver
                entry.action.approved_at = datetime.utcnow().isoformat()
                if notes:
                    entry.notes = notes
                self._approved.append(entry)
                self._pending.pop(i)
                self._save()
                return True
        return False

    def reject_action(
        self,
        action_id: str,
        reason: str = "",
        rejector: str = "user",
    ) -> bool:
        """
        Reject a pending action.

        Args:
            action_id: Action to reject
            reason: Rejection reason
            rejector: Who rejected

        Returns:
            True if rejected, False if not found
        """
        for i, entry in enumerate(self._pending):
            if entry.action.action_id == action_id:
                entry.action.approval_status = ApprovalStatus.REJECTED
                entry.action.rejection_reason = reason
                entry.context["rejected_by"] = rejector
                entry.context["rejected_at"] = datetime.utcnow().isoformat()
                self._rejected.append(entry)
                self._pending.pop(i)
                self._save()
                return True
        return False

    def approve_all_pending(self, approver: str = "batch") -> int:
        """
        Approve all pending actions.

        Args:
            approver: Who approved

        Returns:
            Number of actions approved
        """
        count = len(self._pending)
        for entry in self._pending:
            entry.action.approval_status = ApprovalStatus.APPROVED
            entry.action.approved_by = approver
            entry.action.approved_at = datetime.utcnow().isoformat()
            self._approved.append(entry)
        self._pending.clear()
        self._save()
        return count

    def approve_by_category(
        self,
        category: ActionCategory,
        approver: str = "batch",
    ) -> int:
        """
        Approve all pending actions of a specific category.

        Args:
            category: Category to approve
            approver: Who approved

        Returns:
            Number of actions approved
        """
        to_approve = []
        to_keep = []

        for entry in self._pending:
            if entry.action.category == category:
                entry.action.approval_status = ApprovalStatus.APPROVED
                entry.action.approved_by = approver
                entry.action.approved_at = datetime.utcnow().isoformat()
                to_approve.append(entry)
            else:
                to_keep.append(entry)

        self._approved.extend(to_approve)
        self._pending = to_keep
        self._save()
        return len(to_approve)

    # -------------------------------------------------------------------------
    # Execution Tracking
    # -------------------------------------------------------------------------

    def mark_executed(self, action_id: str) -> bool:
        """
        Mark action as executed.

        Args:
            action_id: Action that was executed

        Returns:
            True if marked, False if not found
        """
        for i, entry in enumerate(self._approved):
            if entry.action.action_id == action_id:
                entry.action.executed = True
                entry.action.executed_at = datetime.utcnow().isoformat()
                self._executed.append(entry)
                self._approved.pop(i)
                self._save()
                return True
        return False

    def get_next_approved(self) -> Optional[BrowserAction]:
        """
        Get next approved action to execute.

        Returns:
            Next BrowserAction or None if queue empty
        """
        if self._approved:
            return self._approved[0].action
        return None

    # -------------------------------------------------------------------------
    # Queue Access
    # -------------------------------------------------------------------------

    def get_pending(self) -> List[ActionQueueEntry]:
        """Get all pending actions."""
        return self._pending.copy()

    def get_approved(self) -> List[ActionQueueEntry]:
        """Get all approved actions not yet executed."""
        return self._approved.copy()

    def get_rejected(self) -> List[ActionQueueEntry]:
        """Get all rejected actions."""
        return self._rejected.copy()

    def get_executed(self) -> List[ActionQueueEntry]:
        """Get all executed actions."""
        return self._executed.copy()

    def get_action(self, action_id: str) -> Optional[ActionQueueEntry]:
        """Get specific action by ID."""
        for queue in [self._pending, self._approved, self._rejected, self._executed]:
            for entry in queue:
                if entry.action.action_id == action_id:
                    return entry
        return None

    # -------------------------------------------------------------------------
    # Stats and Management
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_id": self.queue_id,
            "pending": len(self._pending),
            "approved": len(self._approved),
            "rejected": len(self._rejected),
            "executed": len(self._executed),
            "total": (
                len(self._pending)
                + len(self._approved)
                + len(self._rejected)
                + len(self._executed)
            ),
        }

    def clear_pending(self):
        """Clear all pending actions."""
        self._pending.clear()
        self._save()

    def clear_all(self):
        """Clear all queues."""
        self._pending.clear()
        self._approved.clear()
        self._rejected.clear()
        self._executed.clear()
        self._save()

    def delete(self):
        """Delete queue file."""
        if self.queue_file.exists():
            self.queue_file.unlink()


# -----------------------------------------------------------------------------
# Queue Management Functions
# -----------------------------------------------------------------------------


def list_queues(storage_path: Path) -> List[Dict[str, Any]]:
    """
    List all saved queues.

    Args:
        storage_path: Directory containing queue files

    Returns:
        List of queue info dicts
    """
    storage_path = Path(storage_path)
    queues = []

    for queue_file in storage_path.glob("queue_*.json"):
        try:
            data = json.loads(queue_file.read_text())
            queues.append(
                {
                    "queue_id": data.get("queue_id"),
                    "updated_at": data.get("updated_at"),
                    "pending": len(data.get("pending", [])),
                    "approved": len(data.get("approved", [])),
                    "executed": len(data.get("executed", [])),
                }
            )
        except Exception:
            continue

    return sorted(queues, key=lambda x: x.get("updated_at", ""), reverse=True)


def get_queue(storage_path: Path, queue_id: str) -> Optional[ActionQueue]:
    """
    Load an existing queue.

    Args:
        storage_path: Directory containing queue files
        queue_id: Queue identifier

    Returns:
        ActionQueue or None if not found
    """
    queue = ActionQueue(storage_path, queue_id=queue_id)
    if queue.queue_file.exists():
        return queue
    return None
