"""
Iteration Loop for feedback-driven product improvements.
Collects feedback, generates improvement plans, and tracks iterations.
"""

import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from ..config import SYSTEM_DIR


# Iteration data storage
ITERATION_DATA_PATH = SYSTEM_DIR / "iteration_data.json"


class FeedbackType(Enum):
    """Types of feedback."""

    BUG = "bug"  # Bug report
    FEATURE = "feature"  # Feature request
    UX = "ux"  # UX improvement
    PERFORMANCE = "performance"  # Performance issue
    CONTENT = "content"  # Content/copy change
    GENERAL = "general"  # General feedback


class FeedbackPriority(Enum):
    """Feedback priority levels."""

    CRITICAL = "critical"  # Must fix immediately
    HIGH = "high"  # Important, fix soon
    MEDIUM = "medium"  # Should address
    LOW = "low"  # Nice to have


class IterationStatus(Enum):
    """Status of an iteration."""

    PLANNED = "planned"  # In the plan
    IN_PROGRESS = "in_progress"  # Being implemented
    COMPLETED = "completed"  # Done
    CANCELLED = "cancelled"  # Won't do


@dataclass
class FeedbackEntry:
    """A piece of feedback for a product."""

    id: str
    product_id: str
    feedback_type: str  # FeedbackType value
    priority: str  # FeedbackPriority value
    content: str
    source: str  # Where feedback came from
    created_at: str
    resolved: bool = False
    resolved_at: Optional[str] = None
    resolution_notes: Optional[str] = None


@dataclass
class ImprovementItem:
    """A single improvement in a plan."""

    id: str
    title: str
    description: str
    feedback_ids: List[str]  # Related feedback
    effort: str  # small, medium, large
    impact: str  # low, medium, high
    status: str = "planned"


@dataclass
class ImprovementPlan:
    """Plan for product improvements."""

    id: str
    product_id: str
    version: str
    created_at: str
    items: List[ImprovementItem] = field(default_factory=list)
    notes: str = ""


@dataclass
class IterationRecord:
    """Record of an iteration cycle."""

    id: str
    product_id: str
    plan_id: str
    started_at: str
    completed_at: Optional[str] = None
    items_completed: List[str] = field(default_factory=list)
    commit_sha: Optional[str] = None
    deployment_url: Optional[str] = None
    notes: str = ""


class IterationLoop:
    """
    Feedback-driven iteration for deployed products.

    Features:
    - Collect and categorize feedback
    - Generate improvement plans
    - Track iteration progress
    - Link iterations to deployments
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize iteration loop.

        Args:
            data_path: Path to data file
        """
        self.data_path = Path(data_path or ITERATION_DATA_PATH)
        self._feedback: Dict[str, FeedbackEntry] = {}
        self._plans: Dict[str, ImprovementPlan] = {}
        self._iterations: Dict[str, IterationRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load data from file."""
        if self.data_path.exists():
            try:
                data = json.loads(self.data_path.read_text())

                for fid, fdata in data.get("feedback", {}).items():
                    self._feedback[fid] = FeedbackEntry(**fdata)

                for pid, pdata in data.get("plans", {}).items():
                    pdata["items"] = [
                        ImprovementItem(**item)
                        for item in pdata.get("items", [])
                    ]
                    self._plans[pid] = ImprovementPlan(**pdata)

                for iid, idata in data.get("iterations", {}).items():
                    self._iterations[iid] = IterationRecord(**idata)

            except Exception:
                pass

    def _save(self) -> None:
        """Save data to file."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "feedback": {
                fid: asdict(f) for fid, f in self._feedback.items()
            },
            "plans": {
                pid: {
                    **asdict(p),
                    "items": [asdict(item) for item in p.items],
                }
                for pid, p in self._plans.items()
            },
            "iterations": {
                iid: asdict(i) for iid, i in self._iterations.items()
            },
        }

        self.data_path.write_text(json.dumps(data, indent=2))

    def _generate_id(self, prefix: str, content: str) -> str:
        """Generate unique ID."""
        timestamp = datetime.utcnow().isoformat()
        return f"{prefix}-{hashlib.sha256(f'{content}-{timestamp}'.encode()).hexdigest()[:8]}"

    # =========================================================================
    # Feedback Management
    # =========================================================================

    def record_feedback(
        self,
        product_id: str,
        content: str,
        feedback_type: FeedbackType = FeedbackType.GENERAL,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        source: str = "manual",
    ) -> FeedbackEntry:
        """
        Record feedback for a product.

        Args:
            product_id: Product ID
            content: Feedback content
            feedback_type: Type of feedback
            priority: Priority level
            source: Where feedback came from

        Returns:
            Created feedback entry
        """
        feedback_id = self._generate_id("fb", content)
        now = datetime.utcnow().isoformat()

        entry = FeedbackEntry(
            id=feedback_id,
            product_id=product_id,
            feedback_type=feedback_type.value,
            priority=priority.value,
            content=content,
            source=source,
            created_at=now,
        )

        self._feedback[feedback_id] = entry
        self._save()

        return entry

    def resolve_feedback(
        self,
        feedback_id: str,
        resolution_notes: str = "",
    ) -> Optional[FeedbackEntry]:
        """
        Mark feedback as resolved.

        Args:
            feedback_id: Feedback ID
            resolution_notes: Notes about resolution

        Returns:
            Updated feedback or None
        """
        if feedback_id not in self._feedback:
            return None

        entry = self._feedback[feedback_id]
        entry.resolved = True
        entry.resolved_at = datetime.utcnow().isoformat()
        entry.resolution_notes = resolution_notes

        self._save()
        return entry

    def get_feedback(
        self,
        product_id: str,
        resolved: Optional[bool] = None,
        feedback_type: Optional[FeedbackType] = None,
    ) -> List[FeedbackEntry]:
        """
        Get feedback for a product.

        Args:
            product_id: Product ID
            resolved: Filter by resolution status
            feedback_type: Filter by type

        Returns:
            List of feedback entries
        """
        feedback = [
            f for f in self._feedback.values()
            if f.product_id == product_id
        ]

        if resolved is not None:
            feedback = [f for f in feedback if f.resolved == resolved]

        if feedback_type:
            feedback = [
                f for f in feedback
                if f.feedback_type == feedback_type.value
            ]

        return sorted(feedback, key=lambda f: f.created_at, reverse=True)

    def get_unresolved_feedback(self, product_id: str) -> List[FeedbackEntry]:
        """Get all unresolved feedback for a product."""
        return self.get_feedback(product_id, resolved=False)

    # =========================================================================
    # Improvement Plans
    # =========================================================================

    def create_improvement_plan(
        self,
        product_id: str,
        version: str,
        notes: str = "",
    ) -> ImprovementPlan:
        """
        Create a new improvement plan.

        Args:
            product_id: Product ID
            version: Version number for this plan
            notes: Additional notes

        Returns:
            Created plan
        """
        plan_id = self._generate_id("plan", f"{product_id}-{version}")
        now = datetime.utcnow().isoformat()

        plan = ImprovementPlan(
            id=plan_id,
            product_id=product_id,
            version=version,
            created_at=now,
            notes=notes,
        )

        self._plans[plan_id] = plan
        self._save()

        return plan

    def add_improvement_item(
        self,
        plan_id: str,
        title: str,
        description: str,
        feedback_ids: Optional[List[str]] = None,
        effort: str = "medium",
        impact: str = "medium",
    ) -> Optional[ImprovementItem]:
        """
        Add an item to an improvement plan.

        Args:
            plan_id: Plan ID
            title: Item title
            description: Item description
            feedback_ids: Related feedback IDs
            effort: Estimated effort (small, medium, large)
            impact: Expected impact (low, medium, high)

        Returns:
            Created item or None
        """
        if plan_id not in self._plans:
            return None

        item_id = self._generate_id("item", title)

        item = ImprovementItem(
            id=item_id,
            title=title,
            description=description,
            feedback_ids=feedback_ids or [],
            effort=effort,
            impact=impact,
        )

        self._plans[plan_id].items.append(item)
        self._save()

        return item

    def generate_improvement_plan(
        self,
        product_id: str,
        version: str,
    ) -> ImprovementPlan:
        """
        Auto-generate an improvement plan from unresolved feedback.

        Args:
            product_id: Product ID
            version: Version for the plan

        Returns:
            Generated plan with items
        """
        # Create plan
        plan = self.create_improvement_plan(
            product_id=product_id,
            version=version,
            notes="Auto-generated from unresolved feedback",
        )

        # Get unresolved feedback
        feedback = self.get_unresolved_feedback(product_id)

        # Group by type
        by_type: Dict[str, List[FeedbackEntry]] = {}
        for f in feedback:
            if f.feedback_type not in by_type:
                by_type[f.feedback_type] = []
            by_type[f.feedback_type].append(f)

        # Create items for each group
        for ftype, items in by_type.items():
            # Determine priority/effort from feedback
            priorities = [f.priority for f in items]
            has_critical = "critical" in priorities
            has_high = "high" in priorities

            effort = "small" if len(items) <= 2 else "medium" if len(items) <= 5 else "large"
            impact = "high" if has_critical else "medium" if has_high else "low"

            # Create summary item
            titles = [f.content[:50] for f in items[:3]]
            description = f"Address {len(items)} {ftype} issue(s):\n" + "\n".join(
                f"- {t}" for t in titles
            )
            if len(items) > 3:
                description += f"\n- ...and {len(items) - 3} more"

            self.add_improvement_item(
                plan_id=plan.id,
                title=f"Address {ftype.title()} feedback",
                description=description,
                feedback_ids=[f.id for f in items],
                effort=effort,
                impact=impact,
            )

        return self._plans[plan.id]

    def get_plan(self, plan_id: str) -> Optional[ImprovementPlan]:
        """Get improvement plan by ID."""
        return self._plans.get(plan_id)

    def get_plans_for_product(self, product_id: str) -> List[ImprovementPlan]:
        """Get all plans for a product."""
        plans = [
            p for p in self._plans.values()
            if p.product_id == product_id
        ]
        return sorted(plans, key=lambda p: p.created_at, reverse=True)

    def update_item_status(
        self,
        plan_id: str,
        item_id: str,
        status: IterationStatus,
    ) -> bool:
        """
        Update the status of an improvement item.

        Args:
            plan_id: Plan ID
            item_id: Item ID
            status: New status

        Returns:
            True if updated
        """
        plan = self._plans.get(plan_id)
        if not plan:
            return False

        for item in plan.items:
            if item.id == item_id:
                item.status = status.value
                self._save()
                return True

        return False

    # =========================================================================
    # Iteration Records
    # =========================================================================

    def start_iteration(
        self,
        product_id: str,
        plan_id: str,
    ) -> IterationRecord:
        """
        Start an iteration cycle.

        Args:
            product_id: Product ID
            plan_id: Plan being implemented

        Returns:
            Iteration record
        """
        iteration_id = self._generate_id("iter", f"{product_id}-{plan_id}")
        now = datetime.utcnow().isoformat()

        record = IterationRecord(
            id=iteration_id,
            product_id=product_id,
            plan_id=plan_id,
            started_at=now,
        )

        self._iterations[iteration_id] = record
        self._save()

        return record

    def complete_iteration(
        self,
        iteration_id: str,
        items_completed: List[str],
        commit_sha: Optional[str] = None,
        deployment_url: Optional[str] = None,
        notes: str = "",
    ) -> Optional[IterationRecord]:
        """
        Complete an iteration.

        Args:
            iteration_id: Iteration ID
            items_completed: IDs of completed items
            commit_sha: Git commit SHA
            deployment_url: URL after deployment
            notes: Notes about the iteration

        Returns:
            Updated record or None
        """
        if iteration_id not in self._iterations:
            return None

        record = self._iterations[iteration_id]
        record.completed_at = datetime.utcnow().isoformat()
        record.items_completed = items_completed
        record.commit_sha = commit_sha
        record.deployment_url = deployment_url
        record.notes = notes

        # Update item statuses in the plan
        plan = self._plans.get(record.plan_id)
        if plan:
            for item in plan.items:
                if item.id in items_completed:
                    item.status = IterationStatus.COMPLETED.value

        # Resolve related feedback
        for item_id in items_completed:
            if plan:
                for item in plan.items:
                    if item.id == item_id:
                        for fb_id in item.feedback_ids:
                            self.resolve_feedback(
                                fb_id,
                                f"Resolved in iteration {iteration_id}",
                            )

        self._save()
        return record

    def get_iteration(self, iteration_id: str) -> Optional[IterationRecord]:
        """Get iteration by ID."""
        return self._iterations.get(iteration_id)

    def get_iteration_history(
        self,
        product_id: str,
        limit: int = 10,
    ) -> List[IterationRecord]:
        """
        Get iteration history for a product.

        Args:
            product_id: Product ID
            limit: Max records

        Returns:
            List of iterations (newest first)
        """
        iterations = [
            i for i in self._iterations.values()
            if i.product_id == product_id
        ]

        sorted_iterations = sorted(
            iterations,
            key=lambda i: i.started_at,
            reverse=True,
        )

        return sorted_iterations[:limit]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self, product_id: str) -> Dict[str, Any]:
        """
        Get iteration statistics for a product.

        Returns:
            Stats dict
        """
        feedback = [f for f in self._feedback.values() if f.product_id == product_id]
        plans = [p for p in self._plans.values() if p.product_id == product_id]
        iterations = [i for i in self._iterations.values() if i.product_id == product_id]

        unresolved = len([f for f in feedback if not f.resolved])
        resolved = len([f for f in feedback if f.resolved])

        by_type = {}
        for f in feedback:
            by_type[f.feedback_type] = by_type.get(f.feedback_type, 0) + 1

        return {
            "total_feedback": len(feedback),
            "unresolved_feedback": unresolved,
            "resolved_feedback": resolved,
            "feedback_by_type": by_type,
            "total_plans": len(plans),
            "total_iterations": len(iterations),
            "completed_iterations": len([i for i in iterations if i.completed_at]),
        }


# Singleton instance
_iteration_loop: Optional[IterationLoop] = None


def get_iteration_loop() -> IterationLoop:
    """Get or create IterationLoop singleton."""
    global _iteration_loop
    if _iteration_loop is None:
        _iteration_loop = IterationLoop()
    return _iteration_loop


def reset_iteration_loop() -> None:
    """Reset the singleton."""
    global _iteration_loop
    _iteration_loop = None
