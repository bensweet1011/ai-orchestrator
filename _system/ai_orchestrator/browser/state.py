"""
Browser automation state management.
Defines types for actions, results, and sessions.
"""

from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class BrowserActionType(Enum):
    """Types of browser actions."""

    NAVIGATE = "navigate"  # Go to URL
    FILL = "fill"  # Fill form field
    CLICK = "click"  # Click element
    EXTRACT = "extract"  # Extract text/data
    SCREENSHOT = "screenshot"  # Take screenshot
    WAIT = "wait"  # Wait for element/condition
    SELECT = "select"  # Select from dropdown
    HOVER = "hover"  # Hover over element
    SCROLL = "scroll"  # Scroll page
    EXECUTE_JS = "execute_js"  # Run JavaScript


class ActionCategory(Enum):
    """Categories for safety classification."""

    READ_ONLY = "read_only"  # Navigate, extract, screenshot - auto-approved
    INPUT = "input"  # Fill, select - reversible
    SUBMIT = "submit"  # Click submit, post forms - requires approval
    DESTRUCTIVE = "destructive"  # Delete actions - always requires approval


class ApprovalStatus(Enum):
    """Status of action approval."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"  # For read-only in non-dry-run


# Map action types to their default categories
ACTION_CATEGORY_MAP: Dict[BrowserActionType, ActionCategory] = {
    BrowserActionType.NAVIGATE: ActionCategory.READ_ONLY,
    BrowserActionType.EXTRACT: ActionCategory.READ_ONLY,
    BrowserActionType.SCREENSHOT: ActionCategory.READ_ONLY,
    BrowserActionType.WAIT: ActionCategory.READ_ONLY,
    BrowserActionType.HOVER: ActionCategory.READ_ONLY,
    BrowserActionType.SCROLL: ActionCategory.READ_ONLY,
    BrowserActionType.FILL: ActionCategory.INPUT,
    BrowserActionType.SELECT: ActionCategory.INPUT,
    BrowserActionType.CLICK: ActionCategory.SUBMIT,  # Conservative default
    BrowserActionType.EXECUTE_JS: ActionCategory.SUBMIT,
}


@dataclass
class BrowserAction:
    """Defines a single browser action."""

    action_id: str
    action_type: BrowserActionType
    category: ActionCategory

    # Target specification
    target_selector: Optional[str] = None  # CSS/XPath selector
    target_url: Optional[str] = None
    value: Optional[str] = None  # For fill actions
    options: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # Safety and approval
    requires_approval: bool = False
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    rejection_reason: Optional[str] = None

    # Execution state
    executed: bool = False
    executed_at: Optional[str] = None
    result: Optional["ActionResult"] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "category": self.category.value,
            "target_selector": self.target_selector,
            "target_url": self.target_url,
            "value": self.value,
            "options": self.options,
            "description": self.description,
            "requires_approval": self.requires_approval,
            "approval_status": self.approval_status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "rejection_reason": self.rejection_reason,
            "executed": self.executed,
            "executed_at": self.executed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrowserAction":
        """Deserialize from dictionary."""
        return cls(
            action_id=data["action_id"],
            action_type=BrowserActionType(data["action_type"]),
            category=ActionCategory(data["category"]),
            target_selector=data.get("target_selector"),
            target_url=data.get("target_url"),
            value=data.get("value"),
            options=data.get("options", {}),
            description=data.get("description", ""),
            requires_approval=data.get("requires_approval", False),
            approval_status=ApprovalStatus(
                data.get("approval_status", "pending")
            ),
            approved_by=data.get("approved_by"),
            approved_at=data.get("approved_at"),
            rejection_reason=data.get("rejection_reason"),
            executed=data.get("executed", False),
            executed_at=data.get("executed_at"),
        )


@dataclass
class ActionResult:
    """Result from executing a browser action."""

    success: bool
    action_id: str
    action_type: BrowserActionType

    # Output data
    extracted_data: Optional[str] = None
    screenshot_path: Optional[str] = None
    page_url: Optional[str] = None
    page_title: Optional[str] = None

    # Timing
    latency_ms: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # Error handling
    error: Optional[str] = None
    error_screenshot_path: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "extracted_data": self.extracted_data,
            "screenshot_path": self.screenshot_path,
            "page_url": self.page_url,
            "page_title": self.page_title,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "error": self.error,
            "error_screenshot_path": self.error_screenshot_path,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionResult":
        """Deserialize from dictionary."""
        return cls(
            success=data["success"],
            action_id=data["action_id"],
            action_type=BrowserActionType(data["action_type"]),
            extracted_data=data.get("extracted_data"),
            screenshot_path=data.get("screenshot_path"),
            page_url=data.get("page_url"),
            page_title=data.get("page_title"),
            latency_ms=data.get("latency_ms", 0),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            error=data.get("error"),
            error_screenshot_path=data.get("error_screenshot_path"),
            retry_count=data.get("retry_count", 0),
        )


@dataclass
class BrowserSession:
    """Represents an active browser session."""

    session_id: str
    started_at: str
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True

    # State tracking
    current_url: Optional[str] = None
    page_title: Optional[str] = None
    context_name: Optional[str] = None

    # Action history
    action_history: List[ActionResult] = field(default_factory=list)
    screenshot_history: List[str] = field(default_factory=list)

    # Credentials context
    active_credentials: Optional[str] = None  # Site identifier

    # Session metadata
    ended_at: Optional[str] = None
    total_actions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "browser_type": self.browser_type,
            "headless": self.headless,
            "current_url": self.current_url,
            "page_title": self.page_title,
            "context_name": self.context_name,
            "action_history": [a.to_dict() for a in self.action_history],
            "screenshot_history": self.screenshot_history,
            "active_credentials": self.active_credentials,
            "ended_at": self.ended_at,
            "total_actions": self.total_actions,
        }


class BrowserState(TypedDict, total=False):
    """State for browser automation in pipelines."""

    # Core state (extends PipelineState pattern)
    session: Optional[Dict[str, Any]]
    current_action: Optional[Dict[str, Any]]
    action_queue: List[Dict[str, Any]]
    completed_actions: List[Dict[str, Any]]

    # Safety controls
    dry_run: bool
    pending_approvals: List[str]  # Action IDs

    # Output
    extracted_data: Dict[str, str]
    screenshots: List[str]

    # Rate limiting
    last_action_time: Optional[str]
    action_count: int


def get_action_category(action_type: BrowserActionType) -> ActionCategory:
    """Get the default category for an action type."""
    return ACTION_CATEGORY_MAP.get(action_type, ActionCategory.SUBMIT)


def create_action(
    action_type: str,
    target_selector: Optional[str] = None,
    target_url: Optional[str] = None,
    value: Optional[str] = None,
    description: str = "",
    options: Optional[Dict[str, Any]] = None,
    action_id: Optional[str] = None,
) -> BrowserAction:
    """
    Helper to create a BrowserAction with sensible defaults.

    Args:
        action_type: Action type string (navigate, fill, click, etc.)
        target_selector: CSS/XPath selector for element
        target_url: URL for navigate actions
        value: Value for fill/select actions
        description: Human-readable description
        options: Additional options dict
        action_id: Optional custom action ID

    Returns:
        BrowserAction instance
    """
    import hashlib

    # Generate action ID if not provided
    if not action_id:
        unique = f"{action_type}{target_url}{target_selector}{datetime.utcnow().isoformat()}"
        action_id = hashlib.md5(unique.encode()).hexdigest()[:8]

    # Parse action type
    action_type_enum = BrowserActionType(action_type)
    category = get_action_category(action_type_enum)

    # Determine if approval required
    requires_approval = category in [ActionCategory.SUBMIT, ActionCategory.DESTRUCTIVE]

    return BrowserAction(
        action_id=action_id,
        action_type=action_type_enum,
        category=category,
        target_selector=target_selector,
        target_url=target_url,
        value=value,
        description=description or f"{action_type} action",
        options=options or {},
        requires_approval=requires_approval,
        approval_status=(
            ApprovalStatus.AUTO_APPROVED
            if category == ActionCategory.READ_ONLY
            else ApprovalStatus.PENDING
        ),
    )
