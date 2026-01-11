"""
Browser automation for AI Orchestrator.

Provides Playwright-based browser automation with:
- Human-in-the-loop safety controls
- Dry-run mode by default
- Approval workflow for submit/destructive actions
- Rate limiting (3-5 second delays)
- Screenshot capture before/after actions
- Encrypted credential storage

Usage:
    from ai_orchestrator.browser import (
        get_playwright_client,
        BrowserAction,
        BrowserActionType,
        ActionCategory,
        create_action,
    )

    # Get client (singleton)
    client = get_playwright_client(headless=True, dry_run=True)

    # Start session
    session = client.start_session()

    # Create and execute action
    action = create_action(
        action_type="navigate",
        target_url="https://example.com",
        description="Go to example.com",
    )
    result = client.execute_action(action)

    # End session
    client.end_session()
"""

from .state import (
    # Enums
    BrowserActionType,
    ActionCategory,
    ApprovalStatus,
    # Dataclasses
    BrowserAction,
    ActionResult,
    BrowserSession,
    # TypedDict
    BrowserState,
    # Helpers
    ACTION_CATEGORY_MAP,
    get_action_category,
    create_action,
)

from .client import (
    PlaywrightClient,
    get_playwright_client,
    reset_playwright_client,
)

from .rate_limiter import (
    RateLimiter,
    AdaptiveRateLimiter,
)

from .screenshots import (
    ScreenshotManager,
)

from .credentials import (
    CredentialManager,
)

from .queue import (
    ActionQueue,
    ActionQueueEntry,
    list_queues,
    get_queue,
)


__all__ = [
    # State types
    "BrowserActionType",
    "ActionCategory",
    "ApprovalStatus",
    "BrowserAction",
    "ActionResult",
    "BrowserSession",
    "BrowserState",
    "ACTION_CATEGORY_MAP",
    "get_action_category",
    "create_action",
    # Client
    "PlaywrightClient",
    "get_playwright_client",
    "reset_playwright_client",
    # Rate limiting
    "RateLimiter",
    "AdaptiveRateLimiter",
    # Screenshots
    "ScreenshotManager",
    # Credentials
    "CredentialManager",
    # Queue
    "ActionQueue",
    "ActionQueueEntry",
    "list_queues",
    "get_queue",
]
