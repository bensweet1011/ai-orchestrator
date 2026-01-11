"""
Playwright client integration for browser automation.
Provides a singleton client with safety controls and session management.
"""

import hashlib
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

from .state import (
    BrowserAction,
    BrowserActionType,
    ActionCategory,
    ActionResult,
    BrowserSession,
    ApprovalStatus,
)
from .rate_limiter import RateLimiter
from .screenshots import ScreenshotManager
from .credentials import CredentialManager


class PlaywrightClient:
    """
    Playwright-based browser automation client.

    Features:
    - Headless and headed modes
    - Page/context management
    - Rate limiting with human-like delays (3-5 seconds)
    - Screenshot capture before/after actions
    - Credential management
    - Dry-run mode by default

    Safety defaults:
    - dry_run=True: Simulates submit/destructive actions
    - Approval required for SUBMIT/DESTRUCTIVE categories
    - Rate limiting enforced: 3-5 second delays
    """

    def __init__(
        self,
        storage_path: Path,
        headless: bool = True,
        browser_type: str = "chromium",
        rate_limit_min: float = 3.0,
        rate_limit_max: float = 5.0,
        dry_run: bool = True,
    ):
        """
        Initialize Playwright client.

        Args:
            storage_path: Base path for screenshots and credentials
            headless: Run in headless mode (default True)
            browser_type: Browser to use (chromium, firefox, webkit)
            rate_limit_min: Minimum delay between actions (seconds)
            rate_limit_max: Maximum delay between actions (seconds)
            dry_run: If True, only simulate submit/destructive actions
        """
        self.storage_path = Path(storage_path)
        self.headless = headless
        self.browser_type = browser_type
        self.dry_run = dry_run

        # Ensure directories exist
        screenshots_dir = self.storage_path / "screenshots"
        credentials_dir = self.storage_path / "credentials"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        credentials_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.rate_limiter = RateLimiter(
            min_delay=rate_limit_min, max_delay=rate_limit_max
        )
        self.screenshot_mgr = ScreenshotManager(screenshots_dir)
        self.credential_mgr = CredentialManager(credentials_dir)

        # Playwright state
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._session: Optional[BrowserSession] = None

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        ts = datetime.utcnow().isoformat()
        unique = hashlib.md5(ts.encode()).hexdigest()[:8]
        return f"browser_{unique}"

    @property
    def is_active(self) -> bool:
        """Check if a browser session is active."""
        return self._page is not None

    @property
    def session(self) -> Optional[BrowserSession]:
        """Get current session."""
        return self._session

    def start_session(
        self,
        site_credentials: Optional[str] = None,
        context_options: Optional[Dict] = None,
        viewport: Optional[Dict[str, int]] = None,
    ) -> BrowserSession:
        """
        Start a new browser session.

        Args:
            site_credentials: Identifier for stored credentials to use
            context_options: Additional browser context options
            viewport: Viewport size (default: 1280x720)

        Returns:
            BrowserSession object
        """
        # Import playwright here to allow deferred installation
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()

        # Get browser launcher
        browser_launcher = getattr(self._playwright, self.browser_type)
        self._browser = browser_launcher.launch(headless=self.headless)

        # Build context options
        ctx_options = context_options or {}

        # Set viewport
        if viewport:
            ctx_options["viewport"] = viewport
        elif "viewport" not in ctx_options:
            ctx_options["viewport"] = {"width": 1280, "height": 720}

        # Load stored credentials if specified
        if site_credentials:
            storage_state = self.credential_mgr.get_storage_state(site_credentials)
            if storage_state:
                ctx_options["storage_state"] = storage_state

        self._context = self._browser.new_context(**ctx_options)
        self._page = self._context.new_page()

        # Create session object
        self._session = BrowserSession(
            session_id=self._generate_session_id(),
            started_at=datetime.utcnow().isoformat(),
            browser_type=self.browser_type,
            headless=self.headless,
            active_credentials=site_credentials,
        )

        return self._session

    def end_session(
        self,
        save_credentials: Optional[str] = None,
    ):
        """
        End browser session.

        Args:
            save_credentials: If provided, save session state for this site identifier
        """
        import sys

        # Save storage state if requested
        if save_credentials and self._context:
            try:
                storage_state = self._context.storage_state()
                self.credential_mgr.save_storage_state(save_credentials, storage_state)
            except Exception as e:
                print(f"Warning: Failed to save credentials on shutdown: {e}", file=sys.stderr)

        # Mark session as ended
        if self._session:
            self._session.ended_at = datetime.utcnow().isoformat()

        # Cleanup with error logging
        if self._page:
            try:
                self._page.close()
            except Exception as e:
                print(f"Warning: Failed to close page: {e}", file=sys.stderr)
        if self._context:
            try:
                self._context.close()
            except Exception as e:
                print(f"Warning: Failed to close context: {e}", file=sys.stderr)
        if self._browser:
            try:
                self._browser.close()
            except Exception as e:
                print(f"Warning: Failed to close browser: {e}", file=sys.stderr)
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception as e:
                print(f"Warning: Failed to stop playwright: {e}", file=sys.stderr)

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    def execute_action(
        self,
        action: BrowserAction,
        take_screenshots: bool = True,
        skip_rate_limit: bool = False,
    ) -> ActionResult:
        """
        Execute a browser action with safety checks.

        Args:
            action: Action to execute
            take_screenshots: Whether to capture before/after screenshots
            skip_rate_limit: Skip rate limiting (use with caution)

        Returns:
            ActionResult with success/failure and data
        """
        if not self._page:
            return ActionResult(
                success=False,
                action_id=action.action_id,
                action_type=action.action_type,
                error="No active browser session. Call start_session() first.",
            )

        # Safety check: require approval for submit/destructive
        if action.category in [ActionCategory.SUBMIT, ActionCategory.DESTRUCTIVE]:
            if action.approval_status not in [
                ApprovalStatus.APPROVED,
                ApprovalStatus.AUTO_APPROVED,
            ]:
                return ActionResult(
                    success=False,
                    action_id=action.action_id,
                    action_type=action.action_type,
                    error=f"Action requires approval (category: {action.category.value}). "
                    f"Current status: {action.approval_status.value}",
                )

        # Dry-run check for submit/destructive
        if self.dry_run and action.category in [
            ActionCategory.SUBMIT,
            ActionCategory.DESTRUCTIVE,
        ]:
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                extracted_data=f"[DRY RUN] Would execute: {action.description or action.action_type.value}",
                page_url=self._page.url,
                page_title=self._page.title(),
            )

        # Rate limiting
        if not skip_rate_limit:
            self.rate_limiter.wait()

        # Take before screenshot
        before_screenshot = None
        if take_screenshots:
            try:
                before_screenshot = self.screenshot_mgr.capture(
                    self._page,
                    prefix=f"{action.action_id}_before",
                    action_id=action.action_id,
                )
            except Exception:
                pass  # Continue even if screenshot fails

        start_time = datetime.utcnow()

        try:
            result = self._execute_action_impl(action)

            # Take after screenshot
            if take_screenshots and result.success:
                try:
                    result.screenshot_path = self.screenshot_mgr.capture(
                        self._page,
                        prefix=f"{action.action_id}_after",
                        action_id=action.action_id,
                    )
                except Exception:
                    pass

            # Calculate latency
            result.latency_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

            # Update session state
            if self._session:
                self._session.current_url = self._page.url
                self._session.page_title = self._page.title()
                self._session.action_history.append(result)
                self._session.total_actions += 1
                if result.screenshot_path:
                    self._session.screenshot_history.append(result.screenshot_path)

            # Mark action as executed
            action.executed = True
            action.executed_at = datetime.utcnow().isoformat()
            action.result = result

            return result

        except Exception as e:
            # Capture error screenshot
            error_screenshot = None
            if take_screenshots:
                try:
                    error_screenshot = self.screenshot_mgr.capture_error(
                        self._page,
                        action.action_id,
                        str(e),
                    )
                except Exception:
                    pass

            return ActionResult(
                success=False,
                action_id=action.action_id,
                action_type=action.action_type,
                error=str(e),
                error_screenshot_path=error_screenshot,
                latency_ms=int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
                page_url=self._page.url if self._page else None,
            )

    def _execute_action_impl(self, action: BrowserAction) -> ActionResult:
        """Execute action implementation."""
        page = self._page

        if action.action_type == BrowserActionType.NAVIGATE:
            timeout = action.options.get("timeout", 30000)
            wait_until = action.options.get("wait_until", "load")
            page.goto(action.target_url, timeout=timeout, wait_until=wait_until)
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                page_url=page.url,
                page_title=page.title(),
            )

        elif action.action_type == BrowserActionType.FILL:
            timeout = action.options.get("timeout", 30000)
            page.fill(action.target_selector, action.value or "", timeout=timeout)
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                page_url=page.url,
            )

        elif action.action_type == BrowserActionType.CLICK:
            timeout = action.options.get("timeout", 30000)
            page.click(action.target_selector, timeout=timeout)
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                page_url=page.url,
            )

        elif action.action_type == BrowserActionType.EXTRACT:
            timeout = action.options.get("timeout", 30000)
            element = page.wait_for_selector(action.target_selector, timeout=timeout)
            extracted = element.text_content() if element else None
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                extracted_data=extracted,
                page_url=page.url,
            )

        elif action.action_type == BrowserActionType.SCREENSHOT:
            path = self.screenshot_mgr.capture(
                page,
                prefix=action.action_id,
                full_page=action.options.get("full_page", False),
                action_id=action.action_id,
            )
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                screenshot_path=path,
                page_url=page.url,
            )

        elif action.action_type == BrowserActionType.WAIT:
            timeout = action.options.get("timeout", 30000)
            state = action.options.get("state", "visible")
            page.wait_for_selector(
                action.target_selector,
                timeout=timeout,
                state=state,
            )
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                page_url=page.url,
            )

        elif action.action_type == BrowserActionType.SELECT:
            timeout = action.options.get("timeout", 30000)
            page.select_option(
                action.target_selector,
                value=action.value,
                timeout=timeout,
            )
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                page_url=page.url,
            )

        elif action.action_type == BrowserActionType.HOVER:
            timeout = action.options.get("timeout", 30000)
            page.hover(action.target_selector, timeout=timeout)
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                page_url=page.url,
            )

        elif action.action_type == BrowserActionType.SCROLL:
            # Scroll to element or by amount
            if action.target_selector:
                element = page.query_selector(action.target_selector)
                if element:
                    element.scroll_into_view_if_needed()
            else:
                # Scroll by pixels
                x = action.options.get("x", 0)
                y = action.options.get("y", 300)
                page.evaluate(f"window.scrollBy({x}, {y})")

            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                page_url=page.url,
            )

        elif action.action_type == BrowserActionType.EXECUTE_JS:
            result = page.evaluate(action.value or "")
            return ActionResult(
                success=True,
                action_id=action.action_id,
                action_type=action.action_type,
                extracted_data=str(result) if result else None,
                page_url=page.url,
            )

        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")

    # -------------------------------------------------------------------------
    # Page Content Access
    # -------------------------------------------------------------------------

    def get_page_content(self) -> str:
        """Get current page text content."""
        if not self._page:
            return ""
        try:
            return self._page.text_content("body") or ""
        except Exception:
            return ""

    def get_page_html(self) -> str:
        """Get current page HTML."""
        if not self._page:
            return ""
        try:
            return self._page.content()
        except Exception:
            return ""

    def get_page_url(self) -> str:
        """Get current page URL."""
        if not self._page:
            return ""
        try:
            return self._page.url
        except Exception:
            return ""

    def get_page_title(self) -> str:
        """Get current page title."""
        if not self._page:
            return ""
        try:
            return self._page.title()
        except Exception:
            return ""


# -----------------------------------------------------------------------------
# Singleton Management
# -----------------------------------------------------------------------------

_playwright_client: Optional[PlaywrightClient] = None


def get_playwright_client(
    storage_path: Optional[Path] = None,
    headless: bool = True,
    dry_run: bool = True,
    reset: bool = False,
    **kwargs,
) -> PlaywrightClient:
    """
    Get or create PlaywrightClient singleton.

    Args:
        storage_path: Base path for browser data (defaults to TEMPLATES_DIR/browser)
        headless: Run in headless mode
        dry_run: Enable dry-run mode for safety
        reset: Force create new client instance
        **kwargs: Additional client options

    Returns:
        PlaywrightClient instance
    """
    global _playwright_client

    if _playwright_client is None or reset:
        if storage_path is None:
            # Import here to avoid circular dependency
            from ..config import TEMPLATES_DIR

            storage_path = TEMPLATES_DIR / "browser"

        _playwright_client = PlaywrightClient(
            storage_path=storage_path,
            headless=headless,
            dry_run=dry_run,
            **kwargs,
        )

    return _playwright_client


def reset_playwright_client():
    """Reset the singleton client (ends any active session)."""
    global _playwright_client

    if _playwright_client is not None:
        try:
            _playwright_client.end_session()
        except Exception:
            pass
        _playwright_client = None
