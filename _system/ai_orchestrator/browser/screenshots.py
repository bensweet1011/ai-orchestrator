"""
Screenshot capture and management for browser automation.
Handles before/after screenshots, error captures, and storage cleanup.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import json


class ScreenshotManager:
    """
    Manages screenshot capture and storage.

    Features:
    - Before/after action screenshots
    - Error state capture
    - Storage path management
    - Automatic cleanup of old screenshots
    - Metadata tracking
    """

    def __init__(
        self,
        storage_path: Path,
        max_screenshots: int = 1000,
        max_age_days: int = 7,
    ):
        """
        Initialize screenshot manager.

        Args:
            storage_path: Directory for screenshot storage
            max_screenshots: Maximum screenshots to retain
            max_age_days: Maximum age of screenshots before cleanup
        """
        self.storage_path = Path(storage_path)
        self.max_screenshots = max_screenshots
        self.max_age_days = max_age_days
        self._metadata_file = self.storage_path / "metadata.json"

        self.storage_path.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        page,  # Playwright Page object
        prefix: str = "screenshot",
        full_page: bool = False,
        action_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Capture screenshot from page.

        Args:
            page: Playwright page object
            prefix: Filename prefix
            full_page: Whether to capture full scrollable page
            action_id: Associated action ID for organization
            metadata: Additional metadata to store

        Returns:
            Path to saved screenshot
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.png"
        path = self.storage_path / filename

        # Capture screenshot
        page.screenshot(path=str(path), full_page=full_page)

        # Store metadata
        self._save_metadata(
            filename,
            {
                "action_id": action_id,
                "prefix": prefix,
                "full_page": full_page,
                "timestamp": datetime.utcnow().isoformat(),
                "page_url": page.url if hasattr(page, "url") else None,
                "page_title": page.title() if hasattr(page, "title") else None,
                **(metadata or {}),
            },
        )

        # Cleanup old screenshots if needed
        self._cleanup_if_needed()

        return str(path)

    def capture_before_after(
        self,
        page,
        action_id: str,
        capture_before: bool = True,
    ) -> Dict[str, Optional[str]]:
        """
        Capture before screenshot for an action.

        Call capture_after() after action execution.

        Args:
            page: Playwright page object
            action_id: Action identifier
            capture_before: Whether to capture before screenshot

        Returns:
            Dict with 'before' path (after path added later)
        """
        result = {"before": None, "after": None}

        if capture_before:
            result["before"] = self.capture(
                page,
                prefix=f"{action_id}_before",
                action_id=action_id,
                metadata={"phase": "before"},
            )

        return result

    def capture_after(
        self,
        page,
        action_id: str,
        success: bool = True,
    ) -> str:
        """
        Capture after screenshot for an action.

        Args:
            page: Playwright page object
            action_id: Action identifier
            success: Whether action succeeded

        Returns:
            Path to after screenshot
        """
        prefix = f"{action_id}_after" if success else f"{action_id}_error"
        return self.capture(
            page,
            prefix=prefix,
            action_id=action_id,
            metadata={"phase": "after", "success": success},
        )

    def capture_error(
        self,
        page,
        action_id: str,
        error: str,
    ) -> str:
        """
        Capture error state screenshot.

        Args:
            page: Playwright page object
            action_id: Action identifier
            error: Error message

        Returns:
            Path to error screenshot
        """
        return self.capture(
            page,
            prefix=f"{action_id}_error",
            action_id=action_id,
            full_page=True,
            metadata={"phase": "error", "error": error},
        )

    def _save_metadata(self, filename: str, metadata: Dict[str, Any]):
        """Save metadata for a screenshot."""
        all_metadata = self._load_all_metadata()
        all_metadata[filename] = metadata
        self._metadata_file.write_text(json.dumps(all_metadata, indent=2))

    def _load_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load all screenshot metadata."""
        if self._metadata_file.exists():
            try:
                return json.loads(self._metadata_file.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _cleanup_if_needed(self):
        """Remove old screenshots if over limits."""
        screenshots = sorted(
            self.storage_path.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
        )

        # Remove by count
        if len(screenshots) > self.max_screenshots:
            for old in screenshots[: -self.max_screenshots]:
                self._remove_screenshot(old)

        # Remove by age
        now = datetime.utcnow()
        for screenshot in screenshots:
            mtime = datetime.fromtimestamp(screenshot.stat().st_mtime)
            age_days = (now - mtime).days
            if age_days > self.max_age_days:
                self._remove_screenshot(screenshot)

    def _remove_screenshot(self, path: Path):
        """Remove screenshot and its metadata."""
        try:
            path.unlink()
            # Remove metadata entry
            all_metadata = self._load_all_metadata()
            if path.name in all_metadata:
                del all_metadata[path.name]
                self._metadata_file.write_text(json.dumps(all_metadata, indent=2))
        except Exception:
            pass  # Ignore removal errors

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent screenshots with metadata.

        Args:
            limit: Maximum number to return

        Returns:
            List of dicts with path and metadata
        """
        screenshots = sorted(
            self.storage_path.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        all_metadata = self._load_all_metadata()
        result = []

        for path in screenshots[:limit]:
            result.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "metadata": all_metadata.get(path.name, {}),
                }
            )

        return result

    def get_by_action(self, action_id: str) -> List[Dict[str, Any]]:
        """
        Get all screenshots for an action.

        Args:
            action_id: Action identifier

        Returns:
            List of dicts with path and metadata
        """
        all_metadata = self._load_all_metadata()
        result = []

        for filename, metadata in all_metadata.items():
            if metadata.get("action_id") == action_id:
                path = self.storage_path / filename
                if path.exists():
                    result.append(
                        {
                            "path": str(path),
                            "filename": filename,
                            "metadata": metadata,
                        }
                    )

        # Sort by timestamp
        result.sort(key=lambda x: x["metadata"].get("timestamp", ""))
        return result

    def get_pairs(self, action_id: str) -> Dict[str, Optional[str]]:
        """
        Get before/after screenshot pair for an action.

        Args:
            action_id: Action identifier

        Returns:
            Dict with 'before', 'after', and 'error' paths
        """
        screenshots = self.get_by_action(action_id)
        result = {"before": None, "after": None, "error": None}

        for item in screenshots:
            phase = item["metadata"].get("phase")
            if phase == "before":
                result["before"] = item["path"]
            elif phase == "after":
                result["after"] = item["path"]
            elif phase == "error":
                result["error"] = item["path"]

        return result

    def clear_all(self):
        """Remove all screenshots and metadata."""
        for screenshot in self.storage_path.glob("*.png"):
            screenshot.unlink()
        if self._metadata_file.exists():
            self._metadata_file.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        screenshots = list(self.storage_path.glob("*.png"))
        total_size = sum(s.stat().st_size for s in screenshots)

        return {
            "total_screenshots": len(screenshots),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_screenshots": self.max_screenshots,
            "storage_path": str(self.storage_path),
        }
