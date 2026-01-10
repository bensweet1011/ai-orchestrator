"""
Budget management for Cost Intelligence.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import PROJECTS_DIR
from .state import AlertLevel, BudgetAlert, BudgetConfig, BudgetPeriod, BudgetStatus
from .tracker import get_cost_tracker


class BudgetManager:
    """Manages project budgets and alerts."""

    def __init__(self, project: str = "default"):
        """
        Initialize budget manager.

        Args:
            project: Project name
        """
        self.project = project
        self._config: Optional[BudgetConfig] = None
        self._alerts: List[BudgetAlert] = []
        self._triggered_thresholds: Dict[str, set] = {
            "daily": set(),
            "weekly": set(),
            "monthly": set(),
        }
        self._load_config()
        self._load_alerts()

    def _get_config_path(self) -> Path:
        """Get path to budget config file."""
        return PROJECTS_DIR / self.project / "budget_config.json"

    def _get_alerts_path(self) -> Path:
        """Get path to budget alerts file."""
        return PROJECTS_DIR / self.project / "budget_alerts.json"

    def _load_config(self):
        """Load budget configuration from file."""
        config_path = self._get_config_path()
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                self._config = BudgetConfig.from_dict(data)
            except Exception as e:
                print(f"Warning: Failed to load budget config: {e}")
                self._config = None
        else:
            self._config = None

    def _save_config(self):
        """Save budget configuration to file."""
        if not self._config:
            return

        config_path = self._get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w") as f:
                json.dump(self._config.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save budget config: {e}")

    def _load_alerts(self):
        """Load alerts from file."""
        alerts_path = self._get_alerts_path()
        if alerts_path.exists():
            try:
                with open(alerts_path, "r") as f:
                    data = json.load(f)
                self._alerts = [BudgetAlert.from_dict(a) for a in data]
            except Exception:
                self._alerts = []
        else:
            self._alerts = []

    def _save_alerts(self):
        """Save alerts to file."""
        alerts_path = self._get_alerts_path()
        alerts_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(alerts_path, "w") as f:
                json.dump([a.to_dict() for a in self._alerts], f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save alerts: {e}")

    def set_budget(
        self,
        daily: Optional[float] = None,
        weekly: Optional[float] = None,
        monthly: Optional[float] = None,
        enforce: bool = False,
        alerts_enabled: bool = True,
    ):
        """
        Set budget limits for the project.

        Args:
            daily: Daily limit in USD
            weekly: Weekly limit in USD
            monthly: Monthly limit in USD
            enforce: Whether to block execution when exceeded
            alerts_enabled: Whether to trigger alerts
        """
        now = datetime.utcnow().isoformat()

        if self._config:
            self._config.daily_limit = daily
            self._config.weekly_limit = weekly
            self._config.monthly_limit = monthly
            self._config.enforce_limits = enforce
            self._config.alerts_enabled = alerts_enabled
            self._config.updated_at = now
        else:
            self._config = BudgetConfig(
                id=f"budget_{uuid.uuid4().hex[:8]}",
                project=self.project,
                created_at=now,
                updated_at=now,
                daily_limit=daily,
                weekly_limit=weekly,
                monthly_limit=monthly,
                enforce_limits=enforce,
                alerts_enabled=alerts_enabled,
            )

        self._save_config()

        # Reset triggered thresholds for new budget
        self._triggered_thresholds = {
            "daily": set(),
            "weekly": set(),
            "monthly": set(),
        }

    def get_config(self) -> Optional[BudgetConfig]:
        """Get current budget configuration."""
        return self._config

    def get_status(self) -> BudgetStatus:
        """
        Get current budget status.

        Returns:
            BudgetStatus with usage and limit information
        """
        now = datetime.utcnow()
        tracker = get_cost_tracker(self.project)

        # Calculate period boundaries
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = day_start - timedelta(days=day_start.weekday())
        month_start = day_start.replace(day=1)

        # Get usage for each period
        daily_usage = tracker.get_total_spend(
            start_date=day_start.isoformat(),
            end_date=now.isoformat(),
        )
        weekly_usage = tracker.get_total_spend(
            start_date=week_start.isoformat(),
            end_date=now.isoformat(),
        )
        monthly_usage = tracker.get_total_spend(
            start_date=month_start.isoformat(),
            end_date=now.isoformat(),
        )

        # Get limits
        daily_limit = self._config.daily_limit if self._config else None
        weekly_limit = self._config.weekly_limit if self._config else None
        monthly_limit = self._config.monthly_limit if self._config else None

        # Calculate percentages
        daily_pct = (daily_usage / daily_limit) if daily_limit else None
        weekly_pct = (weekly_usage / weekly_limit) if weekly_limit else None
        monthly_pct = (monthly_usage / monthly_limit) if monthly_limit else None

        # Determine alert level
        max_pct = max(
            p for p in [daily_pct, weekly_pct, monthly_pct, 0.0] if p is not None
        )
        if max_pct >= 1.0:
            alert_level = AlertLevel.EXCEEDED
        elif max_pct >= 0.9:
            alert_level = AlertLevel.CRITICAL
        elif max_pct >= 0.75:
            alert_level = AlertLevel.WARNING
        elif max_pct >= 0.5:
            alert_level = AlertLevel.INFO
        else:
            alert_level = AlertLevel.INFO

        # Calculate projections
        days_elapsed = (now - month_start).days + 1
        days_in_month = 30  # Approximation
        projected_monthly = (monthly_usage / days_elapsed) * days_in_month if days_elapsed > 0 else 0

        days_until_limit = None
        if monthly_limit and daily_usage > 0:
            remaining = monthly_limit - monthly_usage
            daily_rate = monthly_usage / days_elapsed
            if daily_rate > 0:
                days_until_limit = int(remaining / daily_rate)

        return BudgetStatus(
            project=self.project,
            timestamp=now.isoformat(),
            daily_usage=round(daily_usage, 4),
            weekly_usage=round(weekly_usage, 4),
            monthly_usage=round(monthly_usage, 4),
            daily_limit=daily_limit,
            weekly_limit=weekly_limit,
            monthly_limit=monthly_limit,
            daily_percentage=round(daily_pct, 4) if daily_pct else None,
            weekly_percentage=round(weekly_pct, 4) if weekly_pct else None,
            monthly_percentage=round(monthly_pct, 4) if monthly_pct else None,
            current_alert_level=alert_level,
            alerts_triggered=[a.to_dict() for a in self.get_alerts(unacknowledged_only=True)],
            projected_monthly_spend=round(projected_monthly, 4),
            days_until_limit=days_until_limit,
        )

    def check_budget(
        self,
        estimated_cost: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if estimated cost would exceed budget.

        Args:
            estimated_cost: Estimated cost of upcoming operation

        Returns:
            Tuple of (allowed, warning_message)
        """
        if not self._config:
            return True, None

        status = self.get_status()

        # Check daily limit
        if status.daily_limit:
            if status.daily_usage + estimated_cost > status.daily_limit:
                msg = f"Would exceed daily budget (${status.daily_usage:.4f} + ${estimated_cost:.4f} > ${status.daily_limit:.2f})"
                if self._config.enforce_limits:
                    return False, msg
                return True, msg

        # Check weekly limit
        if status.weekly_limit:
            if status.weekly_usage + estimated_cost > status.weekly_limit:
                msg = f"Would exceed weekly budget (${status.weekly_usage:.4f} + ${estimated_cost:.4f} > ${status.weekly_limit:.2f})"
                if self._config.enforce_limits:
                    return False, msg
                return True, msg

        # Check monthly limit
        if status.monthly_limit:
            if status.monthly_usage + estimated_cost > status.monthly_limit:
                msg = f"Would exceed monthly budget (${status.monthly_usage:.4f} + ${estimated_cost:.4f} > ${status.monthly_limit:.2f})"
                if self._config.enforce_limits:
                    return False, msg
                return True, msg

        return True, None

    def record_spend(self, amount: float):
        """
        Record spending and check for alerts.

        Args:
            amount: Amount spent in USD
        """
        if not self._config or not self._config.alerts_enabled:
            return

        status = self.get_status()
        self._check_alerts(status)

    def _check_alerts(self, status: BudgetStatus):
        """
        Check and trigger budget alerts.

        Args:
            status: Current budget status
        """
        if not self._config:
            return

        thresholds = self._config.alert_thresholds

        # Check each period
        periods = [
            ("daily", status.daily_percentage, status.daily_usage, status.daily_limit),
            ("weekly", status.weekly_percentage, status.weekly_usage, status.weekly_limit),
            ("monthly", status.monthly_percentage, status.monthly_usage, status.monthly_limit),
        ]

        for period_name, percentage, usage, limit in periods:
            if percentage is None or limit is None:
                continue

            for threshold in thresholds:
                if percentage >= threshold:
                    threshold_key = f"{threshold}"
                    if threshold_key not in self._triggered_thresholds[period_name]:
                        # Create alert
                        alert = self._create_alert(
                            period_name, threshold, percentage, usage, limit
                        )
                        self._alerts.append(alert)
                        self._triggered_thresholds[period_name].add(threshold_key)

        self._save_alerts()

    def _create_alert(
        self,
        period: str,
        threshold: float,
        percentage: float,
        usage: float,
        limit: float,
    ) -> BudgetAlert:
        """Create a budget alert."""
        # Determine alert level
        if threshold >= 1.0:
            level = AlertLevel.EXCEEDED
        elif threshold >= 0.9:
            level = AlertLevel.CRITICAL
        elif threshold >= 0.75:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO

        # Create message
        pct_str = f"{percentage * 100:.1f}%"
        if level == AlertLevel.EXCEEDED:
            message = f"{period.capitalize()} budget EXCEEDED: ${usage:.4f} of ${limit:.2f} ({pct_str})"
        else:
            message = f"{period.capitalize()} budget at {pct_str}: ${usage:.4f} of ${limit:.2f}"

        return BudgetAlert(
            id=f"alert_{uuid.uuid4().hex[:8]}",
            project=self.project,
            timestamp=datetime.utcnow().isoformat(),
            alert_level=level,
            period=BudgetPeriod(period),
            current_usage=usage,
            limit=limit,
            percentage=percentage,
            message=message,
        )

    def get_alerts(
        self,
        unacknowledged_only: bool = True,
        limit: int = 50,
    ) -> List[BudgetAlert]:
        """
        Get budget alerts.

        Args:
            unacknowledged_only: Only return unacknowledged alerts
            limit: Maximum number of alerts

        Returns:
            List of BudgetAlert objects
        """
        alerts = self._alerts

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        # Sort by timestamp descending
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return alerts[:limit]

    def acknowledge_alert(self, alert_id: str):
        """
        Acknowledge a budget alert.

        Args:
            alert_id: Alert ID to acknowledge
        """
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow().isoformat()
                break

        self._save_alerts()

    def clear_alerts(self):
        """Clear all alerts."""
        self._alerts = []
        self._triggered_thresholds = {
            "daily": set(),
            "weekly": set(),
            "monthly": set(),
        }
        self._save_alerts()

    def reset_daily_tracking(self):
        """Reset daily threshold tracking (call at midnight)."""
        self._triggered_thresholds["daily"] = set()

    def reset_weekly_tracking(self):
        """Reset weekly threshold tracking (call at week start)."""
        self._triggered_thresholds["weekly"] = set()

    def reset_monthly_tracking(self):
        """Reset monthly threshold tracking (call at month start)."""
        self._triggered_thresholds["monthly"] = set()


# Singleton instance per project
_budget_managers: Dict[str, BudgetManager] = {}


def get_budget_manager(project: str = "default") -> BudgetManager:
    """Get or create BudgetManager for a project."""
    global _budget_managers
    if project not in _budget_managers:
        _budget_managers[project] = BudgetManager(project)
    return _budget_managers[project]
