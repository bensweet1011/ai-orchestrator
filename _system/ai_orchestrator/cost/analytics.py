"""
Cost analytics and reporting for Cost Intelligence.
"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PROJECTS_DIR
from .state import BudgetPeriod, CostRecord


class CostAnalytics:
    """Cost analysis and reporting."""

    def __init__(self, project: str = "default"):
        """
        Initialize cost analytics.

        Args:
            project: Project name
        """
        self.project = project

    def _get_cost_dir(self) -> Path:
        """Get cost records directory."""
        return PROJECTS_DIR / self.project / "cost_records"

    def _load_records(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[CostRecord]:
        """
        Load cost records within date range.

        Args:
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)

        Returns:
            List of CostRecords
        """
        records: List[CostRecord] = []
        cost_dir = self._get_cost_dir()

        if not cost_dir.exists():
            return records

        for month_dir in cost_dir.iterdir():
            if not month_dir.is_dir():
                continue

            for file_path in month_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    timestamp = data.get("timestamp", "")

                    # Apply date filters
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue

                    records.append(CostRecord.from_dict(data))
                except Exception:
                    continue

        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp)
        return records

    def get_spend_by_period(
        self,
        period: BudgetPeriod = BudgetPeriod.DAILY,
        num_periods: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get spending breakdown by time period.

        Args:
            period: Period granularity
            num_periods: Number of periods to return

        Returns:
            List of dicts with date and cost
        """
        now = datetime.utcnow()

        # Calculate start date based on period
        if period == BudgetPeriod.DAILY:
            start = now - timedelta(days=num_periods)
            date_format = "%Y-%m-%d"
        elif period == BudgetPeriod.WEEKLY:
            start = now - timedelta(weeks=num_periods)
            date_format = "%Y-W%W"
        else:  # MONTHLY
            start = now - timedelta(days=num_periods * 30)
            date_format = "%Y-%m"

        records = self._load_records(
            start_date=start.isoformat(),
            end_date=now.isoformat(),
        )

        # Aggregate by period
        spend_by_period: Dict[str, float] = defaultdict(float)
        tokens_by_period: Dict[str, int] = defaultdict(int)

        for record in records:
            try:
                dt = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
                period_key = dt.strftime(date_format)
                spend_by_period[period_key] += record.total_cost
                tokens_by_period[period_key] += record.total_tokens
            except Exception:
                continue

        # Convert to list
        result = []
        for period_key in sorted(spend_by_period.keys()):
            result.append({
                "period": period_key,
                "cost": round(spend_by_period[period_key], 4),
                "tokens": tokens_by_period[period_key],
            })

        return result[-num_periods:]

    def get_spend_by_model(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get spending breakdown by model.

        Args:
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dict mapping model to cost/token info
        """
        records = self._load_records(start_date, end_date)

        model_costs: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
        )

        for record in records:
            for node_cost in record.node_costs:
                model = node_cost.model
                model_costs[model]["cost"] += node_cost.cost
                model_costs[model]["input_tokens"] += node_cost.input_tokens
                model_costs[model]["output_tokens"] += node_cost.output_tokens
                model_costs[model]["calls"] += 1

        # Round costs
        for model in model_costs:
            model_costs[model]["cost"] = round(model_costs[model]["cost"], 4)

        return dict(model_costs)

    def get_spend_by_pipeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get spending breakdown by pipeline.

        Args:
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dict mapping pipeline name to cost/stats
        """
        records = self._load_records(start_date, end_date)

        pipeline_costs: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "cost": 0.0,
                "tokens": 0,
                "runs": 0,
                "avg_cost": 0.0,
                "avg_latency_ms": 0,
            }
        )

        for record in records:
            pipeline = record.pipeline_name or "single_call"
            pipeline_costs[pipeline]["cost"] += record.total_cost
            pipeline_costs[pipeline]["tokens"] += record.total_tokens
            pipeline_costs[pipeline]["runs"] += 1
            pipeline_costs[pipeline]["total_latency"] = (
                pipeline_costs[pipeline].get("total_latency", 0) + record.latency_ms
            )

        # Calculate averages
        for pipeline in pipeline_costs:
            runs = pipeline_costs[pipeline]["runs"]
            if runs > 0:
                pipeline_costs[pipeline]["avg_cost"] = round(
                    pipeline_costs[pipeline]["cost"] / runs, 4
                )
                pipeline_costs[pipeline]["avg_latency_ms"] = int(
                    pipeline_costs[pipeline].get("total_latency", 0) / runs
                )
            pipeline_costs[pipeline]["cost"] = round(pipeline_costs[pipeline]["cost"], 4)
            if "total_latency" in pipeline_costs[pipeline]:
                del pipeline_costs[pipeline]["total_latency"]

        return dict(pipeline_costs)

    def get_estimation_accuracy(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze estimation accuracy over time.

        Args:
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dict with accuracy statistics
        """
        records = self._load_records(start_date, end_date)

        variances: List[float] = []
        accurate_count = 0
        over_count = 0
        under_count = 0

        for record in records:
            if record.estimate_variance is not None:
                variances.append(record.estimate_variance)
                if abs(record.estimate_variance) <= 0.2:  # Within 20%
                    accurate_count += 1
                elif record.estimate_variance > 0:
                    under_count += 1  # Actual > estimated
                else:
                    over_count += 1  # Actual < estimated

        total_with_estimate = len(variances)
        if total_with_estimate == 0:
            return {
                "total_records": len(records),
                "records_with_estimates": 0,
                "avg_variance": None,
                "accuracy_rate": None,
            }

        avg_variance = sum(variances) / total_with_estimate
        accuracy_rate = accurate_count / total_with_estimate

        return {
            "total_records": len(records),
            "records_with_estimates": total_with_estimate,
            "avg_variance": round(avg_variance, 4),
            "accuracy_rate": round(accuracy_rate, 4),
            "over_estimated": over_count,
            "under_estimated": under_count,
            "accurate": accurate_count,
        }

    def get_summary_stats(
        self,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
    ) -> Dict[str, Any]:
        """
        Get summary statistics for cost dashboard.

        Args:
            period: Period for comparison

        Returns:
            Dict with summary stats
        """
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = today_start.replace(day=1)

        # Load all records
        records = self._load_records()

        # Calculate totals
        total_cost = sum(r.total_cost for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_runs = len(records)

        # Today's stats
        today_records = [
            r for r in records
            if r.timestamp >= today_start.isoformat()
        ]
        today_cost = sum(r.total_cost for r in today_records)
        today_tokens = sum(r.total_tokens for r in today_records)

        # Week stats
        week_records = [
            r for r in records
            if r.timestamp >= week_start.isoformat()
        ]
        week_cost = sum(r.total_cost for r in week_records)

        # Month stats
        month_records = [
            r for r in records
            if r.timestamp >= month_start.isoformat()
        ]
        month_cost = sum(r.total_cost for r in month_records)

        # Calculate change (vs previous period)
        yesterday_start = today_start - timedelta(days=1)
        yesterday_records = [
            r for r in records
            if yesterday_start.isoformat() <= r.timestamp < today_start.isoformat()
        ]
        yesterday_cost = sum(r.total_cost for r in yesterday_records)
        daily_change = (
            ((today_cost - yesterday_cost) / yesterday_cost)
            if yesterday_cost > 0 else 0.0
        )

        # Most expensive model
        model_costs = self.get_spend_by_model(
            start_date=month_start.isoformat(),
            end_date=now.isoformat(),
        )
        top_model = max(model_costs.items(), key=lambda x: x[1]["cost"])[0] if model_costs else None

        # Most expensive pipeline
        pipeline_costs = self.get_spend_by_pipeline(
            start_date=month_start.isoformat(),
            end_date=now.isoformat(),
        )
        top_pipeline = max(pipeline_costs.items(), key=lambda x: x[1]["cost"])[0] if pipeline_costs else None

        return {
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_runs": total_runs,
            "today_cost": round(today_cost, 4),
            "today_tokens": today_tokens,
            "week_cost": round(week_cost, 4),
            "month_cost": round(month_cost, 4),
            "daily_change": round(daily_change, 4),
            "avg_cost_per_run": round(total_cost / total_runs, 4) if total_runs > 0 else 0,
            "top_model": top_model,
            "top_pipeline": top_pipeline,
        }

    def export_csv(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export cost records to CSV.

        Args:
            start_date: Optional start date
            end_date: Optional end date
            output_path: Output file path (default: temp file)

        Returns:
            Path to CSV file
        """
        records = self._load_records(start_date, end_date)

        if not output_path:
            export_dir = PROJECTS_DIR / self.project / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = export_dir / f"cost_export_{timestamp}.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "timestamp",
                "pipeline_name",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "total_cost",
                "models_used",
                "latency_ms",
                "success",
                "estimated_cost",
                "estimate_variance",
            ])

            # Data rows
            for record in records:
                writer.writerow([
                    record.timestamp,
                    record.pipeline_name or "single_call",
                    record.total_input_tokens,
                    record.total_output_tokens,
                    record.total_tokens,
                    record.total_cost,
                    ",".join(record.models_used),
                    record.latency_ms,
                    record.success,
                    record.estimated_cost or "",
                    record.estimate_variance or "",
                ])

        return output_path

    def get_cost_trend(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get daily cost trend for charting.

        Args:
            days: Number of days to include

        Returns:
            List of daily cost data points
        """
        now = datetime.utcnow()
        start = now - timedelta(days=days)

        records = self._load_records(
            start_date=start.isoformat(),
            end_date=now.isoformat(),
        )

        # Initialize all days with zero
        daily_costs: Dict[str, Dict[str, Any]] = {}
        current = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while current <= now:
            date_key = current.strftime("%Y-%m-%d")
            daily_costs[date_key] = {
                "date": date_key,
                "cost": 0.0,
                "tokens": 0,
                "runs": 0,
            }
            current += timedelta(days=1)

        # Aggregate records
        for record in records:
            try:
                dt = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
                date_key = dt.strftime("%Y-%m-%d")
                if date_key in daily_costs:
                    daily_costs[date_key]["cost"] += record.total_cost
                    daily_costs[date_key]["tokens"] += record.total_tokens
                    daily_costs[date_key]["runs"] += 1
            except Exception:
                continue

        # Convert to sorted list
        result = list(daily_costs.values())
        result.sort(key=lambda x: x["date"])

        # Round costs
        for item in result:
            item["cost"] = round(item["cost"], 4)

        return result


# Singleton instance per project
_analytics: Dict[str, CostAnalytics] = {}


def get_cost_analytics(project: str = "default") -> CostAnalytics:
    """Get or create CostAnalytics for a project."""
    global _analytics
    if project not in _analytics:
        _analytics[project] = CostAnalytics(project)
    return _analytics[project]
