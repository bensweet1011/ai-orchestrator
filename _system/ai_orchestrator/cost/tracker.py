"""
Post-run cost tracking for Cost Intelligence.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..config import PROJECTS_DIR
from .estimator import get_cost_estimator
from .pricing import get_pricing_manager
from .state import CostEstimate, CostRecord, NodeCostRecord

if TYPE_CHECKING:
    from ..pipelines.state import PipelineResult
    from ..core.llm_clients import LLMResponse


class CostTracker:
    """Tracks and stores execution costs."""

    def __init__(self, project: str = "default"):
        """
        Initialize cost tracker.

        Args:
            project: Project name for cost isolation
        """
        self.project = project
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.pricing_manager = get_pricing_manager()
        self._cost_cache: Dict[str, CostRecord] = {}

    def track_pipeline_run(
        self,
        result: "PipelineResult",
        pipeline_name: str,
        pre_estimate: Optional[CostEstimate] = None,
        memory_id: Optional[str] = None,
    ) -> CostRecord:
        """
        Track costs for a completed pipeline run.

        Args:
            result: Pipeline execution result
            pipeline_name: Name of the pipeline
            pre_estimate: Optional pre-run cost estimate
            memory_id: Optional memory store ID

        Returns:
            CostRecord with full cost breakdown
        """
        record_id = f"cost_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.utcnow().isoformat()

        # Calculate cost per node
        node_costs: List[NodeCostRecord] = []
        total_input = 0
        total_output = 0
        total_cost = 0.0
        models_used: List[str] = []

        for node_name, output in result.node_outputs.items():
            tokens_used = output.get("tokens_used", {})
            input_tokens = tokens_used.get("input_tokens", 0)
            output_tokens = tokens_used.get("output_tokens", 0)

            model = output.get("model", "unknown")
            provider = output.get("provider", "unknown")

            if model not in models_used:
                models_used.append(model)

            # Calculate node cost
            cost = self.pricing_manager.calculate_cost(model, input_tokens, output_tokens)

            # Find matching node estimate
            estimated_cost = None
            estimation_accuracy = None
            if pre_estimate:
                for node_est in pre_estimate.node_estimates:
                    if node_est.node_name == node_name:
                        estimated_cost = node_est.cost_likely
                        if estimated_cost > 0:
                            estimation_accuracy = cost / estimated_cost
                        break

            node_cost = NodeCostRecord(
                node_name=node_name,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost=cost,
                latency_ms=output.get("latency_ms", 0),
                timestamp=timestamp,
                estimated_cost=estimated_cost,
                estimation_accuracy=estimation_accuracy,
            )
            node_costs.append(node_cost)

            total_input += input_tokens
            total_output += output_tokens
            total_cost += cost

        # Calculate estimate variance
        estimated_cost = None
        estimate_variance = None
        if pre_estimate:
            estimated_cost = pre_estimate.estimated_cost_likely
            if estimated_cost > 0:
                estimate_variance = (total_cost - estimated_cost) / estimated_cost

        cost_record = CostRecord(
            id=record_id,
            timestamp=timestamp,
            project=self.project,
            session_id=self.session_id,
            pipeline_id=result.pipeline_id,
            pipeline_name=pipeline_name,
            memory_id=memory_id,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            total_cost=round(total_cost, 6),
            node_costs=node_costs,
            pre_estimate_id=pre_estimate.id if pre_estimate else None,
            estimated_cost=estimated_cost,
            estimate_variance=estimate_variance,
            success=result.success,
            latency_ms=result.total_latency_ms,
            models_used=models_used,
        )

        # Store the record
        self._store_cost_record(cost_record)

        # Update historical ratios for better future estimates
        self._update_historical_ratios(cost_record)

        # Cache for retrieval
        self._cost_cache[record_id] = cost_record

        return cost_record

    def track_llm_call(
        self,
        response: "LLMResponse",
        latency_ms: int,
        pre_estimate: Optional[CostEstimate] = None,
    ) -> NodeCostRecord:
        """
        Track cost for a single LLM call.

        Args:
            response: LLM response with usage data
            latency_ms: Call latency in milliseconds
            pre_estimate: Optional pre-run cost estimate

        Returns:
            NodeCostRecord for the call
        """
        timestamp = datetime.utcnow().isoformat()

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.get("input_tokens", 0)
            output_tokens = response.usage.get("output_tokens", 0)

        model = response.model
        provider = response.provider

        cost = self.pricing_manager.calculate_cost(model, input_tokens, output_tokens)

        estimated_cost = None
        estimation_accuracy = None
        if pre_estimate:
            estimated_cost = pre_estimate.estimated_cost_likely
            if estimated_cost > 0:
                estimation_accuracy = cost / estimated_cost

        return NodeCostRecord(
            node_name="single_call",
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            timestamp=timestamp,
            estimated_cost=estimated_cost,
            estimation_accuracy=estimation_accuracy,
        )

    def _store_cost_record(self, record: CostRecord):
        """
        Store cost record to local JSON file.

        Args:
            record: Cost record to store
        """
        # Create directory structure
        cost_dir = PROJECTS_DIR / self.project / "cost_records"
        month_dir = cost_dir / datetime.utcnow().strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)

        # Store as JSON
        file_path = month_dir / f"{record.id}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(record.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to store cost record: {e}")

    def _update_historical_ratios(self, record: CostRecord):
        """
        Update historical ratios based on actual usage.

        Args:
            record: Cost record with actual token counts
        """
        estimator = get_cost_estimator()

        for node_cost in record.node_costs:
            if node_cost.input_tokens > 0:
                estimator.update_historical_ratio(
                    node_cost.model,
                    node_cost.input_tokens,
                    node_cost.output_tokens,
                )

    def get_cost_record(self, record_id: str) -> Optional[CostRecord]:
        """
        Retrieve a cost record by ID.

        Args:
            record_id: Cost record ID

        Returns:
            CostRecord or None
        """
        # Check cache first
        if record_id in self._cost_cache:
            return self._cost_cache[record_id]

        # Search in files
        cost_dir = PROJECTS_DIR / self.project / "cost_records"
        if not cost_dir.exists():
            return None

        for month_dir in cost_dir.iterdir():
            if not month_dir.is_dir():
                continue
            file_path = month_dir / f"{record_id}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    return CostRecord.from_dict(data)
                except Exception:
                    pass

        return None

    def get_recent_records(
        self,
        limit: int = 50,
        pipeline_name: Optional[str] = None,
    ) -> List[CostRecord]:
        """
        Get recent cost records.

        Args:
            limit: Maximum number of records
            pipeline_name: Optional filter by pipeline

        Returns:
            List of CostRecords sorted by timestamp descending
        """
        records: List[CostRecord] = []

        cost_dir = PROJECTS_DIR / self.project / "cost_records"
        if not cost_dir.exists():
            return records

        # Get all JSON files sorted by modification time
        files: List[Path] = []
        for month_dir in cost_dir.iterdir():
            if month_dir.is_dir():
                files.extend(month_dir.glob("*.json"))

        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for file_path in files:
            if len(records) >= limit:
                break

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                record = CostRecord.from_dict(data)

                if pipeline_name and record.pipeline_name != pipeline_name:
                    continue

                records.append(record)
            except Exception:
                continue

        return records

    def get_total_spend(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> float:
        """
        Get total spend for the project.

        Args:
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)

        Returns:
            Total spend in USD
        """
        total = 0.0

        cost_dir = PROJECTS_DIR / self.project / "cost_records"
        if not cost_dir.exists():
            return total

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

                    total += data.get("total_cost", 0.0)
                except Exception:
                    continue

        return round(total, 6)


# Singleton instance per project
_cost_trackers: Dict[str, CostTracker] = {}


def get_cost_tracker(project: str = "default") -> CostTracker:
    """Get or create CostTracker for a project."""
    global _cost_trackers
    if project not in _cost_trackers:
        _cost_trackers[project] = CostTracker(project)
    return _cost_trackers[project]
