"""
Data structures for Cost Intelligence system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class PricingTier(Enum):
    """Pricing tiers for different usage levels."""

    STANDARD = "standard"
    BATCH = "batch"
    CACHED = "cached"


class BudgetPeriod(Enum):
    """Budget period types."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AlertLevel(Enum):
    """Budget alert severity levels."""

    INFO = "info"  # 50% threshold
    WARNING = "warning"  # 75% threshold
    CRITICAL = "critical"  # 90% threshold
    EXCEEDED = "exceeded"  # 100%+ threshold


# =========================================================================
# Pricing Data Structures
# =========================================================================


@dataclass
class ModelPricing:
    """Pricing information for a specific model."""

    model_id: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    effective_date: str
    currency: str = "USD"
    cached_input_cost_per_1k: Optional[float] = None
    batch_discount: float = 0.0
    notes: str = ""

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        tier: PricingTier = PricingTier.STANDARD,
    ) -> float:
        """Calculate total cost for given token counts."""
        input_rate = self.input_cost_per_1k
        if tier == PricingTier.CACHED and self.cached_input_cost_per_1k:
            input_rate = self.cached_input_cost_per_1k

        cost = (input_tokens / 1000 * input_rate) + (
            output_tokens / 1000 * self.output_cost_per_1k
        )

        if tier == PricingTier.BATCH:
            cost *= 1 - self.batch_discount

        return round(cost, 6)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            "effective_date": self.effective_date,
            "currency": self.currency,
            "cached_input_cost_per_1k": self.cached_input_cost_per_1k,
            "batch_discount": self.batch_discount,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelPricing":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            provider=data["provider"],
            input_cost_per_1k=data["input_cost_per_1k"],
            output_cost_per_1k=data["output_cost_per_1k"],
            effective_date=data.get("effective_date", ""),
            currency=data.get("currency", "USD"),
            cached_input_cost_per_1k=data.get("cached_input_cost_per_1k"),
            batch_discount=data.get("batch_discount", 0.0),
            notes=data.get("notes", ""),
        )


# =========================================================================
# Cost Estimation Structures
# =========================================================================


@dataclass
class NodeEstimate:
    """Cost estimate for a single node."""

    node_name: str
    model: str
    provider: str
    input_tokens: int
    output_tokens_low: int
    output_tokens_likely: int
    output_tokens_high: int
    cost_low: float
    cost_likely: float
    cost_high: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_name": self.node_name,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens_low": self.output_tokens_low,
            "output_tokens_likely": self.output_tokens_likely,
            "output_tokens_high": self.output_tokens_high,
            "cost_low": self.cost_low,
            "cost_likely": self.cost_likely,
            "cost_high": self.cost_high,
        }


@dataclass
class CostEstimate:
    """Pre-run cost estimate for a pipeline or LLM call."""

    id: str
    timestamp: str
    pipeline_name: Optional[str]

    # Token estimates
    estimated_input_tokens: int
    estimated_output_tokens_low: int
    estimated_output_tokens_likely: int
    estimated_output_tokens_high: int

    # Cost range
    estimated_cost_low: float
    estimated_cost_likely: float
    estimated_cost_high: float

    # Breakdown by node (for pipelines)
    node_estimates: List[NodeEstimate] = field(default_factory=list)

    # Metadata
    models_used: List[str] = field(default_factory=list)
    estimation_method: str = "input_ratio"
    confidence: float = 0.5
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "pipeline_name": self.pipeline_name,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens_low": self.estimated_output_tokens_low,
            "estimated_output_tokens_likely": self.estimated_output_tokens_likely,
            "estimated_output_tokens_high": self.estimated_output_tokens_high,
            "estimated_cost_low": self.estimated_cost_low,
            "estimated_cost_likely": self.estimated_cost_likely,
            "estimated_cost_high": self.estimated_cost_high,
            "node_estimates": [n.to_dict() for n in self.node_estimates],
            "models_used": self.models_used,
            "estimation_method": self.estimation_method,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }


# =========================================================================
# Cost Record Structures
# =========================================================================


@dataclass
class NodeCostRecord:
    """Cost record for a single node execution."""

    node_name: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency_ms: int
    timestamp: str

    # Optional estimation comparison
    estimated_cost: Optional[float] = None
    estimation_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_name": self.node_name,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "estimated_cost": self.estimated_cost,
            "estimation_accuracy": self.estimation_accuracy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeCostRecord":
        """Create from dictionary."""
        return cls(
            node_name=data["node_name"],
            model=data["model"],
            provider=data["provider"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            total_tokens=data["total_tokens"],
            cost=data["cost"],
            latency_ms=data["latency_ms"],
            timestamp=data["timestamp"],
            estimated_cost=data.get("estimated_cost"),
            estimation_accuracy=data.get("estimation_accuracy"),
        )


@dataclass
class CostRecord:
    """Complete cost record for a pipeline run or LLM call."""

    id: str
    timestamp: str
    project: str
    session_id: str

    # Association
    pipeline_id: Optional[str]
    pipeline_name: Optional[str]
    memory_id: Optional[str]  # Links to PipelineMemory.id

    # Totals
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: float

    # Breakdown
    node_costs: List[NodeCostRecord] = field(default_factory=list)

    # Comparison with estimate
    pre_estimate_id: Optional[str] = None
    estimated_cost: Optional[float] = None
    estimate_variance: Optional[float] = None

    # Execution metadata
    success: bool = True
    latency_ms: int = 0
    models_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "project": self.project,
            "session_id": self.session_id,
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "memory_id": self.memory_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "node_costs": [n.to_dict() for n in self.node_costs],
            "pre_estimate_id": self.pre_estimate_id,
            "estimated_cost": self.estimated_cost,
            "estimate_variance": self.estimate_variance,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "models_used": self.models_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            project=data["project"],
            session_id=data["session_id"],
            pipeline_id=data.get("pipeline_id"),
            pipeline_name=data.get("pipeline_name"),
            memory_id=data.get("memory_id"),
            total_input_tokens=data["total_input_tokens"],
            total_output_tokens=data["total_output_tokens"],
            total_tokens=data["total_tokens"],
            total_cost=data["total_cost"],
            node_costs=[NodeCostRecord.from_dict(n) for n in data.get("node_costs", [])],
            pre_estimate_id=data.get("pre_estimate_id"),
            estimated_cost=data.get("estimated_cost"),
            estimate_variance=data.get("estimate_variance"),
            success=data.get("success", True),
            latency_ms=data.get("latency_ms", 0),
            models_used=data.get("models_used", []),
        )


# =========================================================================
# Budget Structures
# =========================================================================


@dataclass
class BudgetConfig:
    """Budget configuration for a project."""

    id: str
    project: str
    created_at: str
    updated_at: str

    # Budget limits (in USD)
    daily_limit: Optional[float] = None
    weekly_limit: Optional[float] = None
    monthly_limit: Optional[float] = None

    # Alert thresholds (percentages as decimals)
    alert_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 0.9, 1.0]
    )

    # Alert settings
    alerts_enabled: bool = True

    # Enforcement
    enforce_limits: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "project": self.project,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "daily_limit": self.daily_limit,
            "weekly_limit": self.weekly_limit,
            "monthly_limit": self.monthly_limit,
            "alert_thresholds": self.alert_thresholds,
            "alerts_enabled": self.alerts_enabled,
            "enforce_limits": self.enforce_limits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetConfig":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            project=data["project"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            daily_limit=data.get("daily_limit"),
            weekly_limit=data.get("weekly_limit"),
            monthly_limit=data.get("monthly_limit"),
            alert_thresholds=data.get("alert_thresholds", [0.5, 0.75, 0.9, 1.0]),
            alerts_enabled=data.get("alerts_enabled", True),
            enforce_limits=data.get("enforce_limits", False),
        )


@dataclass
class BudgetStatus:
    """Current budget status for a project."""

    project: str
    timestamp: str

    # Usage by period (in USD)
    daily_usage: float
    weekly_usage: float
    monthly_usage: float

    # Limits
    daily_limit: Optional[float]
    weekly_limit: Optional[float]
    monthly_limit: Optional[float]

    # Percentages (0.0 - 1.0+)
    daily_percentage: Optional[float] = None
    weekly_percentage: Optional[float] = None
    monthly_percentage: Optional[float] = None

    # Alert status
    current_alert_level: AlertLevel = AlertLevel.INFO
    alerts_triggered: List[Dict[str, Any]] = field(default_factory=list)

    # Projections
    projected_monthly_spend: float = 0.0
    days_until_limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project": self.project,
            "timestamp": self.timestamp,
            "daily_usage": self.daily_usage,
            "weekly_usage": self.weekly_usage,
            "monthly_usage": self.monthly_usage,
            "daily_limit": self.daily_limit,
            "weekly_limit": self.weekly_limit,
            "monthly_limit": self.monthly_limit,
            "daily_percentage": self.daily_percentage,
            "weekly_percentage": self.weekly_percentage,
            "monthly_percentage": self.monthly_percentage,
            "current_alert_level": self.current_alert_level.value,
            "alerts_triggered": self.alerts_triggered,
            "projected_monthly_spend": self.projected_monthly_spend,
            "days_until_limit": self.days_until_limit,
        }


@dataclass
class BudgetAlert:
    """Budget alert record."""

    id: str
    project: str
    timestamp: str
    alert_level: AlertLevel
    period: BudgetPeriod
    current_usage: float
    limit: float
    percentage: float
    message: str
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "project": self.project,
            "timestamp": self.timestamp,
            "alert_level": self.alert_level.value,
            "period": self.period.value,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "percentage": self.percentage,
            "message": self.message,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetAlert":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            project=data["project"],
            timestamp=data["timestamp"],
            alert_level=AlertLevel(data["alert_level"]),
            period=BudgetPeriod(data["period"]),
            current_usage=data["current_usage"],
            limit=data["limit"],
            percentage=data["percentage"],
            message=data["message"],
            acknowledged=data.get("acknowledged", False),
            acknowledged_at=data.get("acknowledged_at"),
        )
