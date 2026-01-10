"""
Pre-run cost estimation for Cost Intelligence.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .pricing import get_pricing_manager
from .state import CostEstimate, NodeEstimate
from .tokenizer import get_token_counter

if TYPE_CHECKING:
    from ..pipelines.base import BasePipeline

# Default output/input token ratios by model type
# Based on empirical observation of model behavior
DEFAULT_OUTPUT_RATIOS = {
    "claude": {"low": 0.5, "likely": 1.0, "high": 2.0},
    "claude-opus": {"low": 0.8, "likely": 1.5, "high": 3.0},
    "gpt4o": {"low": 0.4, "likely": 0.8, "high": 1.5},
    "gpt4": {"low": 0.5, "likely": 1.0, "high": 2.0},
    "o1": {"low": 1.0, "likely": 2.0, "high": 4.0},  # Reasoning models output more
    "o3-mini": {"low": 0.8, "likely": 1.5, "high": 3.0},
    "gemini": {"low": 0.5, "likely": 1.0, "high": 2.0},
    "grok": {"low": 0.5, "likely": 1.0, "high": 2.0},
    "perplexity": {"low": 0.5, "likely": 1.0, "high": 2.0},
    "default": {"low": 0.5, "likely": 1.0, "high": 2.0},
}


class CostEstimator:
    """Pre-run cost estimation for pipelines and LLM calls."""

    def __init__(self):
        """Initialize cost estimator."""
        self.token_counter = get_token_counter()
        self.pricing_manager = get_pricing_manager()
        self._historical_ratios: Dict[str, Dict[str, float]] = {}

    def estimate_llm_call(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        provider: Optional[str] = None,
    ) -> CostEstimate:
        """
        Estimate cost for a single LLM call.

        Args:
            prompt: User prompt
            model: Model name
            system_prompt: Optional system prompt
            max_tokens: Maximum output tokens
            provider: Provider name

        Returns:
            CostEstimate with range estimates
        """
        estimate_id = f"est_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.utcnow().isoformat()

        # Count input tokens
        input_tokens = self.token_counter.count_with_system(
            prompt, system_prompt, model, provider
        )

        # Get output token estimates
        output_low, output_likely, output_high = self._estimate_output_tokens(
            input_tokens, model, max_tokens
        )

        # Calculate costs
        pricing = self.pricing_manager.get_pricing(model)
        warnings = []

        if pricing:
            cost_low = pricing.calculate_cost(input_tokens, output_low)
            cost_likely = pricing.calculate_cost(input_tokens, output_likely)
            cost_high = pricing.calculate_cost(input_tokens, output_high)
        else:
            warnings.append(f"No pricing found for model: {model}")
            cost_low = self.pricing_manager.calculate_cost(model, input_tokens, output_low)
            cost_likely = self.pricing_manager.calculate_cost(model, input_tokens, output_likely)
            cost_high = self.pricing_manager.calculate_cost(model, input_tokens, output_high)

        return CostEstimate(
            id=estimate_id,
            timestamp=timestamp,
            pipeline_name=None,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens_low=output_low,
            estimated_output_tokens_likely=output_likely,
            estimated_output_tokens_high=output_high,
            estimated_cost_low=cost_low,
            estimated_cost_likely=cost_likely,
            estimated_cost_high=cost_high,
            node_estimates=[],
            models_used=[model],
            estimation_method="input_ratio",
            confidence=0.6,
            warnings=warnings,
        )

    def estimate_pipeline(
        self,
        pipeline: "BasePipeline",
        input_text: str,
    ) -> CostEstimate:
        """
        Estimate cost for a complete pipeline run.

        Args:
            pipeline: Pipeline to estimate
            input_text: Input text for the pipeline

        Returns:
            CostEstimate with breakdown by node
        """
        estimate_id = f"est_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.utcnow().isoformat()

        node_estimates: List[NodeEstimate] = []
        total_input = 0
        total_output_low = 0
        total_output_likely = 0
        total_output_high = 0
        total_cost_low = 0.0
        total_cost_likely = 0.0
        total_cost_high = 0.0
        models_used: List[str] = []
        warnings: List[str] = []

        # Get node execution order
        try:
            node_order = pipeline.get_node_order()
        except Exception:
            node_order = list(pipeline.nodes.keys())

        # Track estimated outputs for chained nodes
        estimated_outputs: Dict[str, int] = {}

        for node_name in node_order:
            config = pipeline.nodes.get(node_name)
            if not config:
                continue

            # Skip non-LLM nodes
            if config.node_type.value != "llm":
                continue

            model = config.llm
            if model not in models_used:
                models_used.append(model)

            # Determine input for this node
            if config.input_key == "input":
                node_input_text = input_text
            elif config.input_key in estimated_outputs:
                # Use estimated output tokens from previous node
                node_input_text = "x" * (estimated_outputs[config.input_key] * 4)
            else:
                node_input_text = input_text

            # Count input tokens (including system prompt and template)
            full_input = node_input_text
            if config.input_template:
                try:
                    full_input = config.input_template.format(
                        input=node_input_text, **{"input": node_input_text}
                    )
                except Exception:
                    pass

            input_tokens = self.token_counter.count_with_system(
                full_input, config.system_prompt, model
            )

            # Estimate output tokens
            output_low, output_likely, output_high = self._estimate_output_tokens(
                input_tokens, model, config.max_tokens
            )

            # Store for chained nodes
            estimated_outputs[config.output_key] = output_likely

            # Calculate node costs
            pricing = self.pricing_manager.get_pricing(model)
            if pricing:
                cost_low = pricing.calculate_cost(input_tokens, output_low)
                cost_likely = pricing.calculate_cost(input_tokens, output_likely)
                cost_high = pricing.calculate_cost(input_tokens, output_high)
            else:
                if f"No pricing for {model}" not in warnings:
                    warnings.append(f"No pricing found for model: {model}")
                cost_low = self.pricing_manager.calculate_cost(model, input_tokens, output_low)
                cost_likely = self.pricing_manager.calculate_cost(model, input_tokens, output_likely)
                cost_high = self.pricing_manager.calculate_cost(model, input_tokens, output_high)

            # Get provider from pricing or infer
            provider = pricing.provider if pricing else self._infer_provider(model)

            node_estimate = NodeEstimate(
                node_name=node_name,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens_low=output_low,
                output_tokens_likely=output_likely,
                output_tokens_high=output_high,
                cost_low=cost_low,
                cost_likely=cost_likely,
                cost_high=cost_high,
            )
            node_estimates.append(node_estimate)

            # Accumulate totals
            total_input += input_tokens
            total_output_low += output_low
            total_output_likely += output_likely
            total_output_high += output_high
            total_cost_low += cost_low
            total_cost_likely += cost_likely
            total_cost_high += cost_high

        return CostEstimate(
            id=estimate_id,
            timestamp=timestamp,
            pipeline_name=pipeline.name,
            estimated_input_tokens=total_input,
            estimated_output_tokens_low=total_output_low,
            estimated_output_tokens_likely=total_output_likely,
            estimated_output_tokens_high=total_output_high,
            estimated_cost_low=round(total_cost_low, 6),
            estimated_cost_likely=round(total_cost_likely, 6),
            estimated_cost_high=round(total_cost_high, 6),
            node_estimates=node_estimates,
            models_used=models_used,
            estimation_method="input_ratio",
            confidence=0.5 if len(node_estimates) > 1 else 0.6,
            warnings=warnings,
        )

    def _estimate_output_tokens(
        self,
        input_tokens: int,
        model: str,
        max_tokens: int = 4096,
    ) -> Tuple[int, int, int]:
        """
        Estimate output tokens based on input tokens and model.

        Args:
            input_tokens: Number of input tokens
            model: Model name
            max_tokens: Maximum output tokens allowed

        Returns:
            Tuple of (low, likely, high) token estimates
        """
        # Get model-specific ratios
        ratios = self._get_output_ratios(model)

        # Calculate estimates
        low = int(input_tokens * ratios["low"])
        likely = int(input_tokens * ratios["likely"])
        high = int(input_tokens * ratios["high"])

        # Clamp to max_tokens
        low = min(low, max_tokens)
        likely = min(likely, max_tokens)
        high = min(high, max_tokens)

        # Ensure minimum of 10 tokens
        low = max(10, low)
        likely = max(20, likely)
        high = max(50, high)

        return low, likely, high

    def _get_output_ratios(self, model: str) -> Dict[str, float]:
        """
        Get output/input ratios for a model.

        Args:
            model: Model name

        Returns:
            Dict with 'low', 'likely', 'high' ratio values
        """
        # Check historical ratios first
        if model in self._historical_ratios:
            return self._historical_ratios[model]

        # Map to default ratios
        model_lower = model.lower()

        if "opus" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["claude-opus"]
        elif "claude" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["claude"]
        elif "gpt-4o" in model_lower or "gpt4o" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["gpt4o"]
        elif "gpt-4" in model_lower or "gpt4" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["gpt4"]
        elif "o1" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["o1"]
        elif "o3" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["o3-mini"]
        elif "gemini" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["gemini"]
        elif "grok" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["grok"]
        elif "perplexity" in model_lower or "sonar" in model_lower:
            return DEFAULT_OUTPUT_RATIOS["perplexity"]

        return DEFAULT_OUTPUT_RATIOS["default"]

    def _infer_provider(self, model: str) -> str:
        """Infer provider from model name."""
        model_lower = model.lower()

        if "claude" in model_lower:
            return "anthropic"
        elif any(x in model_lower for x in ["gpt", "o1", "o3"]):
            return "openai"
        elif "gemini" in model_lower:
            return "google"
        elif "grok" in model_lower:
            return "xai"
        elif "sonar" in model_lower:
            return "perplexity"

        return "unknown"

    def update_historical_ratio(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """
        Update historical ratios based on actual usage.

        Args:
            model: Model name
            input_tokens: Actual input tokens
            output_tokens: Actual output tokens
        """
        if input_tokens == 0:
            return

        actual_ratio = output_tokens / input_tokens

        if model not in self._historical_ratios:
            # Initialize with defaults and adjust
            base_ratios = self._get_output_ratios(model)
            self._historical_ratios[model] = {
                "low": min(actual_ratio * 0.5, base_ratios["low"]),
                "likely": actual_ratio,
                "high": max(actual_ratio * 2.0, base_ratios["high"]),
            }
        else:
            # Exponential moving average
            alpha = 0.3
            current = self._historical_ratios[model]
            current["likely"] = (alpha * actual_ratio) + ((1 - alpha) * current["likely"])
            current["low"] = min(current["likely"] * 0.5, current["low"])
            current["high"] = max(current["likely"] * 2.0, current["high"])


# Singleton instance
_cost_estimator: Optional[CostEstimator] = None


def get_cost_estimator() -> CostEstimator:
    """Get or create CostEstimator singleton."""
    global _cost_estimator
    if _cost_estimator is None:
        _cost_estimator = CostEstimator()
    return _cost_estimator
