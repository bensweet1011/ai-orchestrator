"""
Dynamic pricing configuration management for Cost Intelligence.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import SYSTEM_DIR
from .state import ModelPricing, PricingTier

# Default pricing config path
DEFAULT_CONFIG_PATH = SYSTEM_DIR / "pricing_config.json"


class PricingManager:
    """Manages model pricing configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize pricing manager.

        Args:
            config_path: Path to pricing config JSON file
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._pricing: Dict[str, ModelPricing] = {}
        self._config_version: str = "1.0"
        self._last_updated: str = ""
        self._load_config()

    def _load_config(self):
        """Load pricing from JSON config."""
        if not self.config_path.exists():
            self._create_default_config()
            return

        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)

            self._config_version = data.get("version", "1.0")
            self._last_updated = data.get("last_updated", "")

            for model_id, model_data in data.get("models", {}).items():
                self._pricing[model_id] = ModelPricing(
                    model_id=model_id,
                    provider=model_data.get("provider", "unknown"),
                    input_cost_per_1k=model_data.get("input_cost_per_1k", 0.0),
                    output_cost_per_1k=model_data.get("output_cost_per_1k", 0.0),
                    effective_date=self._last_updated,
                    cached_input_cost_per_1k=model_data.get("cached_input_cost_per_1k"),
                    batch_discount=model_data.get("batch_discount", 0.0),
                    notes=model_data.get("notes", ""),
                )
        except Exception as e:
            print(f"Warning: Failed to load pricing config: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """Create default pricing configuration."""
        default_pricing = {
            "claude-sonnet-4-20250514": ModelPricing(
                model_id="claude-sonnet-4-20250514",
                provider="anthropic",
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
                cached_input_cost_per_1k=0.0003,
            ),
            "claude-opus-4-20250514": ModelPricing(
                model_id="claude-opus-4-20250514",
                provider="anthropic",
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.075,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
                cached_input_cost_per_1k=0.00175,
            ),
            "gpt-4o": ModelPricing(
                model_id="gpt-4o",
                provider="openai",
                input_cost_per_1k=0.005,
                output_cost_per_1k=0.015,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
                batch_discount=0.5,
            ),
            "gpt-4-turbo": ModelPricing(
                model_id="gpt-4-turbo",
                provider="openai",
                input_cost_per_1k=0.01,
                output_cost_per_1k=0.03,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
            ),
            "o1-preview": ModelPricing(
                model_id="o1-preview",
                provider="openai",
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.06,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
            ),
            "o3-mini": ModelPricing(
                model_id="o3-mini",
                provider="openai",
                input_cost_per_1k=0.0011,
                output_cost_per_1k=0.0044,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
            ),
            "gemini-1.5-pro": ModelPricing(
                model_id="gemini-1.5-pro",
                provider="google",
                input_cost_per_1k=0.00125,
                output_cost_per_1k=0.005,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
            ),
            "grok-2-latest": ModelPricing(
                model_id="grok-2-latest",
                provider="xai",
                input_cost_per_1k=0.002,
                output_cost_per_1k=0.01,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
            ),
            "llama-3.1-sonar-large-128k-online": ModelPricing(
                model_id="llama-3.1-sonar-large-128k-online",
                provider="perplexity",
                input_cost_per_1k=0.001,
                output_cost_per_1k=0.001,
                effective_date=datetime.utcnow().strftime("%Y-%m-%d"),
            ),
        }
        self._pricing = default_pricing
        self._save_config()

    def _save_config(self):
        """Save pricing to JSON config."""
        data = {
            "version": self._config_version,
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d"),
            "currency": "USD",
            "models": {},
        }

        for model_id, pricing in self._pricing.items():
            model_data = {
                "provider": pricing.provider,
                "input_cost_per_1k": pricing.input_cost_per_1k,
                "output_cost_per_1k": pricing.output_cost_per_1k,
            }
            if pricing.cached_input_cost_per_1k:
                model_data["cached_input_cost_per_1k"] = pricing.cached_input_cost_per_1k
            if pricing.batch_discount > 0:
                model_data["batch_discount"] = pricing.batch_discount
            if pricing.notes:
                model_data["notes"] = pricing.notes

            data["models"][model_id] = model_data

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save pricing config: {e}")

    def get_pricing(
        self, model: str, provider: Optional[str] = None
    ) -> Optional[ModelPricing]:
        """
        Get pricing for a model.

        Args:
            model: Model name or ID (e.g., "claude", "gpt4o", or full ID)
            provider: Optional provider to help resolve ambiguous names

        Returns:
            ModelPricing or None if not found
        """
        # Direct lookup by full model ID
        if model in self._pricing:
            return self._pricing[model]

        # Map short names to full model IDs
        model_map = {
            "claude": "claude-sonnet-4-20250514",
            "claude-opus": "claude-opus-4-20250514",
            "gpt4o": "gpt-4o",
            "gpt4": "gpt-4-turbo",
            "o1": "o1-preview",
            "o3-mini": "o3-mini",
            "gemini": "gemini-1.5-pro",
            "grok": "grok-2-latest",
            "perplexity": "llama-3.1-sonar-large-128k-online",
        }

        mapped_id = model_map.get(model)
        if mapped_id and mapped_id in self._pricing:
            return self._pricing[mapped_id]

        # Search by provider if specified
        if provider:
            for pricing in self._pricing.values():
                if pricing.provider == provider and model in pricing.model_id:
                    return pricing

        return None

    def update_pricing(self, model_id: str, pricing: ModelPricing):
        """
        Update pricing for a model.

        Args:
            model_id: Model identifier
            pricing: New pricing information
        """
        self._pricing[model_id] = pricing
        self._save_config()

    def get_all_pricing(self) -> Dict[str, ModelPricing]:
        """Get all pricing configurations."""
        return self._pricing.copy()

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tier: PricingTier = PricingTier.STANDARD,
    ) -> float:
        """
        Calculate cost for a model call.

        Args:
            model: Model name or ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tier: Pricing tier

        Returns:
            Cost in USD
        """
        pricing = self.get_pricing(model)
        if not pricing:
            # Return estimate based on average pricing
            avg_input = 0.005
            avg_output = 0.015
            return round(
                (input_tokens / 1000 * avg_input)
                + (output_tokens / 1000 * avg_output),
                6,
            )

        return pricing.calculate_cost(input_tokens, output_tokens, tier)

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get model information including pricing.

        Args:
            model: Model name or ID

        Returns:
            Dictionary with model info
        """
        pricing = self.get_pricing(model)
        if not pricing:
            return {
                "model": model,
                "found": False,
                "input_cost_per_1k": 0.005,
                "output_cost_per_1k": 0.015,
            }

        return {
            "model": pricing.model_id,
            "provider": pricing.provider,
            "found": True,
            "input_cost_per_1k": pricing.input_cost_per_1k,
            "output_cost_per_1k": pricing.output_cost_per_1k,
            "cached_input_cost_per_1k": pricing.cached_input_cost_per_1k,
            "batch_discount": pricing.batch_discount,
        }


# Singleton instance
_pricing_manager: Optional[PricingManager] = None


def get_pricing_manager() -> PricingManager:
    """Get or create PricingManager singleton."""
    global _pricing_manager
    if _pricing_manager is None:
        _pricing_manager = PricingManager()
    return _pricing_manager
