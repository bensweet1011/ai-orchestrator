"""
Context injection system for pipelines.
Automatically retrieves and injects relevant memory into pipeline inputs.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .state import (
    ContextConfig,
    MemorySearchResult,
    MemoryType,
)
from .store import MemoryStore, get_memory_store


@dataclass
class InjectedContext:
    """Result of context injection."""

    original_input: str
    enhanced_input: str
    context_added: bool
    sources: List[MemorySearchResult]
    source_count: int


class ContextInjector:
    """
    Manages automatic context injection from memory into pipeline inputs.

    Features:
    - Semantic search for relevant past outputs
    - Configurable injection per node
    - Recency weighting for more recent results
    - Project-scoped retrieval
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        default_config: Optional[ContextConfig] = None,
    ):
        """
        Initialize context injector.

        Args:
            store: MemoryStore instance (uses singleton if not provided)
            default_config: Default context configuration
        """
        self.store = store or get_memory_store()
        self.default_config = default_config or ContextConfig()

        # Per-node configuration overrides
        self._node_configs: Dict[str, ContextConfig] = {}

    def configure_node(self, node_name: str, config: ContextConfig):
        """
        Set context injection configuration for a specific node.

        Args:
            node_name: Name of the node
            config: Context configuration for this node
        """
        self._node_configs[node_name] = config

    def get_node_config(self, node_name: str) -> ContextConfig:
        """Get configuration for a node (falls back to default)."""
        return self._node_configs.get(node_name, self.default_config)

    def inject_for_input(
        self,
        input_text: str,
        project: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        node_name: Optional[str] = None,
        config: Optional[ContextConfig] = None,
    ) -> InjectedContext:
        """
        Inject relevant context into input text.

        Args:
            input_text: Original input text
            project: Project to search for context
            pipeline_name: Filter by specific pipeline (optional)
            node_name: Node name for config lookup
            config: Override context configuration

        Returns:
            InjectedContext with enhanced input
        """
        # Get configuration
        if config is None:
            config = (
                self.get_node_config(node_name) if node_name else self.default_config
            )

        # Check if injection is enabled
        if not config.enabled:
            return InjectedContext(
                original_input=input_text,
                enhanced_input=input_text,
                context_added=False,
                sources=[],
                source_count=0,
            )

        # Search for relevant past outputs
        results = self.store.search_pipeline_outputs(
            query=input_text,
            project=project or self.store.current_project,
            pipeline_name=pipeline_name,
            success_only=True,
            limit=config.max_results * 2,  # Fetch more for filtering
            min_score=config.min_score,
        )

        if not results:
            return InjectedContext(
                original_input=input_text,
                enhanced_input=input_text,
                context_added=False,
                sources=[],
                source_count=0,
            )

        # Apply recency weighting if configured
        if config.recency_weight > 0:
            results = self._apply_recency_weight(results, config.recency_weight)

        # Take top results
        top_results = results[: config.max_results]

        # Build context prompt
        context_text = self._build_context_prompt(
            top_results, config.include_node_outputs
        )

        # Create enhanced input
        enhanced_input = self._combine_input_with_context(input_text, context_text)

        return InjectedContext(
            original_input=input_text,
            enhanced_input=enhanced_input,
            context_added=True,
            sources=top_results,
            source_count=len(top_results),
        )

    def _apply_recency_weight(
        self,
        results: List[MemorySearchResult],
        recency_weight: float,
    ) -> List[MemorySearchResult]:
        """
        Apply recency weighting to results.
        More recent results get boosted scores.
        """
        if not results:
            return results

        now = datetime.utcnow()

        # Calculate age-based boost for each result
        weighted_results = []
        for result in results:
            try:
                timestamp = datetime.fromisoformat(
                    result["timestamp"].replace("Z", "+00:00")
                )
                age_hours = (
                    now - timestamp.replace(tzinfo=None)
                ).total_seconds() / 3600
                # Decay factor: newer = higher boost (max 1.0 at 0 hours, decays over time)
                recency_boost = 1.0 / (1.0 + age_hours / 24)  # Half-life of ~24 hours
                adjusted_score = (
                    result["score"] * (1 - recency_weight)
                    + recency_boost * recency_weight
                )
            except (ValueError, TypeError):
                adjusted_score = result["score"]

            # Create new result with adjusted score (keeping original structure)
            weighted_result = MemorySearchResult(
                id=result["id"],
                score=adjusted_score,
                memory_type=result["memory_type"],
                timestamp=result["timestamp"],
                project=result["project"],
                content_preview=result["content_preview"],
                metadata=result["metadata"],
            )
            weighted_results.append(weighted_result)

        # Sort by adjusted score
        weighted_results.sort(key=lambda x: x["score"], reverse=True)
        return weighted_results

    def _build_context_prompt(
        self,
        results: List[MemorySearchResult],
        include_node_outputs: bool = True,
    ) -> str:
        """
        Build formatted context prompt from search results.

        Args:
            results: Search results to format
            include_node_outputs: Whether to include detailed node outputs

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        lines = [
            "## Relevant Past Context",
            "",
            "The following are relevant past interactions that may provide useful context:",
            "",
        ]

        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            timestamp = result["timestamp"][:10] if result["timestamp"] else "Unknown"
            pipeline = meta.get("pipeline_name", "Unknown")
            score = result["score"]

            lines.append(f"### Context {i} (similarity: {score:.2f})")
            lines.append(f"- **Date**: {timestamp}")
            lines.append(f"- **Pipeline**: {pipeline}")

            if meta.get("input_preview"):
                lines.append(f"- **Input**: {meta['input_preview'][:200]}...")

            lines.append(f"- **Output**: {result['content_preview']}")

            if include_node_outputs and meta.get("nodes"):
                lines.append(f"- **Nodes used**: {meta['nodes']}")

            lines.append("")

        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _combine_input_with_context(self, input_text: str, context_text: str) -> str:
        """
        Combine original input with injected context.

        Args:
            input_text: Original input
            context_text: Context to inject

        Returns:
            Combined text with context prepended
        """
        if not context_text:
            return input_text

        return f"""{context_text}
## Current Request

{input_text}"""

    def get_context_for_debugging(
        self,
        error_message: str,
        node_name: str,
        project: Optional[str] = None,
        limit: int = 3,
    ) -> str:
        """
        Get relevant past context for debugging a failed node.
        Searches for successful runs of the same node.

        Args:
            error_message: The error that occurred
            node_name: Node that failed
            project: Project context
            limit: Max results

        Returns:
            Formatted context of past successful runs
        """
        # Search for past successful outputs mentioning this node
        query = f"{node_name} successful execution"

        results = self.store.search(
            query=query,
            project=project or self.store.current_project,
            memory_types=[MemoryType.PIPELINE_RUN],
            limit=limit,
        )

        if not results:
            return ""

        lines = ["## Past Successful Runs (for reference)", ""]

        for result in results:
            lines.append(
                f"- {result['timestamp'][:10]}: {result['content_preview'][:150]}..."
            )

        return "\n".join(lines)


class ContextAwareNode:
    """
    Wrapper that adds context injection to any node execution.
    """

    def __init__(
        self,
        node_fn,
        injector: Optional[ContextInjector] = None,
        config: Optional[ContextConfig] = None,
    ):
        """
        Initialize context-aware node wrapper.

        Args:
            node_fn: Original node function
            injector: Context injector instance
            config: Context configuration
        """
        self.node_fn = node_fn
        self.injector = injector or ContextInjector()
        self.config = config or ContextConfig()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute node with context injection.

        Args:
            state: Pipeline state

        Returns:
            Updated state
        """
        # Get input text from state
        input_key = state.get("current_input_key", "input")
        original_input = state.get(input_key, "")

        if not original_input:
            # No input to enhance
            return self.node_fn(state)

        # Inject context
        injection_result = self.injector.inject_for_input(
            input_text=original_input,
            project=state.get("project"),
            config=self.config,
        )

        if injection_result.context_added:
            # Create modified state with enhanced input
            enhanced_state = dict(state)
            enhanced_state[input_key] = injection_result.enhanced_input
            enhanced_state["_context_sources"] = [
                {"id": s["id"], "score": s["score"]} for s in injection_result.sources
            ]

            # Execute with enhanced input
            result = self.node_fn(enhanced_state)

            # Preserve context metadata in output
            if "_context_sources" not in result:
                result["_context_sources"] = enhanced_state["_context_sources"]

            return result
        else:
            return self.node_fn(state)


# Singleton instance
_injector: Optional[ContextInjector] = None


def get_context_injector() -> ContextInjector:
    """Get or create ContextInjector singleton."""
    global _injector
    if _injector is None:
        _injector = ContextInjector()
    return _injector
