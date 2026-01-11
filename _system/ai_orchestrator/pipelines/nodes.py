"""
Pipeline node definitions.
Provides node types for LangGraph pipelines with user-defined LLM selection.
"""

import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .state import PipelineState, NodeOutput


class NodeType(Enum):
    """Types of nodes available in pipelines."""

    LLM = "llm"  # Calls an LLM
    CONDITIONAL = "conditional"  # Routes based on condition
    TRANSFORM = "transform"  # Pure Python transformation
    AGGREGATE = "aggregate"  # Combines multiple outputs
    BROWSER = "browser"  # Browser automation with Playwright


@dataclass
class NodeConfig:
    """
    Configuration for a pipeline node.
    User specifies the LLM to use for each node.
    """

    name: str
    node_type: NodeType = NodeType.LLM

    # LLM configuration (required for LLM nodes)
    llm: Optional[str] = None  # e.g., "claude", "gpt4o", "gemini"
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096

    # Input/Output configuration
    input_key: str = "input"  # Key in state to read from
    output_key: Optional[str] = None  # Defaults to node name
    input_template: Optional[str] = None  # Template for formatting input

    # For conditional nodes
    condition_fn: Optional[Callable[[PipelineState], str]] = None
    routes: Dict[str, str] = field(default_factory=dict)

    # For transform nodes
    transform_fn: Optional[Callable[[PipelineState], str]] = None

    # Memory integration
    use_memory: bool = False  # Whether to inject context from memory
    memory_context_k: int = 3  # Number of past results to inject
    memory_min_score: float = 0.5  # Minimum similarity score for context

    # Browser automation configuration (for BROWSER nodes)
    browser_actions: List[Dict[str, Any]] = field(default_factory=list)
    browser_headless: bool = True
    browser_dry_run: bool = True  # Safety: simulate submit/destructive actions
    browser_site_credentials: Optional[str] = None
    browser_take_screenshots: bool = True

    # Metadata
    description: str = ""

    def __post_init__(self):
        if self.output_key is None:
            self.output_key = self.name

        if self.node_type == NodeType.LLM and self.llm is None:
            raise ValueError(
                f"Node '{self.name}': LLM nodes require 'llm' to be specified"
            )

        if self.node_type == NodeType.CONDITIONAL and self.condition_fn is None:
            raise ValueError(
                f"Node '{self.name}': Conditional nodes require 'condition_fn'"
            )

        if self.node_type == NodeType.TRANSFORM and self.transform_fn is None:
            raise ValueError(
                f"Node '{self.name}': Transform nodes require 'transform_fn'"
            )

        if self.node_type == NodeType.BROWSER and not self.browser_actions:
            raise ValueError(
                f"Node '{self.name}': Browser nodes require 'browser_actions'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excludes functions)."""
        return {
            "name": self.name,
            "node_type": self.node_type.value,
            "llm": self.llm,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "input_template": self.input_template,
            "routes": self.routes,
            "use_memory": self.use_memory,
            "memory_context_k": self.memory_context_k,
            "memory_min_score": self.memory_min_score,
            "browser_actions": self.browser_actions,
            "browser_headless": self.browser_headless,
            "browser_dry_run": self.browser_dry_run,
            "browser_site_credentials": self.browser_site_credentials,
            "browser_take_screenshots": self.browser_take_screenshots,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeConfig":
        """Deserialize from dictionary."""
        data = data.copy()
        data["node_type"] = NodeType(data.get("node_type", "llm"))
        # Functions must be set separately after deserialization
        data.pop("condition_fn", None)
        data.pop("transform_fn", None)
        return cls(**data)


def create_llm_node(config: NodeConfig) -> Callable[[PipelineState], PipelineState]:
    """
    Create a LangGraph node function that calls an LLM.

    Args:
        config: Node configuration with LLM selection

    Returns:
        Function that takes state and returns updated state
    """
    from ..core.llm_clients import get_clients

    def node_fn(state: PipelineState) -> PipelineState:
        """Execute LLM call and update state."""
        clients = get_clients()

        # Get input text
        if config.input_key == "input":
            input_text = state["input"]
        elif config.input_key in state.get("outputs", {}):
            input_text = state["outputs"][config.input_key]["content"]
        else:
            input_text = state.get("custom", {}).get(config.input_key, state["input"])

        # Inject memory context if enabled
        context_sources = []
        if config.use_memory:
            try:
                from ..memory import get_context_injector, ContextConfig

                injector = get_context_injector()
                context_config = ContextConfig(
                    enabled=True,
                    max_results=config.memory_context_k,
                    min_score=config.memory_min_score,
                )
                injection = injector.inject_for_input(
                    input_text=input_text,
                    project=state.get("custom", {}).get("project"),
                    node_name=config.name,
                    config=context_config,
                )
                if injection.context_added:
                    input_text = injection.enhanced_input
                    context_sources = [
                        {"id": s["id"], "score": s["score"]} for s in injection.sources
                    ]
            except Exception:
                pass  # Continue without memory if it fails

        # Apply template if provided
        if config.input_template:
            prompt = config.input_template.format(
                input=input_text,
                **{
                    k: v.get("content", "") for k, v in state.get("outputs", {}).items()
                },
            )
        else:
            prompt = input_text

        # Call LLM
        start_time = time.time()
        try:
            response = clients.call(
                prompt=prompt,
                model=config.llm,
                system=config.system_prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Create output
            output = NodeOutput(
                content=response.content,
                model=response.model,
                provider=response.provider,
                latency_ms=latency_ms,
                tokens_used=response.usage,
            )

            # Update state
            outputs = dict(state.get("outputs", {}))
            outputs[config.output_key] = output

            # Include memory context sources if used
            custom = dict(state.get("custom", {}))
            if context_sources:
                custom["memory_context_sources"] = context_sources

            return {
                **state,
                "outputs": outputs,
                "current_node": config.name,
                "final_output": response.content,
                "custom": custom,
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            errors = list(state.get("errors", []))
            errors.append(
                {
                    "node": config.name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return {
                **state,
                "current_node": config.name,
                "errors": errors,
            }

    return node_fn


def create_conditional_node(
    config: NodeConfig,
) -> Callable[[PipelineState], PipelineState]:
    """
    Create a conditional routing node.

    Args:
        config: Node configuration with condition function and routes

    Returns:
        Function that evaluates condition and sets route in state
    """

    def node_fn(state: PipelineState) -> PipelineState:
        """Evaluate condition and set route."""
        try:
            route = config.condition_fn(state)
            return {
                **state,
                "current_node": config.name,
                "route": route,
                "route_reason": f"Condition evaluated to: {route}",
            }
        except Exception as e:
            errors = list(state.get("errors", []))
            errors.append(
                {
                    "node": config.name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return {
                **state,
                "current_node": config.name,
                "errors": errors,
                "route": list(config.routes.values())[0] if config.routes else None,
            }

    return node_fn


def create_transform_node(
    config: NodeConfig,
) -> Callable[[PipelineState], PipelineState]:
    """
    Create a pure Python transformation node.

    Args:
        config: Node configuration with transform function

    Returns:
        Function that transforms state
    """

    def node_fn(state: PipelineState) -> PipelineState:
        """Apply transformation."""
        start_time = time.time()
        try:
            result = config.transform_fn(state)
            latency_ms = int((time.time() - start_time) * 1000)

            output = NodeOutput(
                content=result,
                model="transform",
                provider="local",
                latency_ms=latency_ms,
                tokens_used=None,
            )

            outputs = dict(state.get("outputs", {}))
            outputs[config.output_key] = output

            return {
                **state,
                "outputs": outputs,
                "current_node": config.name,
                "final_output": result,
            }
        except Exception as e:
            errors = list(state.get("errors", []))
            errors.append(
                {
                    "node": config.name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return {
                **state,
                "current_node": config.name,
                "errors": errors,
            }

    return node_fn


def create_aggregate_node(
    name: str, input_keys: List[str], aggregate_fn: Callable[[Dict[str, str]], str]
) -> Callable[[PipelineState], PipelineState]:
    """
    Create a node that aggregates multiple outputs.

    Args:
        name: Node name
        input_keys: Keys of outputs to aggregate
        aggregate_fn: Function that combines outputs

    Returns:
        Function that aggregates outputs
    """

    def node_fn(state: PipelineState) -> PipelineState:
        """Aggregate outputs."""
        start_time = time.time()
        try:
            # Gather outputs
            inputs = {}
            for key in input_keys:
                if key in state.get("outputs", {}):
                    inputs[key] = state["outputs"][key]["content"]

            result = aggregate_fn(inputs)
            latency_ms = int((time.time() - start_time) * 1000)

            output = NodeOutput(
                content=result,
                model="aggregate",
                provider="local",
                latency_ms=latency_ms,
                tokens_used=None,
            )

            outputs = dict(state.get("outputs", {}))
            outputs[name] = output

            return {
                **state,
                "outputs": outputs,
                "current_node": name,
                "final_output": result,
            }
        except Exception as e:
            errors = list(state.get("errors", []))
            errors.append(
                {
                    "node": name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return {
                **state,
                "current_node": name,
                "errors": errors,
            }

    return node_fn


def create_browser_node(
    config: NodeConfig,
) -> Callable[[PipelineState], PipelineState]:
    """
    Create a browser automation node.

    Args:
        config: Node configuration with browser actions

    Returns:
        Function that executes browser actions and updates state
    """

    def node_fn(state: PipelineState) -> PipelineState:
        """Execute browser actions."""
        from ..browser import (
            get_playwright_client,
            BrowserAction,
            ActionCategory,
            ApprovalStatus,
        )

        start_time = time.time()

        try:
            client = get_playwright_client(
                headless=config.browser_headless,
                dry_run=config.browser_dry_run,
            )

            # Start session if not active
            if not client.is_active:
                client.start_session(
                    site_credentials=config.browser_site_credentials
                )

            results = []
            extracted_data = []
            screenshots = []

            for action_def in config.browser_actions:
                # Create action from definition
                action = BrowserAction.from_dict(action_def)

                # Auto-approve read-only actions
                if action.category == ActionCategory.READ_ONLY:
                    action.approval_status = ApprovalStatus.AUTO_APPROVED

                result = client.execute_action(
                    action,
                    take_screenshots=config.browser_take_screenshots,
                )
                results.append(result.to_dict())

                if result.extracted_data:
                    extracted_data.append(result.extracted_data)
                if result.screenshot_path:
                    screenshots.append(result.screenshot_path)

            latency_ms = int((time.time() - start_time) * 1000)

            # Compile output
            content = "\n\n".join(extracted_data) if extracted_data else "Browser actions completed"

            output = NodeOutput(
                content=content,
                model="browser",
                provider="playwright",
                latency_ms=latency_ms,
                tokens_used=None,
            )

            outputs = dict(state.get("outputs", {}))
            outputs[config.output_key] = output

            # Store browser-specific data in custom
            custom = dict(state.get("custom", {}))
            custom["browser_results"] = results
            custom["browser_screenshots"] = screenshots

            return {
                **state,
                "outputs": outputs,
                "current_node": config.name,
                "final_output": content,
                "custom": custom,
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            errors = list(state.get("errors", []))
            errors.append(
                {
                    "node": config.name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            return {
                **state,
                "current_node": config.name,
                "errors": errors,
            }

    return node_fn


def create_node(config: NodeConfig) -> Callable[[PipelineState], PipelineState]:
    """
    Factory function to create appropriate node based on type.

    Args:
        config: Node configuration

    Returns:
        Node function for LangGraph
    """
    if config.node_type == NodeType.LLM:
        return create_llm_node(config)
    elif config.node_type == NodeType.CONDITIONAL:
        return create_conditional_node(config)
    elif config.node_type == NodeType.TRANSFORM:
        return create_transform_node(config)
    elif config.node_type == NodeType.BROWSER:
        return create_browser_node(config)
    else:
        raise ValueError(f"Unknown node type: {config.node_type}")


# Pre-built condition functions for common routing patterns


def route_by_length(threshold: int = 500) -> Callable[[PipelineState], str]:
    """Route based on output length."""

    def condition(state: PipelineState) -> str:
        last_output = state.get("final_output", "")
        return "long" if len(last_output) > threshold else "short"

    return condition


def route_by_keyword(
    keywords: Dict[str, List[str]], default: str = "other"
) -> Callable[[PipelineState], str]:
    """Route based on presence of keywords in output."""

    def condition(state: PipelineState) -> str:
        last_output = state.get("final_output", "").lower()
        for route, words in keywords.items():
            if any(word.lower() in last_output for word in words):
                return route
        return default

    return condition


def route_by_confidence(threshold: float = 0.7) -> Callable[[PipelineState], str]:
    """Route based on confidence score in output (expects JSON with 'confidence' key)."""
    import json

    def condition(state: PipelineState) -> str:
        last_output = state.get("final_output", "")
        try:
            data = json.loads(last_output)
            confidence = data.get("confidence", 0)
            return "high" if confidence >= threshold else "low"
        except (json.JSONDecodeError, TypeError):
            return "unknown"

    return condition
