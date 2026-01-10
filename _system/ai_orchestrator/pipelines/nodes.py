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

    # Metadata
    description: str = ""

    def __post_init__(self):
        if self.output_key is None:
            self.output_key = self.name

        if self.node_type == NodeType.LLM and self.llm is None:
            raise ValueError(f"Node '{self.name}': LLM nodes require 'llm' to be specified")

        if self.node_type == NodeType.CONDITIONAL and self.condition_fn is None:
            raise ValueError(f"Node '{self.name}': Conditional nodes require 'condition_fn'")

        if self.node_type == NodeType.TRANSFORM and self.transform_fn is None:
            raise ValueError(f"Node '{self.name}': Transform nodes require 'transform_fn'")

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

        # Apply template if provided
        if config.input_template:
            prompt = config.input_template.format(
                input=input_text,
                **{k: v.get("content", "") for k, v in state.get("outputs", {}).items()},
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

            return {
                **state,
                "outputs": outputs,
                "current_node": config.name,
                "final_output": response.content,
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


def create_transform_node(config: NodeConfig) -> Callable[[PipelineState], PipelineState]:
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
