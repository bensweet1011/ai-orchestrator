"""
Base pipeline class for LangGraph integration.
Provides a high-level API for building and executing LLM pipelines.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import time

from langgraph.graph import StateGraph, END

from .state import PipelineState, PipelineResult, create_initial_state
from .nodes import NodeConfig, create_node


@dataclass
class EdgeConfig:
    """Configuration for an edge between nodes."""

    from_node: str
    to_node: str
    condition: Optional[Callable[[PipelineState], str]] = None
    condition_map: Optional[Dict[str, str]] = None  # route_value -> target_node


class BasePipeline:
    """
    Base class for building LangGraph pipelines with user-defined LLM selection.

    Example:
        pipeline = BasePipeline("my_pipeline")
        pipeline.add_node(NodeConfig(
            name="summarize",
            llm="gpt4o",
            system_prompt="Summarize the following text concisely."
        ))
        pipeline.add_node(NodeConfig(
            name="critique",
            llm="claude",
            input_key="summarize",
            system_prompt="Critique the following summary."
        ))
        pipeline.add_edge("summarize", "critique")
        pipeline.set_entry_point("summarize")
        pipeline.set_finish_point("critique")

        result = pipeline.run("Long text to process...")
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        log_to_memory: bool = True,
        project: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.pipeline_id = self._generate_id()

        # Node and edge storage
        self.nodes: Dict[str, NodeConfig] = {}
        self.edges: List[EdgeConfig] = []
        self.conditional_edges: List[EdgeConfig] = []

        # Entry and exit points
        self.entry_point: Optional[str] = None
        self.finish_points: List[str] = []

        # Compiled graph
        self._graph: Optional[StateGraph] = None
        self._compiled = False

        # Memory integration
        self.log_to_memory = log_to_memory
        self.project = project

    def _generate_id(self) -> str:
        """Generate unique pipeline ID."""
        ts = datetime.utcnow().isoformat()
        unique = hashlib.md5(f"{self.name}{ts}".encode()).hexdigest()[:8]
        return f"pipeline_{unique}"

    def _log_to_memory(self, result: PipelineResult, input_text: str) -> str:
        """Log pipeline result to memory store. Returns memory_id."""
        try:
            from ..memory import get_memory_store

            store = get_memory_store()
            return store.log_pipeline_run(
                result=result,
                pipeline_name=self.name,
                input_text=input_text,
                project=self.project,
            )
        except Exception as e:
            # Log error but don't break pipeline
            import sys

            print(f"Warning: Memory logging failed: {e}", file=sys.stderr)
            return ""

    def _estimate_cost(self, input_text: str):
        """Estimate cost before pipeline execution."""
        try:
            from ..cost import get_cost_estimator

            estimator = get_cost_estimator()
            return estimator.estimate_pipeline(self, input_text)
        except Exception as e:
            import sys

            print(f"Warning: Cost estimation failed: {e}", file=sys.stderr)
            return None

    def _track_cost(self, result: PipelineResult, pre_estimate, memory_id: str = None):
        """Track cost after pipeline execution."""
        try:
            from ..cost import get_cost_tracker

            tracker = get_cost_tracker(self.project or "default")
            cost_record = tracker.track_pipeline_run(
                result=result,
                pipeline_name=self.name,
                pre_estimate=pre_estimate,
                memory_id=memory_id,
            )
            return cost_record
        except Exception as e:
            import sys

            print(f"Warning: Cost tracking failed: {e}", file=sys.stderr)
            return None

    def _check_budget(self, estimated_cost: float) -> tuple:
        """Check if budget allows execution. Returns (allowed, warning_msg)."""
        try:
            from ..cost import get_budget_manager

            budget_mgr = get_budget_manager(self.project or "default")
            return budget_mgr.check_budget(estimated_cost)
        except Exception as e:
            import sys
            print(f"Warning: Budget check failed: {e}", file=sys.stderr)
            return True, None

    def add_node(self, config: NodeConfig) -> "BasePipeline":
        """
        Add a node to the pipeline.

        Args:
            config: Node configuration (must specify LLM for LLM nodes)

        Returns:
            Self for chaining
        """
        if config.name in self.nodes:
            raise ValueError(f"Node '{config.name}' already exists")

        self.nodes[config.name] = config
        self._compiled = False
        return self

    def add_edge(self, from_node: str, to_node: str) -> "BasePipeline":
        """
        Add a sequential edge between nodes.

        Args:
            from_node: Source node name
            to_node: Target node name

        Returns:
            Self for chaining
        """
        self.edges.append(EdgeConfig(from_node=from_node, to_node=to_node))
        self._compiled = False
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable[[PipelineState], str],
        routes: Dict[str, str],
    ) -> "BasePipeline":
        """
        Add a conditional edge that routes based on state.

        Args:
            from_node: Source node name
            condition: Function that takes state and returns route key
            routes: Mapping of route keys to target node names

        Returns:
            Self for chaining
        """
        self.conditional_edges.append(
            EdgeConfig(
                from_node=from_node,
                to_node="",  # Determined by condition
                condition=condition,
                condition_map=routes,
            )
        )
        self._compiled = False
        return self

    def set_entry_point(self, node_name: str) -> "BasePipeline":
        """Set the starting node for the pipeline."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
        self.entry_point = node_name
        self._compiled = False
        return self

    def set_finish_point(self, node_name: str) -> "BasePipeline":
        """Add a node as a finish point (can have multiple)."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")
        if node_name not in self.finish_points:
            self.finish_points.append(node_name)
        self._compiled = False
        return self

    def compile(self) -> "BasePipeline":
        """
        Compile the pipeline into a LangGraph StateGraph.

        Returns:
            Self for chaining
        """
        if not self.entry_point:
            raise ValueError("Entry point must be set before compiling")

        if not self.finish_points:
            raise ValueError("At least one finish point must be set before compiling")

        # Create StateGraph
        self._graph = StateGraph(PipelineState)

        # Add nodes
        for name, config in self.nodes.items():
            node_fn = create_node(config)
            self._graph.add_node(name, node_fn)

        # Set entry point
        self._graph.set_entry_point(self.entry_point)

        # Add edges
        for edge in self.edges:
            if edge.to_node in self.finish_points and edge.to_node == edge.from_node:
                # Self-loop to finish - just add edge to END
                continue
            self._graph.add_edge(edge.from_node, edge.to_node)

        # Add conditional edges
        for edge in self.conditional_edges:
            # Add END to routes if any route points to a finish point
            routes = dict(edge.condition_map)
            for key, target in list(routes.items()):
                if target == "__end__":
                    routes[key] = END

            self._graph.add_conditional_edges(edge.from_node, edge.condition, routes)

        # Add edges from finish points to END
        for finish_node in self.finish_points:
            # Check if this node already has outgoing edges
            has_outgoing = any(
                e.from_node == finish_node for e in self.edges + self.conditional_edges
            )
            if not has_outgoing:
                self._graph.add_edge(finish_node, END)

        self._compiled = True
        return self

    def run(
        self,
        input_text: str,
        config: Optional[Dict] = None,
        log_to_memory: Optional[bool] = None,
        track_cost: bool = True,
    ) -> PipelineResult:
        """
        Execute the pipeline on input text.

        Args:
            input_text: User input to process
            config: Optional LangGraph config
            log_to_memory: Override memory logging (uses pipeline default if None)
            track_cost: Whether to track cost (default True)

        Returns:
            PipelineResult with outputs from all nodes
        """
        if not self._compiled:
            self.compile()

        # Pre-run cost estimation
        cost_estimate = None
        if track_cost:
            cost_estimate = self._estimate_cost(input_text)

        # Create initial state with project context
        state = create_initial_state(input_text, self.pipeline_id)
        if self.project:
            state["custom"]["project"] = self.project

        # Compile and run
        app = self._graph.compile()
        start_time = time.time()

        try:
            final_state = app.invoke(state, config=config)
            total_latency = int((time.time() - start_time) * 1000)

            # Calculate total tokens
            total_tokens = {"input_tokens": 0, "output_tokens": 0}
            for output in final_state.get("outputs", {}).values():
                if output.get("tokens_used"):
                    total_tokens["input_tokens"] += output["tokens_used"].get(
                        "input_tokens", 0
                    )
                    total_tokens["output_tokens"] += output["tokens_used"].get(
                        "output_tokens", 0
                    )

            result = PipelineResult(
                success=len(final_state.get("errors", [])) == 0,
                final_output=final_state.get("final_output", ""),
                node_outputs=final_state.get("outputs", {}),
                total_latency_ms=total_latency,
                total_tokens=total_tokens,
                errors=final_state.get("errors", []),
                pipeline_id=self.pipeline_id,
                started_at=final_state.get("started_at", ""),
                completed_at=datetime.utcnow().isoformat(),
                cost_estimate=cost_estimate.to_dict() if cost_estimate else None,
            )

            # Log to memory if enabled
            should_log = (
                log_to_memory if log_to_memory is not None else self.log_to_memory
            )
            memory_id = ""
            if should_log and result.success:
                memory_id = self._log_to_memory(result, input_text)

            # Post-run cost tracking
            if track_cost and result.success:
                cost_record = self._track_cost(result, cost_estimate, memory_id)
                if cost_record:
                    result.total_cost = cost_record.total_cost
                    result.cost_record_id = cost_record.id

            return result

        except Exception as e:
            total_latency = int((time.time() - start_time) * 1000)
            return PipelineResult(
                success=False,
                final_output="",
                node_outputs={},
                total_latency_ms=total_latency,
                total_tokens={"input_tokens": 0, "output_tokens": 0},
                errors=[
                    {
                        "node": "__pipeline__",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ],
                pipeline_id=self.pipeline_id,
                started_at=state.get("started_at", ""),
                completed_at=datetime.utcnow().isoformat(),
            )

    async def arun(
        self, input_text: str, config: Optional[Dict] = None
    ) -> PipelineResult:
        """Async version of run."""
        if not self._compiled:
            self.compile()

        state = create_initial_state(input_text, self.pipeline_id)
        app = self._graph.compile()
        start_time = time.time()

        try:
            final_state = await app.ainvoke(state, config=config)
            total_latency = int((time.time() - start_time) * 1000)

            total_tokens = {"input_tokens": 0, "output_tokens": 0}
            for output in final_state.get("outputs", {}).values():
                if output.get("tokens_used"):
                    total_tokens["input_tokens"] += output["tokens_used"].get(
                        "input_tokens", 0
                    )
                    total_tokens["output_tokens"] += output["tokens_used"].get(
                        "output_tokens", 0
                    )

            return PipelineResult(
                success=len(final_state.get("errors", [])) == 0,
                final_output=final_state.get("final_output", ""),
                node_outputs=final_state.get("outputs", {}),
                total_latency_ms=total_latency,
                total_tokens=total_tokens,
                errors=final_state.get("errors", []),
                pipeline_id=self.pipeline_id,
                started_at=final_state.get("started_at", ""),
                completed_at=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            total_latency = int((time.time() - start_time) * 1000)
            return PipelineResult(
                success=False,
                final_output="",
                node_outputs={},
                total_latency_ms=total_latency,
                total_tokens={"input_tokens": 0, "output_tokens": 0},
                errors=[
                    {
                        "node": "__pipeline__",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ],
                pipeline_id=self.pipeline_id,
                started_at=state.get("started_at", ""),
                completed_at=datetime.utcnow().isoformat(),
            )

    def stream(self, input_text: str, config: Optional[Dict] = None):
        """
        Stream pipeline execution, yielding state after each node.

        Args:
            input_text: User input
            config: Optional LangGraph config

        Yields:
            Tuple of (node_name, state) after each node execution
        """
        if not self._compiled:
            self.compile()

        state = create_initial_state(input_text, self.pipeline_id)
        app = self._graph.compile()

        for event in app.stream(state, config=config):
            for node_name, node_state in event.items():
                yield node_name, node_state

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline definition to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "log_to_memory": self.log_to_memory,
            "project": self.project,
            "nodes": {name: config.to_dict() for name, config in self.nodes.items()},
            "edges": [{"from": e.from_node, "to": e.to_node} for e in self.edges],
            "conditional_edges": [
                {
                    "from": e.from_node,
                    "routes": e.condition_map,
                }
                for e in self.conditional_edges
            ],
            "entry_point": self.entry_point,
            "finish_points": self.finish_points,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        condition_registry: Optional[Dict[str, Callable]] = None,
    ) -> "BasePipeline":
        """
        Deserialize pipeline from dictionary.

        Args:
            data: Pipeline definition
            condition_registry: Map of condition names to functions

        Returns:
            BasePipeline instance
        """
        pipeline = cls(
            name=data["name"],
            description=data.get("description", ""),
            log_to_memory=data.get("log_to_memory", True),
            project=data.get("project"),
        )

        # Add nodes
        for name, node_data in data.get("nodes", {}).items():
            config = NodeConfig.from_dict(node_data)
            pipeline.add_node(config)

        # Add edges
        for edge in data.get("edges", []):
            pipeline.add_edge(edge["from"], edge["to"])

        # Add conditional edges (requires condition registry)
        if condition_registry:
            for edge in data.get("conditional_edges", []):
                condition_name = edge.get("condition")
                if condition_name and condition_name in condition_registry:
                    pipeline.add_conditional_edge(
                        edge["from"],
                        condition_registry[condition_name],
                        edge["routes"],
                    )

        # Set entry and finish points
        if data.get("entry_point"):
            pipeline.set_entry_point(data["entry_point"])

        for finish_point in data.get("finish_points", []):
            pipeline.set_finish_point(finish_point)

        return pipeline

    def get_node_order(self) -> List[str]:
        """Get execution order of nodes (topological sort)."""
        if not self.entry_point:
            return list(self.nodes.keys())

        visited = set()
        order = []

        def visit(node: str):
            if node in visited:
                return
            visited.add(node)
            order.append(node)

            # Find next nodes
            for edge in self.edges:
                if edge.from_node == node:
                    visit(edge.to_node)

            for edge in self.conditional_edges:
                if edge.from_node == node:
                    for target in edge.condition_map.values():
                        if target != "__end__":
                            visit(target)

        visit(self.entry_point)
        return order

    def visualize(self) -> str:
        """Generate ASCII visualization of the pipeline."""
        lines = [f"Pipeline: {self.name}", "=" * (len(self.name) + 10), ""]

        order = self.get_node_order()
        for i, node_name in enumerate(order):
            config = self.nodes[node_name]
            prefix = "[START] " if node_name == self.entry_point else ""
            suffix = " [END]" if node_name in self.finish_points else ""

            lines.append(f"{prefix}({node_name}){suffix}")
            lines.append(f"  LLM: {config.llm or 'N/A'}")

            if i < len(order) - 1:
                # Check for conditional edges
                cond_edge = next(
                    (e for e in self.conditional_edges if e.from_node == node_name),
                    None,
                )
                if cond_edge:
                    lines.append("    |")
                    lines.append("    v [conditional]")
                    for route, target in cond_edge.condition_map.items():
                        lines.append(f"    - {route} -> {target}")
                else:
                    lines.append("    |")
                    lines.append("    v")

            lines.append("")

        return "\n".join(lines)
