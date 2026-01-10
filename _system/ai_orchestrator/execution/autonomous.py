"""
Autonomous executor for pipeline execution with auto-debugging.
Wraps BasePipeline with retry logic, checkpoints, and escalation handling.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime

from .state import (
    ExecutionMode,
    ExecutionState,
    ExecutionResult,
    ErrorType,
    RetryStrategy,
    FixType,
    RetryAttempt,
    Escalation,
    create_execution_state,
)
from .error_handler import ErrorHandler
from .debug_engine import DebugEngine
from .checkpoints import CheckpointManager
from .escalation import EscalationManager
from ..pipelines.base import BasePipeline
from ..pipelines.nodes import NodeConfig


class AutonomousExecutor:
    """
    Wraps BasePipeline with autonomous execution capabilities.

    Features:
    - Auto-retry on transient errors (up to max_retries)
    - LLM-powered auto-debugging for fixable errors
    - Checkpoint system for recovery and approval gates
    - Escalation to user only when truly stuck

    Non-invasive: Existing pipelines work unchanged.
    """

    def __init__(
        self,
        pipeline: BasePipeline,
        max_retries: int = 3,
        checkpoint_nodes: Optional[List[str]] = None,
        auto_debug: bool = True,
        debug_model: str = "claude",
        escalation_callback: Optional[Callable[[Escalation], None]] = None,
    ):
        """
        Initialize autonomous executor.

        Args:
            pipeline: BasePipeline to execute
            max_retries: Maximum retry attempts per node (default: 3)
            checkpoint_nodes: Node names that require user approval
            auto_debug: Whether to use LLM-powered debugging
            debug_model: LLM to use for debug analysis
            escalation_callback: Optional callback for escalation notifications
        """
        self.pipeline = pipeline
        self.max_retries = max_retries
        self.checkpoint_nodes = checkpoint_nodes or []
        self.auto_debug = auto_debug

        # Initialize components
        self.error_handler = ErrorHandler(max_retries=max_retries)
        self.debug_engine = DebugEngine(debug_model=debug_model) if auto_debug else None
        self.checkpoint_mgr = CheckpointManager()
        self.escalation_mgr = EscalationManager(
            notification_callback=escalation_callback
        )

        # Ensure pipeline is compiled
        if not self.pipeline._compiled:
            self.pipeline.compile()

    def run(
        self,
        input_text: str,
        execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS,
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute pipeline with autonomous error handling.

        Args:
            input_text: User input to process
            execution_mode: How to handle errors and checkpoints
            custom_state: Optional custom state to include

        Returns:
            ExecutionResult with full execution trace
        """
        start_time = time.time()

        # Initialize execution state
        state = create_execution_state(
            input_text=input_text,
            pipeline_id=self.pipeline.pipeline_id,
            execution_mode=execution_mode,
            max_retries=self.max_retries,
            checkpoint_nodes=self.checkpoint_nodes,
        )

        if custom_state:
            state["custom"].update(custom_state)

        # Log execution start
        self._add_trace(
            state,
            "execution_start",
            {
                "pipeline": self.pipeline.name,
                "mode": execution_mode.value,
                "max_retries": self.max_retries,
            },
        )

        # Execute nodes in order
        node_order = self.pipeline.get_node_order()
        escalations: List[Escalation] = []

        for node_name in node_order:
            # Check for pending approval
            if state.get("pending_approval"):
                self._add_trace(
                    state,
                    "awaiting_approval",
                    {"checkpoint_id": state["pending_approval"]},
                )
                # In real use, this would pause. For now, auto-approve.
                if execution_mode == ExecutionMode.AUTONOMOUS:
                    self.checkpoint_mgr.approve(state["pending_approval"])
                    state["approved_checkpoints"].append(state["pending_approval"])
                    state["pending_approval"] = None

            # Execute node with retry logic
            node_config = self.pipeline.nodes[node_name]
            success, node_escalation = self._execute_node_with_retry(
                node_name, node_config, state
            )

            if node_escalation:
                escalations.append(node_escalation)

            if not success:
                # Node failed after all retries
                self._add_trace(state, "node_failed_final", {"node": node_name})

                if state.get("escalation_triggered"):
                    break  # Stop execution on escalation

            else:
                # Node succeeded
                state["completed_nodes"].append(node_name)

                # Save checkpoint if this is a checkpoint node
                requires_approval = node_name in self.checkpoint_nodes
                if requires_approval or execution_mode != ExecutionMode.MANUAL:
                    checkpoint = self.checkpoint_mgr.save_checkpoint(
                        state=state,
                        node_name=node_name,
                        checkpoint_type="user_defined" if requires_approval else "auto",
                        requires_approval=requires_approval,
                    )
                    self._add_trace(
                        state,
                        "checkpoint_saved",
                        {
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "requires_approval": requires_approval,
                        },
                    )

        # Calculate final metrics
        total_latency = int((time.time() - start_time) * 1000)
        total_tokens = {"input_tokens": 0, "output_tokens": 0}

        for output in state.get("outputs", {}).values():
            if isinstance(output, dict) and output.get("tokens_used"):
                total_tokens["input_tokens"] += output["tokens_used"].get(
                    "input_tokens", 0
                )
                total_tokens["output_tokens"] += output["tokens_used"].get(
                    "output_tokens", 0
                )

        # Build retry attempts list
        retry_attempts = [
            RetryAttempt(
                attempt_number=a["attempt_number"],
                node_name=a["node_name"],
                timestamp=a["timestamp"],
                error_type=ErrorType(a["error_type"]),
                error_message=a["error_message"],
                fix_type=FixType(a["fix_type"]) if a.get("fix_type") else None,
                fix_details=a.get("fix_details"),
                success=a["success"],
                latency_ms=a["latency_ms"],
            )
            for a in state.get("retry_history", [])
        ]

        self._add_trace(
            state,
            "execution_complete",
            {
                "success": len(state.get("errors", [])) == 0
                and not state.get("escalation_triggered"),
                "total_latency_ms": total_latency,
            },
        )

        return ExecutionResult(
            success=len(state.get("errors", [])) == 0
            and not state.get("escalation_triggered"),
            final_output=state.get("final_output", ""),
            node_outputs=state.get("outputs", {}),
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            errors=state.get("errors", []),
            retry_attempts=retry_attempts,
            checkpoints_created=len(state.get("checkpoints", {})),
            escalations=escalations,
            execution_mode=execution_mode,
            pipeline_id=self.pipeline.pipeline_id,
            started_at=state.get("started_at", ""),
            completed_at=datetime.utcnow().isoformat(),
            trace_log=state.get("trace_log", []),
        )

    def _execute_node_with_retry(
        self,
        node_name: str,
        node_config: NodeConfig,
        state: ExecutionState,
    ) -> Tuple[bool, Optional[Escalation]]:
        """
        Execute a single node with retry logic.

        Returns:
            Tuple of (success, escalation_if_any)
        """
        self._add_trace(
            state, "node_start", {"node": node_name, "llm": node_config.llm}
        )

        current_config = node_config
        current_input = self._get_node_input(node_config, state)

        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            attempt_start = time.time()
            state["current_attempt"] = attempt + 1

            try:
                # Execute the node
                output = self._execute_single_node(current_config, current_input, state)

                # Success!
                latency = int((time.time() - attempt_start) * 1000)
                self._add_trace(
                    state,
                    "node_success",
                    {
                        "node": node_name,
                        "attempt": attempt + 1,
                        "latency_ms": latency,
                    },
                )

                # Record successful attempt if this was a retry
                if attempt > 0:
                    self.error_handler.record_attempt(
                        state=state,
                        node_name=node_name,
                        error=Exception("N/A"),
                        error_type=ErrorType.UNKNOWN,
                        fix_type=state.get("last_error_analysis", {}).get("fix_type"),
                        fix_details=state.get("last_error_analysis", {}).get(
                            "description"
                        ),
                        success=True,
                        latency_ms=latency,
                    )

                # Store output in state
                state["outputs"][node_name] = output
                state["final_output"] = output.get("content", "")
                state["current_node"] = node_name

                return True, None

            except Exception as e:
                latency = int((time.time() - attempt_start) * 1000)

                # Classify error
                error_type = self.error_handler.classify_error(e)
                self._add_trace(
                    state,
                    "node_error",
                    {
                        "node": node_name,
                        "attempt": attempt + 1,
                        "error_type": error_type.value,
                        "error": str(e)[:200],
                    },
                )

                # Record attempt
                self.error_handler.record_attempt(
                    state=state,
                    node_name=node_name,
                    error=e,
                    error_type=error_type,
                    success=False,
                    latency_ms=latency,
                )

                # Check if we should escalate
                if self.escalation_mgr.should_escalate(state, node_name, error_type):
                    escalation = self.escalation_mgr.create_escalation(
                        state=state,
                        node_name=node_name,
                        reason=(
                            f"Max retries ({self.max_retries}) exhausted"
                            if attempt >= self.max_retries
                            else f"Fatal error: {error_type.value}"
                        ),
                        error_summary=str(e),
                    )
                    self._add_trace(
                        state,
                        "escalation_created",
                        {
                            "escalation_id": escalation.escalation_id,
                            "reason": escalation.reason,
                        },
                    )

                    # Add to errors
                    state["errors"].append(
                        {
                            "node": node_name,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                            "escalated": True,
                        }
                    )

                    return False, escalation

                # Check if we should retry
                if not self.error_handler.should_retry(state, error_type, node_name):
                    state["errors"].append(
                        {
                            "node": node_name,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    return False, None

                # Get retry strategy
                strategy = self.error_handler.get_retry_strategy(
                    error_type, attempt + 1, str(e)
                )
                self._add_trace(
                    state,
                    "retry_strategy",
                    {
                        "node": node_name,
                        "strategy": strategy.value,
                        "attempt": attempt + 1,
                    },
                )

                if strategy == RetryStrategy.ABORT:
                    state["errors"].append(
                        {
                            "node": node_name,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    return False, None

                # Apply retry strategy
                if strategy == RetryStrategy.BACKOFF:
                    backoff = self.error_handler.calculate_backoff(attempt + 1)
                    self._add_trace(state, "backoff_wait", {"seconds": backoff})
                    time.sleep(backoff)

                elif strategy in [
                    RetryStrategy.MODIFY_PROMPT,
                    RetryStrategy.FALLBACK_LLM,
                ]:
                    if self.auto_debug and self.debug_engine:
                        # Use debug engine to analyze and fix
                        analysis = self.debug_engine.analyze_error(
                            error=e,
                            error_type=error_type.value,
                            node_config=current_config,
                            input_text=current_input,
                            retry_history=state.get("retry_history", []),
                        )

                        self._add_trace(
                            state,
                            "debug_analysis",
                            {
                                "root_cause": analysis.root_cause[:100],
                                "fix_type": analysis.fix_type.value,
                                "confidence": analysis.confidence,
                            },
                        )

                        # Check if we should escalate based on confidence
                        if self.debug_engine.should_escalate(analysis, attempt + 1):
                            escalation = self.escalation_mgr.create_escalation(
                                state=state,
                                node_name=node_name,
                                reason=f"Low confidence fix ({analysis.confidence:.1%})",
                                error_summary=f"{str(e)}\n\nAnalysis: {analysis.root_cause}",
                            )
                            return False, escalation

                        # Apply fix
                        fix = self.debug_engine.generate_fix(analysis, current_config)
                        current_config, current_input = self.debug_engine.apply_fix(
                            state, current_config, fix
                        )

                        self._add_trace(
                            state,
                            "fix_applied",
                            {
                                "fix_type": fix["fix_type"].value,
                                "description": fix["description"][:100],
                            },
                        )
                    else:
                        # Simple fallback without debug engine
                        if strategy == RetryStrategy.FALLBACK_LLM:
                            fallback_llm = self._get_simple_fallback(current_config.llm)
                            current_config = NodeConfig(
                                name=current_config.name,
                                node_type=current_config.node_type,
                                llm=fallback_llm,
                                system_prompt=current_config.system_prompt,
                                temperature=current_config.temperature,
                                max_tokens=current_config.max_tokens,
                                input_key=current_config.input_key,
                                output_key=current_config.output_key,
                            )

        # Should not reach here, but just in case
        return False, None

    def _execute_single_node(
        self,
        config: NodeConfig,
        input_text: str,
        state: ExecutionState,
    ) -> Dict[str, Any]:
        """Execute a single node and return its output."""
        from ..core.llm_clients import get_clients

        clients = get_clients()

        # Apply input template if present
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
        response = clients.call(
            prompt=prompt,
            model=config.llm,
            system=config.system_prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "latency_ms": latency_ms,
            "tokens_used": response.usage,
        }

    def _get_node_input(self, config: NodeConfig, state: ExecutionState) -> str:
        """Get input text for a node based on its configuration."""
        if config.input_key == "input":
            return state.get("input", "")
        elif config.input_key in state.get("outputs", {}):
            return state["outputs"][config.input_key].get("content", "")
        else:
            return state.get("custom", {}).get(config.input_key, state.get("input", ""))

    def _get_simple_fallback(self, current_llm: str) -> str:
        """Get a simple fallback LLM."""
        fallbacks = {
            "claude": "gpt4o",
            "gpt4o": "claude",
            "gemini": "claude",
            "grok": "gpt4o",
        }
        return fallbacks.get(current_llm.lower(), "claude")

    def _add_trace(self, state: ExecutionState, event: str, data: Dict[str, Any]):
        """Add an event to the execution trace log."""
        trace_log = list(state.get("trace_log", []))
        trace_log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": event,
                "data": data,
            }
        )
        state["trace_log"] = trace_log

    def resume(
        self,
        checkpoint_id: str,
        execution_mode: Optional[ExecutionMode] = None,
    ) -> ExecutionResult:
        """
        Resume execution from a saved checkpoint.

        Args:
            checkpoint_id: Checkpoint to resume from
            execution_mode: Optional override for execution mode

        Returns:
            ExecutionResult from resumed execution
        """
        # Load checkpoint
        state = self.checkpoint_mgr.restore_state(checkpoint_id)
        if not state:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # Override execution mode if specified
        if execution_mode:
            state["execution_mode"] = execution_mode.value

        # Find where to resume
        completed = set(state.get("completed_nodes", []))
        node_order = self.pipeline.get_node_order()

        # Find first incomplete node
        resume_from = None
        for node_name in node_order:
            if node_name not in completed:
                resume_from = node_name
                break

        if not resume_from:
            # All nodes complete - return current state as result
            return self._state_to_result(state)

        self._add_trace(
            state,
            "resume_from_checkpoint",
            {
                "checkpoint_id": checkpoint_id,
                "resume_node": resume_from,
                "completed_nodes": list(completed),
            },
        )

        # Continue execution from resume point
        escalations: List[Escalation] = []

        resume_index = node_order.index(resume_from)
        for node_name in node_order[resume_index:]:
            node_config = self.pipeline.nodes[node_name]
            success, node_escalation = self._execute_node_with_retry(
                node_name, node_config, state
            )

            if node_escalation:
                escalations.append(node_escalation)

            if not success:
                if state.get("escalation_triggered"):
                    break
            else:
                state["completed_nodes"].append(node_name)

        return self._state_to_result(state, escalations)

    def _state_to_result(
        self,
        state: ExecutionState,
        escalations: Optional[List[Escalation]] = None,
    ) -> ExecutionResult:
        """Convert execution state to result."""
        total_tokens = {"input_tokens": 0, "output_tokens": 0}
        for output in state.get("outputs", {}).values():
            if isinstance(output, dict) and output.get("tokens_used"):
                total_tokens["input_tokens"] += output["tokens_used"].get(
                    "input_tokens", 0
                )
                total_tokens["output_tokens"] += output["tokens_used"].get(
                    "output_tokens", 0
                )

        retry_attempts = [
            RetryAttempt(
                attempt_number=a["attempt_number"],
                node_name=a["node_name"],
                timestamp=a["timestamp"],
                error_type=ErrorType(a["error_type"]),
                error_message=a["error_message"],
                fix_type=FixType(a["fix_type"]) if a.get("fix_type") else None,
                fix_details=a.get("fix_details"),
                success=a["success"],
                latency_ms=a["latency_ms"],
            )
            for a in state.get("retry_history", [])
        ]

        return ExecutionResult(
            success=len(state.get("errors", [])) == 0
            and not state.get("escalation_triggered"),
            final_output=state.get("final_output", ""),
            node_outputs=state.get("outputs", {}),
            total_latency_ms=0,  # Not tracked for resumed executions
            total_tokens=total_tokens,
            errors=state.get("errors", []),
            retry_attempts=retry_attempts,
            checkpoints_created=len(state.get("checkpoints", {})),
            escalations=escalations or [],
            execution_mode=ExecutionMode(state.get("execution_mode", "autonomous")),
            pipeline_id=state.get("pipeline_id", "unknown"),
            started_at=state.get("started_at", ""),
            completed_at=datetime.utcnow().isoformat(),
            trace_log=state.get("trace_log", []),
        )

    def get_execution_summary(self, result: ExecutionResult) -> str:
        """
        Generate a human-readable summary of execution.

        Args:
            result: ExecutionResult to summarize

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "EXECUTION SUMMARY",
            "=" * 60,
            f"Pipeline: {self.pipeline.name}",
            f"Status: {'SUCCESS' if result.success else 'FAILED'}",
            f"Mode: {result.execution_mode.value}",
            f"Duration: {result.total_latency_ms}ms",
            "",
            "NODE OUTPUTS:",
        ]

        for node_name, output in result.node_outputs.items():
            content = output.get("content", "")[:100]
            lines.append(f"  {node_name}: {content}...")

        if result.retry_attempts:
            lines.extend(["", f"RETRY ATTEMPTS: {len(result.retry_attempts)}"])
            for attempt in result.retry_attempts[-5:]:  # Show last 5
                status = "OK" if attempt.success else "FAIL"
                lines.append(
                    f"  {attempt.node_name} #{attempt.attempt_number}: "
                    f"{attempt.error_type.value} - {status}"
                )

        if result.escalations:
            lines.extend(["", f"ESCALATIONS: {len(result.escalations)}"])
            for esc in result.escalations:
                lines.append(f"  {esc.node_name}: {esc.reason}")

        if result.errors:
            lines.extend(["", "ERRORS:"])
            for err in result.errors[-3:]:
                lines.append(
                    f"  {err.get('node', 'unknown')}: {err.get('error', 'unknown')[:100]}"
                )

        lines.extend(
            [
                "",
                f"Tokens: {result.total_tokens.get('input_tokens', 0)} in, "
                f"{result.total_tokens.get('output_tokens', 0)} out",
                f"Checkpoints: {result.checkpoints_created}",
                "=" * 60,
            ]
        )

        return "\n".join(lines)
