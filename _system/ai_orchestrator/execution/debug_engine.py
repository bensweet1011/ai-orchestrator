"""
LLM-powered auto-debugging engine.
Uses Claude to analyze errors and generate fixes.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .state import (
    DebugAnalysis,
    ExecutionState,
    FixType,
)
from ..pipelines.nodes import NodeConfig


DEBUG_ANALYSIS_PROMPT = """You are an expert debugger for LLM pipeline nodes. Analyze the following error and provide a fix.

NODE CONFIGURATION:
- Name: {node_name}
- LLM: {model}
- System Prompt: {system_prompt}
- Temperature: {temperature}
- Max Tokens: {max_tokens}

INPUT TO NODE:
{input_text}

ERROR DETAILS:
- Type: {error_type}
- Message: {error_message}

{retry_history}

PARTIAL OUTPUT (if any):
{partial_output}

Analyze this failure and respond with ONLY a JSON object (no markdown, no explanation):
{{
    "root_cause": "Brief explanation of why this failed",
    "fix_type": "prompt_modification|temperature_change|llm_switch|input_transform|max_tokens_increase|retry_only",
    "specific_fix": "Exact change to make - be specific",
    "confidence": 0.0-1.0,
    "reasoning": "Why this fix should work",
    "alternative_fixes": [
        {{"fix_type": "...", "specific_fix": "...", "confidence": 0.0}}
    ]
}}

Fix types:
- prompt_modification: Change the system prompt to handle edge case
- temperature_change: Adjust temperature (specify new value like "0.3")
- llm_switch: Try different LLM (suggest specific model like "gpt4o" or "claude")
- input_transform: Modify input before sending (specify transformation)
- max_tokens_increase: Increase max_tokens (specify new value)
- retry_only: Just retry without changes (for transient errors)

Return ONLY the JSON object."""


FALLBACK_LLMS = {
    "claude": ["gpt4o", "gemini"],
    "claude-sonnet": ["gpt4o", "gemini"],
    "claude-opus": ["claude", "gpt4o"],
    "gpt4o": ["claude", "gemini"],
    "gpt-4o": ["claude", "gemini"],
    "gpt4": ["gpt4o", "claude"],
    "gemini": ["claude", "gpt4o"],
    "grok": ["claude", "gpt4o"],
    "perplexity": ["claude", "gpt4o"],
    "o1": ["claude-opus", "gpt4o"],
    "o3-mini": ["gpt4o", "claude"],
}


class DebugEngine:
    """
    LLM-powered auto-debugging using Claude as the analyzer.
    Analyzes errors and generates fixes to apply before retry.
    """

    def __init__(self, debug_model: str = "claude"):
        """
        Initialize the debug engine.

        Args:
            debug_model: LLM to use for error analysis (default: Claude)
        """
        self.debug_model = debug_model

    def analyze_error(
        self,
        error: Exception,
        error_type: str,
        node_config: NodeConfig,
        input_text: str,
        partial_output: Optional[str] = None,
        retry_history: Optional[List[Dict]] = None,
    ) -> DebugAnalysis:
        """
        Use Claude to analyze what went wrong and suggest a fix.

        Args:
            error: The exception that occurred
            error_type: Classification of the error
            node_config: Configuration of the failed node
            input_text: Input that was sent to the node
            partial_output: Any partial output before failure
            retry_history: Previous retry attempts for context

        Returns:
            DebugAnalysis with root cause and suggested fix
        """
        from ..core.llm_clients import get_clients

        # Format retry history
        history_str = ""
        if retry_history:
            history_str = "PREVIOUS ATTEMPTS:\n"
            for attempt in retry_history[-3:]:  # Last 3 attempts
                history_str += (
                    f"- Attempt {attempt.get('attempt_number', '?')}: "
                    f"{attempt.get('fix_type', 'none')} - "
                    f"{'Success' if attempt.get('success') else 'Failed'}\n"
                )

        # Build prompt
        prompt = DEBUG_ANALYSIS_PROMPT.format(
            node_name=node_config.name,
            model=node_config.llm,
            system_prompt=node_config.system_prompt or "(none)",
            temperature=node_config.temperature,
            max_tokens=node_config.max_tokens,
            input_text=input_text[:2000],  # Truncate for token limits
            error_type=error_type,
            error_message=str(error),
            retry_history=history_str,
            partial_output=partial_output[:1000] if partial_output else "(none)",
        )

        try:
            clients = get_clients()
            response = clients.call(
                prompt=prompt,
                model=self.debug_model,
                system="You are a debugging assistant. Respond only with valid JSON.",
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1000,
            )

            # Parse response
            analysis = self._parse_analysis(response.content)
            return analysis

        except Exception as e:
            # If debug analysis itself fails, return a safe default
            return DebugAnalysis(
                root_cause=f"Debug analysis failed: {str(e)}",
                fix_type=FixType.RETRY_ONLY,
                specific_fix="Retry without modification",
                confidence=0.3,
                reasoning="Debug analysis encountered an error, defaulting to simple retry",
                alternative_fixes=[],
            )

    def _parse_analysis(self, response_text: str) -> DebugAnalysis:
        """Parse the LLM response into a DebugAnalysis object."""
        try:
            # Try to extract JSON from response
            text = response_text.strip()

            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            # Map fix_type string to enum
            fix_type_str = data.get("fix_type", "retry_only").lower()
            fix_type_map = {
                "prompt_modification": FixType.PROMPT_MODIFICATION,
                "temperature_change": FixType.TEMPERATURE_CHANGE,
                "llm_switch": FixType.LLM_SWITCH,
                "input_transform": FixType.INPUT_TRANSFORM,
                "max_tokens_increase": FixType.MAX_TOKENS_INCREASE,
                "retry_only": FixType.RETRY_ONLY,
            }
            fix_type = fix_type_map.get(fix_type_str, FixType.RETRY_ONLY)

            return DebugAnalysis(
                root_cause=data.get("root_cause", "Unknown"),
                fix_type=fix_type,
                specific_fix=data.get("specific_fix", ""),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                alternative_fixes=data.get("alternative_fixes", []),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback if parsing fails
            return DebugAnalysis(
                root_cause="Could not parse debug analysis",
                fix_type=FixType.RETRY_ONLY,
                specific_fix="Retry without modification",
                confidence=0.3,
                reasoning=f"JSON parsing failed: {str(e)}",
                alternative_fixes=[],
            )

    def generate_fix(
        self, analysis: DebugAnalysis, node_config: NodeConfig
    ) -> Dict[str, Any]:
        """
        Generate a concrete fix based on the analysis.

        Args:
            analysis: The debug analysis result
            node_config: Current node configuration

        Returns:
            Dictionary with fix details to apply
        """
        fix = {
            "fix_type": analysis.fix_type,
            "changes": {},
            "description": analysis.specific_fix,
        }

        if analysis.fix_type == FixType.PROMPT_MODIFICATION:
            # Modify system prompt based on analysis
            fix["changes"]["system_prompt"] = self._modify_prompt(
                node_config.system_prompt, analysis.specific_fix
            )

        elif analysis.fix_type == FixType.TEMPERATURE_CHANGE:
            # Parse new temperature from specific_fix
            try:
                new_temp = float(
                    "".join(c for c in analysis.specific_fix if c.isdigit() or c == ".")
                )
                new_temp = max(0.0, min(1.0, new_temp))  # Clamp to valid range
            except ValueError:
                new_temp = 0.5  # Default if parsing fails
            fix["changes"]["temperature"] = new_temp

        elif analysis.fix_type == FixType.LLM_SWITCH:
            # Get fallback LLM
            fallback = self._get_fallback_llm(node_config.llm, analysis.specific_fix)
            fix["changes"]["llm"] = fallback

        elif analysis.fix_type == FixType.INPUT_TRANSFORM:
            # Store transformation instruction
            fix["changes"]["input_transform"] = analysis.specific_fix

        elif analysis.fix_type == FixType.MAX_TOKENS_INCREASE:
            # Parse new max_tokens
            try:
                new_max = int("".join(c for c in analysis.specific_fix if c.isdigit()))
                new_max = min(new_max, 8000)  # Cap at reasonable limit
            except ValueError:
                new_max = node_config.max_tokens * 2
            fix["changes"]["max_tokens"] = new_max

        # RETRY_ONLY has no changes

        return fix

    def _modify_prompt(self, original_prompt: Optional[str], modification: str) -> str:
        """Apply a modification to the system prompt."""
        if not original_prompt:
            return modification

        # Append modification as additional instruction
        return f"{original_prompt}\n\nADDITIONAL INSTRUCTION: {modification}"

    def _get_fallback_llm(
        self, current_llm: str, suggested: Optional[str] = None
    ) -> str:
        """Get a fallback LLM different from the current one."""
        # If a specific suggestion was made, try to use it
        if suggested:
            # Extract model name from suggestion
            suggested_lower = suggested.lower()
            for model in ["claude", "gpt4o", "gpt4", "gemini", "grok", "perplexity"]:
                if model in suggested_lower:
                    if model != current_llm.lower():
                        return model

        # Use predefined fallbacks
        fallbacks = FALLBACK_LLMS.get(current_llm.lower(), ["claude", "gpt4o"])
        for fallback in fallbacks:
            if fallback != current_llm.lower():
                return fallback

        # Ultimate fallback
        return "claude" if current_llm.lower() != "claude" else "gpt4o"

    def apply_fix(
        self,
        state: ExecutionState,
        node_config: NodeConfig,
        fix: Dict[str, Any],
    ) -> Tuple[NodeConfig, str]:
        """
        Apply a fix to the node configuration.

        Args:
            state: Current execution state
            node_config: Original node configuration
            fix: Fix details from generate_fix()

        Returns:
            Tuple of (modified NodeConfig, modified input_text)
        """

        changes = fix.get("changes", {})
        input_text = state.get("input", "")

        # Get the input for this node
        if node_config.input_key != "input":
            if node_config.input_key in state.get("outputs", {}):
                input_text = state["outputs"][node_config.input_key].get("content", "")

        # Create modified config
        modified_config = NodeConfig(
            name=node_config.name,
            node_type=node_config.node_type,
            llm=changes.get("llm", node_config.llm),
            system_prompt=changes.get("system_prompt", node_config.system_prompt),
            temperature=changes.get("temperature", node_config.temperature),
            max_tokens=changes.get("max_tokens", node_config.max_tokens),
            input_key=node_config.input_key,
            output_key=node_config.output_key,
            input_template=node_config.input_template,
            description=node_config.description,
        )

        # Apply input transform if specified
        if "input_transform" in changes:
            input_text = self._apply_input_transform(
                input_text, changes["input_transform"]
            )

        # Record the applied fix
        applied_fixes = list(state.get("applied_fixes", []))
        applied_fixes.append(
            f"{node_config.name}: {fix['fix_type'].value} - {fix['description']}"
        )
        state["applied_fixes"] = applied_fixes
        state["last_error_analysis"] = {
            "fix_type": fix["fix_type"].value,
            "changes": changes,
            "description": fix["description"],
        }

        return modified_config, input_text

    def _apply_input_transform(self, input_text: str, transform: str) -> str:
        """Apply a transformation to the input text."""
        transform_lower = transform.lower()

        # Common transformations
        if "truncate" in transform_lower:
            # Try to extract length
            import re

            match = re.search(r"(\d+)", transform)
            length = int(match.group(1)) if match else 2000
            return input_text[:length]

        elif "simplify" in transform_lower or "shorter" in transform_lower:
            # Truncate to first portion
            return input_text[: len(input_text) // 2]

        elif "escape" in transform_lower:
            # Escape special characters
            return input_text.replace("\\", "\\\\").replace('"', '\\"')

        elif "clean" in transform_lower:
            # Remove non-printable characters
            return "".join(c for c in input_text if c.isprintable() or c in "\n\t")

        # If no specific transform recognized, prepend instruction
        return f"[{transform}]\n\n{input_text}"

    def verify_fix(
        self,
        new_output: str,
        original_error: str,
        expected_behavior: Optional[str] = None,
    ) -> bool:
        """
        Check if a fix resolved the issue.

        Args:
            new_output: Output from the retry attempt
            original_error: The original error message
            expected_behavior: Optional description of expected behavior

        Returns:
            True if the fix appears successful
        """
        # Basic checks
        if not new_output or len(new_output.strip()) == 0:
            return False

        # Check if the error message appears in output (bad sign)
        if original_error.lower() in new_output.lower():
            return False

        # Check for common error indicators in output
        error_indicators = [
            "error:",
            "exception:",
            "failed:",
            "cannot ",
            "unable to",
            "invalid ",
        ]
        output_lower = new_output.lower()
        for indicator in error_indicators:
            if output_lower.startswith(indicator):
                return False

        return True

    def should_escalate(self, analysis: DebugAnalysis, attempt_number: int) -> bool:
        """
        Determine if we should escalate to user instead of retrying.

        Args:
            analysis: The debug analysis result
            attempt_number: Current attempt number

        Returns:
            True if escalation is recommended
        """
        # Low confidence after multiple attempts
        if analysis.confidence < 0.3 and attempt_number >= 2:
            return True

        # No clear fix available
        if analysis.fix_type == FixType.RETRY_ONLY and attempt_number >= 2:
            return True

        # Root cause suggests human intervention needed
        escalation_keywords = [
            "requires human",
            "manual review",
            "configuration error",
            "permission",
            "authentication",
            "billing",
            "account",
        ]
        root_cause_lower = analysis.root_cause.lower()
        for keyword in escalation_keywords:
            if keyword in root_cause_lower:
                return True

        return False
