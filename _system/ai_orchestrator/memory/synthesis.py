"""
Synthesis system for combining outputs from multiple pipeline runs.
Uses LLMs to intelligently merge, compare, and extract patterns.
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from .state import (
    MemoryType,
    PipelineMemory,
    SynthesisResult,
)
from .store import MemoryStore, get_memory_store


class SynthesisType(Enum):
    """Types of synthesis operations."""

    COMBINE = "combine"  # Merge outputs into unified result
    COMPARE = "compare"  # Compare and contrast outputs
    PATTERNS = "patterns"  # Extract common patterns/themes
    BEST_OF = "best_of"  # Select and enhance best output
    TIMELINE = "timeline"  # Create chronological summary


@dataclass
class ComparisonResult:
    """Result of comparing multiple outputs."""

    run_ids: List[str]
    similarities: List[str]
    differences: List[str]
    recommendations: List[str]
    summary: str


class Synthesizer:
    """
    Combines and analyzes outputs from multiple pipeline runs.

    Features:
    - Combine multiple outputs into unified result
    - Compare outputs across runs
    - Extract common patterns and themes
    - Generate synthesis reports
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        synthesis_model: str = "claude",
    ):
        """
        Initialize synthesizer.

        Args:
            store: MemoryStore instance
            synthesis_model: LLM to use for synthesis
        """
        self.store = store or get_memory_store()
        self.synthesis_model = synthesis_model

    def _generate_id(self, prefix: str = "syn") -> str:
        """Generate unique ID."""
        ts = datetime.utcnow().isoformat()
        unique = hashlib.md5(f"{ts}{time.time_ns()}".encode()).hexdigest()[:8]
        return f"{prefix}_{unique}"

    def _get_llm_client(self):
        """Get LLM client for synthesis."""
        from ..core.llm_clients import get_clients

        return get_clients()

    def synthesize(
        self,
        run_ids: List[str],
        synthesis_type: SynthesisType = SynthesisType.COMBINE,
        instructions: Optional[str] = None,
        project: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesize outputs from multiple pipeline runs.

        Args:
            run_ids: List of pipeline run IDs to synthesize
            synthesis_type: Type of synthesis to perform
            instructions: Custom instructions for synthesis
            project: Project context

        Returns:
            SynthesisResult with combined output
        """
        project = project or self.store.current_project
        request_id = self._generate_id("sreq")
        result_id = self._generate_id("sres")
        timestamp = datetime.utcnow().isoformat()

        # Fetch all runs
        runs: List[PipelineMemory] = []
        for run_id in run_ids:
            run = self.store.get_pipeline_run(run_id, project)
            if run:
                runs.append(run)

        if not runs:
            return SynthesisResult(
                id=result_id,
                request_id=request_id,
                project=project,
                timestamp=timestamp,
                source_run_ids=run_ids,
                source_count=0,
                synthesized_output="No valid runs found to synthesize.",
                synthesis_type=synthesis_type.value,
                model_used=self.synthesis_model,
                latency_ms=0,
            )

        # Build synthesis prompt based on type
        prompt = self._build_synthesis_prompt(runs, synthesis_type, instructions)

        # Call LLM
        clients = self._get_llm_client()
        start_time = time.time()

        response = clients.call(prompt, model=self.synthesis_model)
        latency_ms = int((time.time() - start_time) * 1000)

        # Create result
        result = SynthesisResult(
            id=result_id,
            request_id=request_id,
            project=project,
            timestamp=timestamp,
            source_run_ids=run_ids,
            source_count=len(runs),
            synthesized_output=response.content,
            synthesis_type=synthesis_type.value,
            model_used=self.synthesis_model,
            latency_ms=latency_ms,
            token_usage=response.usage or {},
        )

        # Store synthesis result in memory
        self.store.store_entry(
            content=f"SYNTHESIS ({synthesis_type.value}): {response.content[:2000]}",
            memory_type=MemoryType.SYNTHESIS,
            project=project,
            metadata={
                "synthesis_id": result_id,
                "source_count": len(runs),
                "synthesis_type": synthesis_type.value,
            },
        )

        return result

    def _build_synthesis_prompt(
        self,
        runs: List[PipelineMemory],
        synthesis_type: SynthesisType,
        instructions: Optional[str] = None,
    ) -> str:
        """Build synthesis prompt based on type and runs."""
        # Format runs
        runs_text = self._format_runs_for_prompt(runs)

        if synthesis_type == SynthesisType.COMBINE:
            base_prompt = f"""You are an expert at synthesizing information.
Combine the following pipeline outputs into a single, comprehensive result.

Merge common information, resolve any contradictions, and create a unified output that captures the best from each run.

{runs_text}

Create a synthesized output that:
1. Preserves all key information from each run
2. Eliminates redundancy
3. Resolves any contradictions (preferring more recent or higher-quality outputs)
4. Maintains a coherent structure

SYNTHESIZED OUTPUT:"""

        elif synthesis_type == SynthesisType.COMPARE:
            base_prompt = f"""You are an expert analyst.
Compare and contrast the following pipeline outputs.

{runs_text}

Provide a detailed comparison that includes:
1. **Similarities**: What themes or content appear across all outputs
2. **Differences**: How the outputs differ in approach, content, or conclusions
3. **Quality Assessment**: Which outputs are stronger and why
4. **Recommendations**: Which output(s) to use for different purposes

COMPARISON ANALYSIS:"""

        elif synthesis_type == SynthesisType.PATTERNS:
            base_prompt = f"""You are an expert at pattern recognition.
Analyze the following pipeline outputs to extract common patterns and themes.

{runs_text}

Identify and describe:
1. **Recurring Themes**: Topics or ideas that appear across multiple outputs
2. **Common Structures**: Similar approaches or formats used
3. **Consistent Conclusions**: Points where outputs agree
4. **Emerging Insights**: Patterns that reveal deeper understanding
5. **Anomalies**: Unusual or unique elements worth noting

PATTERN ANALYSIS:"""

        elif synthesis_type == SynthesisType.BEST_OF:
            base_prompt = f"""You are an expert editor.
Review the following pipeline outputs and create an enhanced version that takes the best elements from each.

{runs_text}

Create an improved output that:
1. Uses the strongest content from each run
2. Incorporates the best structural elements
3. Addresses any weaknesses in individual outputs
4. Produces a result better than any single run

ENHANCED OUTPUT:"""

        elif synthesis_type == SynthesisType.TIMELINE:
            base_prompt = f"""You are an expert chronicler.
Create a chronological summary of the following pipeline outputs.

{runs_text}

Create a timeline that:
1. Orders outputs chronologically
2. Shows progression of ideas or content over time
3. Highlights what changed between runs
4. Summarizes the evolution of outputs

CHRONOLOGICAL SUMMARY:"""

        else:
            base_prompt = f"""Synthesize the following outputs:

{runs_text}

SYNTHESIS:"""

        # Add custom instructions if provided
        if instructions:
            base_prompt = f"""CUSTOM INSTRUCTIONS: {instructions}

{base_prompt}"""

        return base_prompt

    def _format_runs_for_prompt(self, runs: List[PipelineMemory]) -> str:
        """Format pipeline runs for inclusion in prompt."""
        formatted = []

        for i, run in enumerate(runs, 1):
            formatted.append(
                f"""
=== Run {i} ===
Date: {run.timestamp[:10]}
Pipeline: {run.pipeline_name}
Status: {'Success' if run.success else 'Failed'}
Latency: {run.total_latency_ms}ms

INPUT:
{run.input_text[:1000]}

OUTPUT:
{run.final_output[:2000]}

NODES:
{self._format_node_outputs(run.node_outputs)}
"""
            )

        return "\n".join(formatted)

    def _format_node_outputs(self, node_outputs: Dict[str, Dict]) -> str:
        """Format node outputs for prompt."""
        lines = []
        for name, output in node_outputs.items():
            content = output.get("content", "")[:500]
            model = output.get("model", "unknown")
            lines.append(f"  - {name} ({model}): {content}...")
        return "\n".join(lines)

    def compare_runs(
        self,
        run_ids: List[str],
        project: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Compare multiple pipeline runs and identify differences.

        Args:
            run_ids: Run IDs to compare
            project: Project context

        Returns:
            ComparisonResult with detailed comparison
        """
        # Use synthesis with COMPARE type
        synthesis = self.synthesize(
            run_ids=run_ids,
            synthesis_type=SynthesisType.COMPARE,
            project=project,
        )

        # Parse the comparison (simplified - could use structured output)
        content = synthesis.synthesized_output

        return ComparisonResult(
            run_ids=run_ids,
            similarities=self._extract_section(content, "Similarities"),
            differences=self._extract_section(content, "Differences"),
            recommendations=self._extract_section(content, "Recommendations"),
            summary=content[:500],
        )

    def _extract_section(self, text: str, section_name: str) -> List[str]:
        """Extract bullet points from a section."""
        lines = []
        in_section = False

        for line in text.split("\n"):
            if section_name.lower() in line.lower() and ("**" in line or "#" in line):
                in_section = True
                continue

            if in_section:
                if line.startswith("**") or line.startswith("#"):
                    break
                if line.strip().startswith("-") or line.strip().startswith("•"):
                    lines.append(line.strip().lstrip("-•").strip())

        return lines

    def extract_patterns(
        self,
        project: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        limit: int = 10,
    ) -> SynthesisResult:
        """
        Extract patterns from recent pipeline runs.

        Args:
            project: Project to analyze
            pipeline_name: Filter by pipeline name
            limit: Number of runs to analyze

        Returns:
            SynthesisResult with pattern analysis
        """
        project = project or self.store.current_project

        # Get recent runs
        runs = self.store.get_pipeline_history(
            pipeline_name=pipeline_name,
            project=project,
            limit=limit,
            success_only=True,
        )

        if not runs:
            return SynthesisResult(
                id=self._generate_id("sres"),
                request_id=self._generate_id("sreq"),
                project=project,
                timestamp=datetime.utcnow().isoformat(),
                source_run_ids=[],
                source_count=0,
                synthesized_output="No runs found to analyze for patterns.",
                synthesis_type=SynthesisType.PATTERNS.value,
                model_used=self.synthesis_model,
                latency_ms=0,
            )

        run_ids = [run.id for run in runs]

        return self.synthesize(
            run_ids=run_ids,
            synthesis_type=SynthesisType.PATTERNS,
            project=project,
        )

    def generate_project_synthesis(
        self,
        project: Optional[str] = None,
        include_failed: bool = False,
    ) -> str:
        """
        Generate a comprehensive synthesis of all project activity.

        Args:
            project: Project to synthesize
            include_failed: Include failed runs

        Returns:
            Formatted project synthesis report
        """
        project = project or self.store.current_project

        # Get all runs
        runs = self.store.get_pipeline_history(
            project=project,
            limit=50,
            success_only=not include_failed,
        )

        if not runs:
            return f"No pipeline runs found for project: {project}"

        # Group by pipeline
        by_pipeline: Dict[str, List[PipelineMemory]] = {}
        for run in runs:
            if run.pipeline_name not in by_pipeline:
                by_pipeline[run.pipeline_name] = []
            by_pipeline[run.pipeline_name].append(run)

        # Build report
        report_lines = [
            f"# Project Synthesis: {project}",
            f"Generated: {datetime.utcnow().isoformat()[:10]}",
            "",
            "## Summary",
            f"- Total runs analyzed: {len(runs)}",
            f"- Pipelines used: {len(by_pipeline)}",
            f"- Success rate: {sum(1 for r in runs if r.success) / len(runs) * 100:.1f}%",
            "",
            "## Pipeline Activity",
            "",
        ]

        for pipeline_name, pipeline_runs in by_pipeline.items():
            success_count = sum(1 for r in pipeline_runs if r.success)
            avg_latency = sum(r.total_latency_ms for r in pipeline_runs) / len(
                pipeline_runs
            )

            report_lines.extend(
                [
                    f"### {pipeline_name}",
                    f"- Runs: {len(pipeline_runs)}",
                    f"- Success rate: {success_count / len(pipeline_runs) * 100:.1f}%",
                    f"- Average latency: {avg_latency:.0f}ms",
                    "",
                ]
            )

            # Sample recent outputs
            if pipeline_runs:
                recent = pipeline_runs[0]
                report_lines.extend(
                    [
                        "**Most Recent Output Preview:**",
                        f"> {recent.final_output[:300]}...",
                        "",
                    ]
                )

        # Use LLM for insights if we have enough data
        if len(runs) >= 5:
            run_ids = [r.id for r in runs[:10]]
            patterns = self.synthesize(
                run_ids=run_ids,
                synthesis_type=SynthesisType.PATTERNS,
                project=project,
            )
            report_lines.extend(
                [
                    "## Key Insights",
                    "",
                    patterns.synthesized_output,
                ]
            )

        return "\n".join(report_lines)


# Singleton instance
_synthesizer: Optional[Synthesizer] = None


def get_synthesizer() -> Synthesizer:
    """Get or create Synthesizer singleton."""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = Synthesizer()
    return _synthesizer
