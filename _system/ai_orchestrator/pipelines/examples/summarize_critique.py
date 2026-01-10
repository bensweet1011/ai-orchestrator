"""
Test pipeline: Summarize then Critique.
Demonstrates user-defined LLM selection per node.
"""


from ..base import BasePipeline
from ..nodes import NodeConfig, NodeType


def create_summarize_critique_pipeline(
    summarizer_llm: str = "gpt4o",
    critic_llm: str = "claude",
    name: str = "summarize_critique",
) -> BasePipeline:
    """
    Create a two-node pipeline that summarizes text then critiques the summary.

    This demonstrates:
    - User-defined LLM selection per node
    - State passing between nodes
    - Sequential execution

    Args:
        summarizer_llm: LLM to use for summarization (e.g., "gpt4o", "claude", "gemini")
        critic_llm: LLM to use for critique (e.g., "claude", "gpt4o", "gemini")
        name: Pipeline name

    Returns:
        Configured BasePipeline ready to run

    Example:
        >>> pipeline = create_summarize_critique_pipeline(
        ...     summarizer_llm="gpt4o",
        ...     critic_llm="claude"
        ... )
        >>> result = pipeline.run("Long text to process...")
        >>> print(result.final_output)  # The critique
        >>> print(result.node_outputs["summarize"]["content"])  # The summary
    """
    pipeline = BasePipeline(
        name=name,
        description="Two-stage pipeline: summarize input text, then critique the summary",
    )

    # Node 1: Summarizer
    pipeline.add_node(
        NodeConfig(
            name="summarize",
            node_type=NodeType.LLM,
            llm=summarizer_llm,
            system_prompt=(
                "You are an expert summarizer. Your task is to create clear, concise "
                "summaries that capture the essential information.\n\n"
                "Guidelines:\n"
                "- Identify and preserve key points\n"
                "- Maintain the original meaning and intent\n"
                "- Use clear, accessible language\n"
                "- Aim for 2-3 paragraphs unless the input is very short"
            ),
            input_key="input",
            output_key="summarize",
            temperature=0.5,
            description="Summarizes the input text",
        )
    )

    # Node 2: Critic
    pipeline.add_node(
        NodeConfig(
            name="critique",
            node_type=NodeType.LLM,
            llm=critic_llm,
            system_prompt=(
                "You are a critical reviewer. Your task is to evaluate summaries for "
                "quality, accuracy, and completeness.\n\n"
                "Provide your critique in this format:\n"
                "1. ACCURACY: Does the summary faithfully represent the original?\n"
                "2. COMPLETENESS: Are all key points captured?\n"
                "3. CLARITY: Is the summary easy to understand?\n"
                "4. CONCISENESS: Is it appropriately brief without losing meaning?\n"
                "5. OVERALL SCORE: Rate 1-10 with brief justification\n"
                "6. SUGGESTIONS: Specific improvements if needed"
            ),
            input_key="summarize",  # Takes output from summarize node
            output_key="critique",
            temperature=0.7,
            description="Critiques the summary",
        )
    )

    # Connect nodes
    pipeline.add_edge("summarize", "critique")

    # Set entry and exit
    pipeline.set_entry_point("summarize")
    pipeline.set_finish_point("critique")

    return pipeline


def create_summarize_critique_conditional(
    summarizer_llm: str = "gpt4o",
    quick_critic_llm: str = "gpt4o",
    deep_critic_llm: str = "claude",
    name: str = "summarize_critique_conditional",
) -> BasePipeline:
    """
    Create a pipeline with conditional routing based on summary length.

    Short summaries get quick review, long summaries get deep analysis.

    Args:
        summarizer_llm: LLM for summarization
        quick_critic_llm: LLM for quick reviews (short summaries)
        deep_critic_llm: LLM for deep analysis (long summaries)
        name: Pipeline name

    Returns:
        Configured BasePipeline with conditional routing
    """
    from ..nodes import route_by_length

    pipeline = BasePipeline(
        name=name,
        description="Conditional pipeline: routes to different critics based on summary length",
    )

    # Node 1: Summarizer
    pipeline.add_node(
        NodeConfig(
            name="summarize",
            node_type=NodeType.LLM,
            llm=summarizer_llm,
            system_prompt="Summarize the following text. Be thorough but concise.",
            input_key="input",
            output_key="summarize",
        )
    )

    # Node 2a: Quick Critic (for short summaries)
    pipeline.add_node(
        NodeConfig(
            name="quick_critique",
            node_type=NodeType.LLM,
            llm=quick_critic_llm,
            system_prompt="Provide a brief quality check of this summary. "
            "Note any obvious issues in 2-3 sentences.",
            input_key="summarize",
            output_key="critique",
        )
    )

    # Node 2b: Deep Critic (for long summaries)
    pipeline.add_node(
        NodeConfig(
            name="deep_critique",
            node_type=NodeType.LLM,
            llm=deep_critic_llm,
            system_prompt="Perform a thorough critique of this summary. "
            "Analyze accuracy, completeness, structure, and suggest improvements.",
            input_key="summarize",
            output_key="critique",
        )
    )

    # Conditional routing based on summary length
    pipeline.add_conditional_edge(
        "summarize",
        route_by_length(500),  # Threshold at 500 characters
        {
            "short": "quick_critique",
            "long": "deep_critique",
        },
    )

    pipeline.set_entry_point("summarize")
    pipeline.set_finish_point("quick_critique")
    pipeline.set_finish_point("deep_critique")

    return pipeline


# Quick test function
def test_pipeline_structure():
    """Test that the pipeline builds correctly (no API calls)."""
    pipeline = create_summarize_critique_pipeline(summarizer_llm="gpt4o", critic_llm="claude")

    # Check structure
    assert len(pipeline.nodes) == 2
    assert "summarize" in pipeline.nodes
    assert "critique" in pipeline.nodes
    assert pipeline.entry_point == "summarize"
    assert "critique" in pipeline.finish_points

    # Check node configs
    assert pipeline.nodes["summarize"].llm == "gpt4o"
    assert pipeline.nodes["critique"].llm == "claude"

    # Test compilation (doesn't call APIs)
    pipeline.compile()
    assert pipeline._compiled

    # Test visualization
    viz = pipeline.visualize()
    assert "summarize" in viz
    assert "critique" in viz

    print("Pipeline structure test passed!")
    print("\nVisualization:")
    print(viz)

    return pipeline


if __name__ == "__main__":
    test_pipeline_structure()
