"""
Research and Synthesis Pipeline.

Demonstrates using different LLMs for their strengths:
- Perplexity for web research (if available)
- Claude for synthesis and analysis
- GPT-4 for final polish

Usage:
    from ai_orchestrator.pipelines.examples import create_research_pipeline

    pipeline = create_research_pipeline(topic="climate change solutions")
    result = pipeline.run("What are the most promising climate change solutions in 2024?")

    # Access individual steps
    print(result.node_outputs["research"]["content"])   # Research findings
    print(result.node_outputs["synthesize"]["content"]) # Synthesis
    print(result.final_output)                          # Final polished output
"""

from ..base import BasePipeline
from ..nodes import NodeConfig, NodeType


def create_research_pipeline(
    researcher_llm: str = "perplexity",
    synthesizer_llm: str = "claude",
    editor_llm: str = "gpt4o",
    topic: str = "technology",
    name: str = "research_synthesize",
) -> BasePipeline:
    """
    Create a research and synthesis pipeline.

    This pipeline:
    1. Uses Perplexity (or another LLM) to gather research on a topic
    2. Uses Claude to synthesize and analyze the findings
    3. Uses GPT-4 to polish and format the final output

    Args:
        researcher_llm: LLM for initial research (perplexity recommended)
        synthesizer_llm: LLM for synthesis and analysis
        editor_llm: LLM for final editing and polish
        topic: The general topic area for research context
        name: Pipeline name

    Returns:
        Configured BasePipeline ready to run

    Example:
        >>> pipeline = create_research_pipeline(topic="AI safety")
        >>> result = pipeline.run("What are current approaches to AI alignment?")
        >>> print(result.final_output)
    """
    pipeline = BasePipeline(
        name=name,
        description=f"Research pipeline for {topic}: gather -> synthesize -> polish",
    )

    # Node 1: Research
    pipeline.add_node(
        NodeConfig(
            name="research",
            node_type=NodeType.LLM,
            llm=researcher_llm,
            system_prompt=(
                f"You are a research assistant specializing in {topic}. "
                "Your task is to provide comprehensive, factual information about the query.\n\n"
                "Guidelines:\n"
                "- Focus on recent, credible information\n"
                "- Include specific examples, statistics, and sources when possible\n"
                "- Cover multiple perspectives and approaches\n"
                "- Organize information by subtopic\n"
                "- Note any areas of uncertainty or debate"
            ),
            input_key="input",
            output_key="research",
            temperature=0.3,  # Lower temperature for factual research
            max_tokens=2000,
            description="Gathers research on the topic",
        )
    )

    # Node 2: Synthesize
    pipeline.add_node(
        NodeConfig(
            name="synthesize",
            node_type=NodeType.LLM,
            llm=synthesizer_llm,
            system_prompt=(
                "You are an analytical synthesizer. Your task is to take research findings "
                "and create a coherent, insightful analysis.\n\n"
                "Your synthesis should:\n"
                "- Identify key themes and patterns\n"
                "- Draw connections between different points\n"
                "- Highlight the most important findings\n"
                "- Provide context and implications\n"
                "- Note any gaps or areas needing further research\n\n"
                "Structure your synthesis with clear sections and logical flow."
            ),
            input_key="research",
            output_key="synthesize",
            temperature=0.5,
            max_tokens=1500,
            description="Synthesizes research into coherent analysis",
        )
    )

    # Node 3: Polish
    pipeline.add_node(
        NodeConfig(
            name="polish",
            node_type=NodeType.LLM,
            llm=editor_llm,
            system_prompt=(
                "You are an expert editor. Your task is to polish and refine the synthesis "
                "into a clear, professional final document.\n\n"
                "Focus on:\n"
                "- Clear, engaging writing\n"
                "- Logical structure and flow\n"
                "- Appropriate formatting (headers, bullets, etc.)\n"
                "- Removing redundancy\n"
                "- Adding a brief executive summary at the start\n"
                "- Ensuring all claims are supported by the research"
            ),
            input_key="synthesize",
            output_key="polish",
            temperature=0.4,
            max_tokens=2000,
            description="Polishes synthesis into final document",
        )
    )

    # Connect nodes sequentially
    pipeline.add_edge("research", "synthesize")
    pipeline.add_edge("synthesize", "polish")

    # Set entry and exit
    pipeline.set_entry_point("research")
    pipeline.set_finish_point("polish")

    return pipeline


def create_quick_research_pipeline(
    llm: str = "claude",
    name: str = "quick_research",
) -> BasePipeline:
    """
    Create a simple single-LLM research pipeline for quick queries.

    Args:
        llm: LLM to use for research
        name: Pipeline name

    Returns:
        Single-node research pipeline
    """
    pipeline = BasePipeline(
        name=name,
        description="Quick single-step research",
    )

    pipeline.add_node(
        NodeConfig(
            name="research",
            node_type=NodeType.LLM,
            llm=llm,
            system_prompt=(
                "You are a knowledgeable research assistant. Provide a comprehensive "
                "but concise answer to the query, including:\n"
                "- Key facts and figures\n"
                "- Recent developments\n"
                "- Multiple perspectives if relevant\n"
                "Format your response clearly with sections if appropriate."
            ),
            input_key="input",
            output_key="research",
            temperature=0.4,
            max_tokens=1500,
        )
    )

    pipeline.set_entry_point("research")
    pipeline.set_finish_point("research")

    return pipeline


if __name__ == "__main__":
    # Test pipeline structure
    pipeline = create_research_pipeline(topic="renewable energy")

    assert len(pipeline.nodes) == 3
    assert "research" in pipeline.nodes
    assert "synthesize" in pipeline.nodes
    assert "polish" in pipeline.nodes

    pipeline.compile()
    print("Research pipeline structure test passed!")
    print("\nVisualization:")
    print(pipeline.visualize())
