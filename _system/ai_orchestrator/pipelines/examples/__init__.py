"""
Example pipelines for testing and demonstration.

Available pipelines:
- Summarize & Critique: Two-stage text summarization with critique
- Research & Synthesize: Multi-LLM research and analysis pipeline
- Code Review: Multi-perspective code review (security, performance, style)
- Content Generation: Draft -> Edit -> Polish content pipeline

Usage:
    from ai_orchestrator.pipelines.examples import (
        create_summarize_critique_pipeline,
        create_research_pipeline,
        create_code_review_pipeline,
        create_content_pipeline,
    )

    # Create and run a pipeline
    pipeline = create_research_pipeline(topic="AI")
    result = pipeline.run("What are the latest developments in AI?")
    print(result.final_output)
"""

from .summarize_critique import (
    create_summarize_critique_pipeline,
    create_summarize_critique_conditional,
)

from .research_synthesize import (
    create_research_pipeline,
    create_quick_research_pipeline,
)

from .code_review import (
    create_code_review_pipeline,
    create_simple_code_review_pipeline,
)

from .content_pipeline import (
    create_content_pipeline,
    create_email_pipeline,
    create_social_media_pipeline,
)

__all__ = [
    # Summarize & Critique
    "create_summarize_critique_pipeline",
    "create_summarize_critique_conditional",
    # Research
    "create_research_pipeline",
    "create_quick_research_pipeline",
    # Code Review
    "create_code_review_pipeline",
    "create_simple_code_review_pipeline",
    # Content
    "create_content_pipeline",
    "create_email_pipeline",
    "create_social_media_pipeline",
]
