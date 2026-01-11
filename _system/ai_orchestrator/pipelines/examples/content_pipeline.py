"""
Content Generation Pipeline.

Demonstrates a content creation workflow:
- Draft generation
- Editing and improvement
- Final polish with formatting

Usage:
    from ai_orchestrator.pipelines.examples import create_content_pipeline

    pipeline = create_content_pipeline(content_type="blog post")
    result = pipeline.run("Write about the benefits of remote work")

    print(result.node_outputs["draft"]["content"])  # Initial draft
    print(result.node_outputs["edit"]["content"])   # Edited version
    print(result.final_output)                      # Final polished content
"""

from ..base import BasePipeline
from ..nodes import NodeConfig, NodeType


def create_content_pipeline(
    drafter_llm: str = "claude",
    editor_llm: str = "gpt4o",
    polisher_llm: str = "claude",
    content_type: str = "article",
    name: str = "content_generation",
) -> BasePipeline:
    """
    Create a content generation pipeline with drafting, editing, and polishing.

    Args:
        drafter_llm: LLM for initial draft creation
        editor_llm: LLM for editing and improvements
        polisher_llm: LLM for final polish
        content_type: Type of content (article, blog post, email, etc.)
        name: Pipeline name

    Returns:
        Configured BasePipeline ready to run

    Example:
        >>> pipeline = create_content_pipeline(content_type="newsletter")
        >>> result = pipeline.run("Topic: New product launch")
        >>> print(result.final_output)
    """
    pipeline = BasePipeline(
        name=name,
        description=f"{content_type.title()} generation: draft -> edit -> polish",
    )

    # Node 1: Draft
    pipeline.add_node(
        NodeConfig(
            name="draft",
            node_type=NodeType.LLM,
            llm=drafter_llm,
            system_prompt=(
                f"You are a skilled content writer creating a {content_type}.\n\n"
                "Your task is to create an engaging first draft based on the topic.\n\n"
                "Guidelines:\n"
                "- Start with a compelling hook\n"
                "- Develop ideas logically\n"
                "- Include relevant examples\n"
                "- Maintain consistent tone\n"
                "- End with a clear conclusion or call-to-action\n\n"
                f"Write the complete {content_type}. Don't include meta-commentary "
                "about the writing process."
            ),
            input_key="input",
            output_key="draft",
            temperature=0.7,  # Higher for creative writing
            max_tokens=2000,
            description=f"Creates initial {content_type} draft",
        )
    )

    # Node 2: Edit
    pipeline.add_node(
        NodeConfig(
            name="edit",
            node_type=NodeType.LLM,
            llm=editor_llm,
            system_prompt=(
                "You are an experienced editor improving content.\n\n"
                "Your task is to edit and improve the draft while preserving the author's voice.\n\n"
                "Focus on:\n"
                "- Clarity: Simplify complex sentences\n"
                "- Flow: Improve transitions between paragraphs\n"
                "- Engagement: Strengthen the hook and conclusion\n"
                "- Accuracy: Fix any factual inconsistencies\n"
                "- Conciseness: Remove unnecessary words\n"
                "- Variety: Vary sentence length and structure\n\n"
                "Output the improved version directly. Don't include editing notes."
            ),
            input_key="draft",
            output_key="edit",
            temperature=0.4,
            max_tokens=2000,
            description="Edits and improves the draft",
        )
    )

    # Node 3: Polish
    pipeline.add_node(
        NodeConfig(
            name="polish",
            node_type=NodeType.LLM,
            llm=polisher_llm,
            system_prompt=(
                "You are a final-stage editor polishing content for publication.\n\n"
                "Your task is to:\n"
                "1. Fix any remaining grammar/spelling issues\n"
                "2. Add appropriate formatting:\n"
                "   - Headings and subheadings where helpful\n"
                "   - Bullet points for lists\n"
                "   - Bold for emphasis on key points\n"
                "3. Ensure consistent formatting throughout\n"
                "4. Add a brief summary or key takeaways if appropriate\n\n"
                "Output the final, publication-ready content."
            ),
            input_key="edit",
            output_key="polish",
            temperature=0.3,
            max_tokens=2000,
            description="Final polish and formatting",
        )
    )

    # Connect nodes sequentially
    pipeline.add_edge("draft", "edit")
    pipeline.add_edge("edit", "polish")

    # Set entry and exit
    pipeline.set_entry_point("draft")
    pipeline.set_finish_point("polish")

    return pipeline


def create_email_pipeline(
    llm: str = "claude",
    tone: str = "professional",
    name: str = "email_generator",
) -> BasePipeline:
    """
    Create a simple email generation pipeline.

    Args:
        llm: LLM to use
        tone: Email tone (professional, friendly, formal, casual)
        name: Pipeline name

    Returns:
        Single-node email pipeline
    """
    pipeline = BasePipeline(
        name=name,
        description=f"Generate {tone} emails",
    )

    pipeline.add_node(
        NodeConfig(
            name="email",
            node_type=NodeType.LLM,
            llm=llm,
            system_prompt=(
                f"You are an expert at writing {tone} emails.\n\n"
                "Based on the context provided, write a complete email including:\n"
                "- Subject line (prefixed with 'Subject:')\n"
                "- Appropriate greeting\n"
                "- Clear, concise body\n"
                "- Professional closing\n\n"
                f"Maintain a {tone} tone throughout."
            ),
            input_key="input",
            output_key="email",
            temperature=0.5,
            max_tokens=1000,
        )
    )

    pipeline.set_entry_point("email")
    pipeline.set_finish_point("email")

    return pipeline


def create_social_media_pipeline(
    llm: str = "gpt4o",
    platform: str = "twitter",
    name: str = "social_media_generator",
) -> BasePipeline:
    """
    Create a social media content pipeline.

    Args:
        llm: LLM to use
        platform: Target platform (twitter, linkedin, instagram)
        name: Pipeline name

    Returns:
        Social media content pipeline
    """
    platform_configs = {
        "twitter": {
            "char_limit": 280,
            "style": "concise, engaging, use relevant hashtags",
        },
        "linkedin": {
            "char_limit": 3000,
            "style": "professional, insightful, use line breaks for readability",
        },
        "instagram": {
            "char_limit": 2200,
            "style": "visual-focused, use emojis, include relevant hashtags at end",
        },
    }

    config = platform_configs.get(platform, platform_configs["twitter"])

    pipeline = BasePipeline(
        name=name,
        description=f"Generate {platform} posts",
    )

    pipeline.add_node(
        NodeConfig(
            name="post",
            node_type=NodeType.LLM,
            llm=llm,
            system_prompt=(
                f"You are a social media expert creating content for {platform}.\n\n"
                f"Character limit: {config['char_limit']}\n"
                f"Style: {config['style']}\n\n"
                "Create an engaging post based on the topic provided.\n"
                "Optimize for engagement and reach."
            ),
            input_key="input",
            output_key="post",
            temperature=0.7,
            max_tokens=500,
        )
    )

    pipeline.set_entry_point("post")
    pipeline.set_finish_point("post")

    return pipeline


if __name__ == "__main__":
    # Test pipeline structure
    pipeline = create_content_pipeline(content_type="blog post")

    assert len(pipeline.nodes) == 3
    assert "draft" in pipeline.nodes
    assert "edit" in pipeline.nodes
    assert "polish" in pipeline.nodes

    pipeline.compile()
    print("Content pipeline structure test passed!")
    print("\nVisualization:")
    print(pipeline.visualize())
