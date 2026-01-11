"""
Multi-Model Code Review Pipeline.

Demonstrates using multiple LLMs to review code from different perspectives:
- Security review
- Performance review
- Style/maintainability review
- Aggregated final review

Usage:
    from ai_orchestrator.pipelines.examples import create_code_review_pipeline

    pipeline = create_code_review_pipeline()
    result = pipeline.run('''
    def process_user_input(user_data):
        query = f"SELECT * FROM users WHERE id = {user_data['id']}"
        return db.execute(query)
    ''')

    print(result.final_output)  # Aggregated review
"""

from ..base import BasePipeline
from ..nodes import NodeConfig, NodeType


def create_code_review_pipeline(
    security_llm: str = "claude",
    performance_llm: str = "gpt4o",
    style_llm: str = "claude",
    aggregator_llm: str = "gpt4o",
    name: str = "code_review",
) -> BasePipeline:
    """
    Create a multi-perspective code review pipeline.

    This pipeline runs three parallel reviews (security, performance, style)
    then aggregates them into a comprehensive final review.

    Args:
        security_llm: LLM for security analysis
        performance_llm: LLM for performance analysis
        style_llm: LLM for style/maintainability analysis
        aggregator_llm: LLM for aggregating all reviews
        name: Pipeline name

    Returns:
        Configured BasePipeline ready to run

    Example:
        >>> pipeline = create_code_review_pipeline()
        >>> result = pipeline.run("def my_function(): ...")
        >>> print(result.node_outputs["security"]["content"])  # Security review
        >>> print(result.final_output)  # Aggregated review
    """
    pipeline = BasePipeline(
        name=name,
        description="Multi-perspective code review: security, performance, style -> aggregate",
    )

    # Node 1: Security Review
    pipeline.add_node(
        NodeConfig(
            name="security",
            node_type=NodeType.LLM,
            llm=security_llm,
            system_prompt=(
                "You are a security expert reviewing code for vulnerabilities.\n\n"
                "Focus on:\n"
                "- SQL injection vulnerabilities\n"
                "- XSS vulnerabilities\n"
                "- Authentication/authorization issues\n"
                "- Input validation problems\n"
                "- Sensitive data exposure\n"
                "- Insecure dependencies\n"
                "- OWASP Top 10 vulnerabilities\n\n"
                "For each issue found:\n"
                "1. Describe the vulnerability\n"
                "2. Rate severity (Critical/High/Medium/Low)\n"
                "3. Provide a fix recommendation\n\n"
                "If no issues found, explicitly state the code is secure."
            ),
            input_key="input",
            output_key="security",
            temperature=0.2,
            max_tokens=1500,
            description="Security vulnerability analysis",
        )
    )

    # Node 2: Performance Review
    pipeline.add_node(
        NodeConfig(
            name="performance",
            node_type=NodeType.LLM,
            llm=performance_llm,
            system_prompt=(
                "You are a performance optimization expert reviewing code.\n\n"
                "Focus on:\n"
                "- Time complexity (Big O analysis)\n"
                "- Space complexity\n"
                "- Unnecessary iterations or operations\n"
                "- Memory leaks or inefficient memory usage\n"
                "- Database query optimization\n"
                "- Caching opportunities\n"
                "- Async/parallel processing opportunities\n\n"
                "For each issue found:\n"
                "1. Describe the performance impact\n"
                "2. Rate impact (High/Medium/Low)\n"
                "3. Suggest optimization\n\n"
                "Include overall performance rating."
            ),
            input_key="input",
            output_key="performance",
            temperature=0.2,
            max_tokens=1500,
            description="Performance optimization analysis",
        )
    )

    # Node 3: Style Review
    pipeline.add_node(
        NodeConfig(
            name="style",
            node_type=NodeType.LLM,
            llm=style_llm,
            system_prompt=(
                "You are a code quality expert focusing on maintainability.\n\n"
                "Focus on:\n"
                "- Code clarity and readability\n"
                "- Naming conventions\n"
                "- Function/method length and complexity\n"
                "- DRY principle adherence\n"
                "- SOLID principles\n"
                "- Error handling patterns\n"
                "- Documentation and comments\n"
                "- Type hints (for Python)\n"
                "- Test coverage considerations\n\n"
                "For each suggestion:\n"
                "1. Describe the issue\n"
                "2. Explain why it matters\n"
                "3. Show improved version if applicable\n\n"
                "Include overall maintainability score (1-10)."
            ),
            input_key="input",
            output_key="style",
            temperature=0.3,
            max_tokens=1500,
            description="Style and maintainability analysis",
        )
    )

    # Node 4: Aggregate Reviews
    pipeline.add_node(
        NodeConfig(
            name="aggregate",
            node_type=NodeType.AGGREGATE,
            llm=aggregator_llm,
            system_prompt=(
                "You are a lead code reviewer synthesizing multiple review perspectives.\n\n"
                "You will receive three reviews: security, performance, and style.\n\n"
                "Create a comprehensive code review summary:\n\n"
                "1. CRITICAL ISSUES (must fix before merge)\n"
                "   - List all security and high-impact issues\n\n"
                "2. RECOMMENDED IMPROVEMENTS\n"
                "   - Performance optimizations\n"
                "   - Important style fixes\n\n"
                "3. SUGGESTIONS (nice-to-have)\n"
                "   - Minor improvements\n\n"
                "4. POSITIVE ASPECTS\n"
                "   - What's done well\n\n"
                "5. OVERALL VERDICT\n"
                "   - Approve / Approve with changes / Request changes\n"
                "   - Summary recommendation"
            ),
            input_key="input",  # Will receive aggregated outputs
            output_key="aggregate",
            aggregate_keys=["security", "performance", "style"],
            temperature=0.3,
            max_tokens=2000,
            description="Aggregates all reviews into final summary",
        )
    )

    # All reviews run from input, then aggregate
    pipeline.add_edge("security", "aggregate")
    pipeline.add_edge("performance", "aggregate")
    pipeline.add_edge("style", "aggregate")

    # Set entry point (all three reviews start from input)
    pipeline.set_entry_point("security")
    pipeline.set_entry_point("performance")
    pipeline.set_entry_point("style")
    pipeline.set_finish_point("aggregate")

    return pipeline


def create_simple_code_review_pipeline(
    llm: str = "claude",
    name: str = "simple_code_review",
) -> BasePipeline:
    """
    Create a simple single-LLM code review pipeline.

    Good for quick reviews when you don't need multi-perspective analysis.

    Args:
        llm: LLM to use for review
        name: Pipeline name

    Returns:
        Single-node code review pipeline
    """
    pipeline = BasePipeline(
        name=name,
        description="Quick single-pass code review",
    )

    pipeline.add_node(
        NodeConfig(
            name="review",
            node_type=NodeType.LLM,
            llm=llm,
            system_prompt=(
                "You are a senior developer performing a code review.\n\n"
                "Review the code for:\n"
                "- Security issues\n"
                "- Performance problems\n"
                "- Code style and maintainability\n"
                "- Potential bugs\n\n"
                "Format your review as:\n"
                "1. ISSUES (with severity)\n"
                "2. SUGGESTIONS\n"
                "3. POSITIVE NOTES\n"
                "4. VERDICT (Approve/Request changes)"
            ),
            input_key="input",
            output_key="review",
            temperature=0.3,
            max_tokens=1500,
        )
    )

    pipeline.set_entry_point("review")
    pipeline.set_finish_point("review")

    return pipeline


if __name__ == "__main__":
    # Test pipeline structure
    pipeline = create_code_review_pipeline()

    assert len(pipeline.nodes) == 4
    assert "security" in pipeline.nodes
    assert "performance" in pipeline.nodes
    assert "style" in pipeline.nodes
    assert "aggregate" in pipeline.nodes

    pipeline.compile()
    print("Code review pipeline structure test passed!")
    print("\nVisualization:")
    print(pipeline.visualize())
