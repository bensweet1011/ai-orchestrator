"""
Pipeline registry for storing, loading, and managing pipeline definitions.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from ..config import TEMPLATES_DIR
from .base import BasePipeline
from .nodes import route_by_length, route_by_keyword


# Default templates directory for pipelines
PIPELINES_DIR = TEMPLATES_DIR / "pipelines"


def ensure_pipelines_dir() -> Path:
    """Ensure pipelines directory exists."""
    PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
    return PIPELINES_DIR


# Built-in condition registry
CONDITION_REGISTRY: Dict[str, Callable] = {
    "route_by_length_500": route_by_length(500),
    "route_by_length_1000": route_by_length(1000),
    "route_by_keyword_sentiment": route_by_keyword(
        {"positive": ["good", "great", "excellent"], "negative": ["bad", "poor", "terrible"]},
        default="neutral",
    ),
}


def register_condition(name: str, condition: Callable) -> None:
    """Register a custom condition function."""
    CONDITION_REGISTRY[name] = condition


def list_pipelines() -> List[Dict[str, Any]]:
    """
    List all available pipelines.

    Returns:
        List of pipeline summaries with name, description, path
    """
    ensure_pipelines_dir()
    pipelines = []

    for file_path in PIPELINES_DIR.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                pipelines.append(
                    {
                        "name": data.get("name", file_path.stem),
                        "description": data.get("description", ""),
                        "path": str(file_path),
                        "nodes": list(data.get("nodes", {}).keys()),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    }
                )
        except (json.JSONDecodeError, IOError):
            continue

    return sorted(pipelines, key=lambda p: p["name"])


def get_pipeline(name: str) -> Optional[BasePipeline]:
    """
    Load a pipeline by name.

    Args:
        name: Pipeline name (without .json extension)

    Returns:
        BasePipeline instance or None if not found
    """
    ensure_pipelines_dir()
    file_path = PIPELINES_DIR / f"{name}.json"

    if not file_path.exists():
        return None

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return BasePipeline.from_dict(data, condition_registry=CONDITION_REGISTRY)
    except (json.JSONDecodeError, IOError, ValueError):
        return None


def save_pipeline(pipeline: BasePipeline, overwrite: bool = False) -> str:
    """
    Save a pipeline definition to the registry.

    Args:
        pipeline: Pipeline to save
        overwrite: Whether to overwrite existing pipeline

    Returns:
        Path to saved file
    """
    ensure_pipelines_dir()
    file_path = PIPELINES_DIR / f"{pipeline.name}.json"

    if file_path.exists() and not overwrite:
        raise ValueError(
            f"Pipeline '{pipeline.name}' already exists. Use overwrite=True to replace."
        )

    data = pipeline.to_dict()
    data["updated_at"] = datetime.utcnow().isoformat()
    if not file_path.exists():
        data["created_at"] = data["updated_at"]
    else:
        # Preserve original creation time
        try:
            with open(file_path, "r") as f:
                existing = json.load(f)
                data["created_at"] = existing.get("created_at", data["updated_at"])
        except (json.JSONDecodeError, IOError):
            data["created_at"] = data["updated_at"]

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(file_path)


def delete_pipeline(name: str) -> bool:
    """
    Delete a pipeline from the registry.

    Args:
        name: Pipeline name

    Returns:
        True if deleted, False if not found
    """
    file_path = PIPELINES_DIR / f"{name}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def duplicate_pipeline(source_name: str, new_name: str) -> Optional[str]:
    """
    Duplicate an existing pipeline with a new name.

    Args:
        source_name: Name of pipeline to duplicate
        new_name: Name for the new pipeline

    Returns:
        Path to new pipeline file or None if source not found
    """
    pipeline = get_pipeline(source_name)
    if not pipeline:
        return None

    pipeline.name = new_name
    pipeline.pipeline_id = pipeline._generate_id()
    return save_pipeline(pipeline)


def export_pipeline(name: str) -> Optional[str]:
    """
    Export pipeline as JSON string.

    Args:
        name: Pipeline name

    Returns:
        JSON string or None if not found
    """
    file_path = PIPELINES_DIR / f"{name}.json"
    if not file_path.exists():
        return None

    with open(file_path, "r") as f:
        return f.read()


def import_pipeline(json_str: str, name: Optional[str] = None) -> str:
    """
    Import pipeline from JSON string.

    Args:
        json_str: JSON pipeline definition
        name: Optional name override

    Returns:
        Name of imported pipeline
    """
    data = json.loads(json_str)
    if name:
        data["name"] = name

    pipeline = BasePipeline.from_dict(data, condition_registry=CONDITION_REGISTRY)
    save_pipeline(pipeline, overwrite=True)
    return pipeline.name


def validate_pipeline_definition(data: Dict[str, Any]) -> List[str]:
    """
    Validate a pipeline definition dictionary.

    Args:
        data: Pipeline definition to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not data.get("name"):
        errors.append("Pipeline must have a name")

    if not data.get("nodes"):
        errors.append("Pipeline must have at least one node")

    if not data.get("entry_point"):
        errors.append("Pipeline must have an entry point")

    if not data.get("finish_points"):
        errors.append("Pipeline must have at least one finish point")

    # Validate nodes
    for node_name, node_data in data.get("nodes", {}).items():
        node_type = node_data.get("node_type", "llm")
        if node_type == "llm" and not node_data.get("llm"):
            errors.append(f"LLM node '{node_name}' must specify an LLM model")

    # Validate entry point exists
    entry = data.get("entry_point")
    if entry and entry not in data.get("nodes", {}):
        errors.append(f"Entry point '{entry}' not found in nodes")

    # Validate finish points exist
    for fp in data.get("finish_points", []):
        if fp not in data.get("nodes", {}):
            errors.append(f"Finish point '{fp}' not found in nodes")

    # Validate edges reference existing nodes
    for edge in data.get("edges", []):
        if edge.get("from") not in data.get("nodes", {}):
            errors.append(f"Edge from '{edge.get('from')}' references non-existent node")
        if edge.get("to") not in data.get("nodes", {}):
            errors.append(f"Edge to '{edge.get('to')}' references non-existent node")

    return errors


def get_pipeline_templates() -> List[Dict[str, Any]]:
    """
    Get built-in pipeline templates for common use cases.

    Returns:
        List of template definitions
    """
    return [
        {
            "name": "summarize_critique",
            "description": "Summarize text then critique the summary",
            "template": {
                "name": "summarize_critique",
                "description": "Two-stage pipeline: summarize then critique",
                "nodes": {
                    "summarize": {
                        "name": "summarize",
                        "node_type": "llm",
                        "llm": None,  # User must select
                        "system_prompt": "Summarize the following text concisely, "
                        "capturing the key points in 2-3 paragraphs.",
                        "input_key": "input",
                        "output_key": "summarize",
                    },
                    "critique": {
                        "name": "critique",
                        "node_type": "llm",
                        "llm": None,  # User must select
                        "system_prompt": "Critique the following summary. "
                        "Evaluate accuracy, completeness, and clarity. "
                        "Suggest improvements if needed.",
                        "input_key": "summarize",
                        "output_key": "critique",
                    },
                },
                "edges": [{"from": "summarize", "to": "critique"}],
                "conditional_edges": [],
                "entry_point": "summarize",
                "finish_points": ["critique"],
            },
        },
        {
            "name": "research_synthesize",
            "description": "Research a topic then synthesize findings",
            "template": {
                "name": "research_synthesize",
                "description": "Research with one LLM, synthesize with another",
                "nodes": {
                    "research": {
                        "name": "research",
                        "node_type": "llm",
                        "llm": None,
                        "system_prompt": "Research the following topic thoroughly. "
                        "Provide detailed findings with multiple perspectives.",
                        "input_key": "input",
                        "output_key": "research",
                    },
                    "synthesize": {
                        "name": "synthesize",
                        "node_type": "llm",
                        "llm": None,
                        "system_prompt": "Synthesize the research findings into a clear, "
                        "actionable summary. Highlight key insights and recommendations.",
                        "input_key": "research",
                        "output_key": "synthesize",
                    },
                },
                "edges": [{"from": "research", "to": "synthesize"}],
                "conditional_edges": [],
                "entry_point": "research",
                "finish_points": ["synthesize"],
            },
        },
        {
            "name": "draft_review_refine",
            "description": "Three-stage writing: draft, review, refine",
            "template": {
                "name": "draft_review_refine",
                "description": "Multi-stage writing pipeline",
                "nodes": {
                    "draft": {
                        "name": "draft",
                        "node_type": "llm",
                        "llm": None,
                        "system_prompt": "Write a first draft based on the following prompt. "
                        "Focus on getting ideas down without over-editing.",
                        "input_key": "input",
                        "output_key": "draft",
                    },
                    "review": {
                        "name": "review",
                        "node_type": "llm",
                        "llm": None,
                        "system_prompt": "Review the following draft critically. "
                        "Identify strengths, weaknesses, and specific areas for improvement. "
                        "Be constructive but thorough.",
                        "input_key": "draft",
                        "output_key": "review",
                    },
                    "refine": {
                        "name": "refine",
                        "node_type": "llm",
                        "llm": None,
                        "input_template": "Original draft:\n{draft}\n\nReview feedback:\n{review}\n\n"
                        "Refine the draft based on the feedback.",
                        "system_prompt": "Improve the draft based on the review feedback. "
                        "Produce a polished final version.",
                        "input_key": "input",
                        "output_key": "refine",
                    },
                },
                "edges": [
                    {"from": "draft", "to": "review"},
                    {"from": "review", "to": "refine"},
                ],
                "conditional_edges": [],
                "entry_point": "draft",
                "finish_points": ["refine"],
            },
        },
    ]


def create_from_template(
    template_name: str, llm_assignments: Dict[str, str], new_name: Optional[str] = None
) -> BasePipeline:
    """
    Create a pipeline from a template with LLM assignments.

    Args:
        template_name: Name of the template
        llm_assignments: Mapping of node names to LLM models
        new_name: Optional custom name for the pipeline

    Returns:
        Configured BasePipeline instance
    """
    templates = {t["name"]: t["template"] for t in get_pipeline_templates()}

    if template_name not in templates:
        raise ValueError(
            f"Template '{template_name}' not found. " f"Available: {list(templates.keys())}"
        )

    data = templates[template_name].copy()
    data["nodes"] = {k: dict(v) for k, v in data["nodes"].items()}

    # Assign LLMs to nodes
    for node_name, llm in llm_assignments.items():
        if node_name in data["nodes"]:
            data["nodes"][node_name]["llm"] = llm

    # Check all LLM nodes have assignments
    for node_name, node_data in data["nodes"].items():
        if node_data.get("node_type", "llm") == "llm" and not node_data.get("llm"):
            raise ValueError(
                f"Node '{node_name}' requires an LLM assignment. "
                f"Provide llm_assignments={{'{node_name}': 'model_name'}}"
            )

    if new_name:
        data["name"] = new_name

    return BasePipeline.from_dict(data, condition_registry=CONDITION_REGISTRY)
