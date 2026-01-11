"""
Deploy module for AI Orchestrator.

Provides deployment capabilities with two paths:
- Functional Tools -> Streamlit Cloud
- Professional Products -> Vercel (Next.js 14)

Components:
- DesignSystem: Next.js project scaffolding
- ComponentLibrary: Pre-built React components
- ProductRegistry: Track deployed products
- IterationLoop: Feedback-driven improvements
"""

from .design_system import (
    DesignSystem,
    ProjectConfig,
    create_nextjs_project,
)

from .components import (
    ComponentLibrary,
    ComponentCategory,
    ComponentInfo,
    get_component,
    list_components,
)

from .registry import (
    ProductRegistry,
    ProductEntry,
    ProductType,
    ProductStatus,
    DeploymentRecord,
    get_product_registry,
    reset_product_registry,
)

from .iteration import (
    IterationLoop,
    FeedbackEntry,
    FeedbackType,
    FeedbackPriority,
    ImprovementPlan,
    ImprovementItem,
    IterationRecord,
    IterationStatus,
    get_iteration_loop,
    reset_iteration_loop,
)


__all__ = [
    # Design System
    "DesignSystem",
    "ProjectConfig",
    "create_nextjs_project",
    # Component Library
    "ComponentLibrary",
    "ComponentCategory",
    "ComponentInfo",
    "get_component",
    "list_components",
    # Product Registry
    "ProductRegistry",
    "ProductEntry",
    "ProductType",
    "ProductStatus",
    "DeploymentRecord",
    "get_product_registry",
    "reset_product_registry",
    # Iteration Loop
    "IterationLoop",
    "FeedbackEntry",
    "FeedbackType",
    "FeedbackPriority",
    "ImprovementPlan",
    "ImprovementItem",
    "IterationRecord",
    "IterationStatus",
    "get_iteration_loop",
    "reset_iteration_loop",
]
