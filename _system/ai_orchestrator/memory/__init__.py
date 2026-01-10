"""
Memory module for cross-session persistence and synthesis.
Provides Pinecone-backed storage, context injection, and multi-run synthesis.
"""

from .state import (
    MemoryType,
    MemoryScope,
    MemoryEntry,
    PipelineMemory,
    SynthesisRequest,
    SynthesisResult,
    MemoryNamespace,
    ContextConfig,
    MemorySearchResult,
)

from .store import (
    MemoryStore,
    get_memory_store,
)

from .context import (
    ContextInjector,
    ContextAwareNode,
    InjectedContext,
    get_context_injector,
)

from .synthesis import (
    SynthesisType,
    Synthesizer,
    ComparisonResult,
    get_synthesizer,
)

__all__ = [
    # State types
    "MemoryType",
    "MemoryScope",
    "MemoryEntry",
    "PipelineMemory",
    "SynthesisRequest",
    "SynthesisResult",
    "MemoryNamespace",
    "ContextConfig",
    "MemorySearchResult",
    # Store
    "MemoryStore",
    "get_memory_store",
    # Context injection
    "ContextInjector",
    "ContextAwareNode",
    "InjectedContext",
    "get_context_injector",
    # Synthesis
    "SynthesisType",
    "Synthesizer",
    "ComparisonResult",
    "get_synthesizer",
]
