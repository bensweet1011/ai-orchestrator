"""
Memory management with Pinecone.
Auto-logs all LLM interactions, provides semantic retrieval.
"""

import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from pinecone import Pinecone

from .llm_clients import LLMResponse


@dataclass
class InteractionLog:
    """Record of a single LLM interaction."""
    id: str
    timestamp: str
    session_id: str
    project: str
    task: str
    model: str
    provider: str
    prompt: str
    response: str
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    rating: Optional[int] = None
    notes: Optional[str] = None
    metadata: Optional[Dict] = None


class Memory:
    """
    Handles all memory operations:
    - Auto-logging LLM calls to Pinecone
    - Semantic search for relevant context
    - Project history retrieval
    - Feedback/rating storage
    """
    
    def __init__(self, index_name: str = "llm-logs"):
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        
        # Session tracking
        self.session_id = self._generate_id("session")
        self.current_project = "default"
        self.current_task = "general"
    
    def _generate_id(self, prefix: str = "log") -> str:
        """Generate unique ID."""
        ts = datetime.utcnow().isoformat()
        unique = hashlib.md5(f"{ts}{time.time_ns()}".encode()).hexdigest()[:8]
        return f"{prefix}_{unique}"
    
    def set_context(self, project: str = None, task: str = None):
        """Set current project and task context."""
        if project:
            self.current_project = project
        if task:
            self.current_task = task
    
    def new_session(self) -> str:
        """Start a new session."""
        self.session_id = self._generate_id("session")
        return self.session_id
    
    def log(
        self,
        prompt: str,
        response: LLMResponse,
        latency_ms: int,
        project: str = None,
        task: str = None,
        metadata: Dict = None
    ) -> str:
        """
        Log an LLM interaction to Pinecone.
        Returns the log ID.
        """
        log_id = self._generate_id("log")
        timestamp = datetime.utcnow().isoformat()
        
        # Create log record
        log = InteractionLog(
            id=log_id,
            timestamp=timestamp,
            session_id=self.session_id,
            project=project or self.current_project,
            task=task or self.current_task,
            model=response.model,
            provider=response.provider,
            prompt=prompt,
            response=response.content,
            prompt_tokens=response.usage.get("input_tokens") if response.usage else None,
            response_tokens=response.usage.get("output_tokens") if response.usage else None,
            latency_ms=latency_ms,
            metadata=metadata
        )
        
        # Create text for embedding (prompt + response for semantic search)
        # Pinecone will automatically generate embeddings using llama-text-embed-v2
        embed_text = f"PROMPT: {prompt[:2000]}\n\nRESPONSE: {response.content[:2000]}"
        
        # Prepare metadata for Pinecone (must be flat, string values for filtering)
        # The "text" field is used by Pinecone's integrated embeddings (llama-text-embed-v2)
        pinecone_metadata = {
            "text": embed_text,  # This field is used for integrated embeddings
            "timestamp": timestamp,
            "session_id": self.session_id,
            "project": log.project,
            "task": log.task,
            "model": log.model,
            "provider": log.provider,
            "latency_ms": latency_ms,
            "prompt_preview": prompt[:500],
            "response_preview": response.content[:500],
        }
        
        if metadata:
            for k, v in metadata.items():
                pinecone_metadata[f"meta_{k}"] = str(v)[:500]
        
        # Store in Pinecone with integrated embeddings
        # Pinecone will automatically generate embeddings from the "text" field
        self.index.upsert(vectors=[{
            "id": log_id,
            "metadata": pinecone_metadata
        }])
        
        return log_id
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        project: str = None,
        task: str = None
    ) -> List[Dict]:
        """
        Semantic search across logged interactions.
        Returns relevant past interactions.
        """
        # Build filter
        filter_dict = {}
        if project:
            filter_dict["project"] = {"$eq": project}
        if task:
            filter_dict["task"] = {"$eq": task}
        
        # Search using Pinecone's integrated embeddings
        # Pass raw query text - Pinecone will automatically generate embeddings
        # For integrated embeddings, pass text as a list in the data parameter
        results = self.index.query(
            data=[query],  # Raw text query as list - Pinecone handles embedding with llama-text-embed-v2
            top_k=n_results,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match["metadata"]
            }
            for match in results["matches"]
        ]
    
    def get_project_history(self, project: str, limit: int = 50) -> List[Dict]:
        """Get all interactions for a project."""
        results = self.search(
            query=f"project {project}",
            n_results=limit,
            project=project
        )
        # Sort by timestamp
        results.sort(key=lambda x: x["metadata"].get("timestamp", ""))
        return results
    
    def get_context_for_prompt(
        self,
        prompt: str,
        n_results: int = 3,
        project: str = None
    ) -> str:
        """
        Retrieve relevant past context to inject into a new prompt.
        Returns formatted context string.
        """
        results = self.search(
            query=prompt,
            n_results=n_results,
            project=project or self.current_project
        )
        
        if not results:
            return ""
        
        context_parts = ["Relevant past interactions:"]
        for r in results:
            meta = r["metadata"]
            context_parts.append(
                f"\n[{meta.get('timestamp', 'Unknown')[:10]}] "
                f"Task: {meta.get('task', 'general')}\n"
                f"Prompt: {meta.get('prompt_preview', 'N/A')}\n"
                f"Response: {meta.get('response_preview', 'N/A')}"
            )
        
        return "\n".join(context_parts)
    
    def add_feedback(self, log_id: str, rating: int, notes: str = None):
        """Add feedback/rating to a logged interaction."""
        # Fetch existing
        result = self.index.fetch(ids=[log_id])
        
        if log_id not in result["vectors"]:
            raise ValueError(f"Log ID not found: {log_id}")
        
        existing = result["vectors"][log_id]
        metadata = existing["metadata"]
        
        # Update with feedback
        metadata["rating"] = rating
        metadata["feedback_timestamp"] = datetime.utcnow().isoformat()
        if notes:
            metadata["feedback_notes"] = notes[:500]
        
        # Re-upsert (with integrated embeddings, we don't need to pass values)
        # Pinecone will regenerate embeddings from the "text" field in metadata
        self.index.upsert(vectors=[{
            "id": log_id,
            "metadata": metadata
        }])
    
    def generate_project_report(self, project: str) -> str:
        """
        Generate a narrative report of project history.
        Uses Claude to synthesize the report.
        """
        history = self.get_project_history(project)
        
        if not history:
            return f"No logged interactions found for project: {project}"
        
        # Format history
        history_text = "\n\n---\n\n".join([
            f"**{entry['metadata'].get('timestamp', 'Unknown')}**\n"
            f"Task: {entry['metadata'].get('task', 'general')}\n"
            f"Model: {entry['metadata'].get('model', 'unknown')}\n"
            f"Prompt: {entry['metadata'].get('prompt_preview', 'N/A')}\n"
            f"Response: {entry['metadata'].get('response_preview', 'N/A')}"
            for entry in history[:30]  # Limit to avoid token issues
        ])
        
        # Use Claude to synthesize
        from .llm_clients import get_clients
        clients = get_clients()
        
        synthesis_prompt = f"""Review this project history and create a report:

1. Summary: What this project accomplished
2. Steps Taken: Chronological list of actions with dates
3. Models Used: Which LLMs were used for what
4. Key Outputs: Important results or decisions

Project: {project}
Interactions: {len(history)}

History:
{history_text}

Generate a clear, structured report."""
        
        response = clients.call(synthesis_prompt, model="claude")
        return response.content


# Singleton
_memory: Optional[Memory] = None

def get_memory() -> Memory:
    """Get or create Memory singleton."""
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory
