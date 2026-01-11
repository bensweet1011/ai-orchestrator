# AI Orchestrator - Developer Context

This document provides context for developers (or AI assistants) working on the AI Orchestrator codebase.

## Architecture Overview

```
ai_orchestrator/
├── llm/                  # LLM Provider Implementations
│   ├── base.py           # LLMProvider abstract base class
│   ├── anthropic.py      # Claude models
│   ├── openai.py         # GPT models
│   ├── google.py         # Gemini models
│   ├── xai.py            # Grok models
│   ├── perplexity.py     # Perplexity (web search)
│   ├── deepseek.py       # DeepSeek models
│   ├── mistral.py        # Mistral models
│   └── groq.py           # Groq (fast inference)
│
├── pipelines/            # LangGraph Pipeline System
│   ├── base.py           # Pipeline, PipelineState classes
│   ├── nodes.py          # NodeType enum, NodeConfig, node factories
│   ├── graph.py          # LangGraph integration
│   └── examples/         # Example pipeline definitions
│
├── memory/               # Cross-Session Memory
│   └── store.py          # MemoryStore (Pinecone + OpenAI embeddings)
│
├── costs/                # Cost Intelligence
│   ├── tracker.py        # CostTracker, CostRecord
│   ├── budget.py         # BudgetManager, alerts
│   ├── estimator.py      # CostEstimator (pre-execution)
│   └── pricing.py        # PricingManager, model costs
│
├── execution/            # Autonomous Execution
│   └── autonomous.py     # AutonomousExecutor, ExecutionState
│
├── browser/              # Browser Automation
│   └── client.py         # BrowserClient (Playwright)
│
├── integrations/         # External Services
│   ├── notion.py         # Notion API
│   ├── google_docs.py    # Google Docs API
│   ├── github.py         # GitHub API (PyGithub)
│   ├── streamlit_cloud.py # Streamlit deployment
│   └── vercel.py         # Vercel deployment
│
├── deploy/               # Deployment Utilities
│   ├── design_system.py  # Next.js project scaffolding
│   ├── components.py     # React component library
│   ├── registry.py       # ProductRegistry
│   └── iteration.py      # IterationLoop (feedback)
│
└── config.py             # Central configuration, env loading
```

## Key Entry Points

### `app.py` - Main Streamlit Application

~3,500 lines. Organized into 8 tabs:

| Tab | Lines (approx) | Key Functions |
|-----|----------------|---------------|
| Chat | 300-600 | `display_chat_message()`, LLM calls |
| Pipelines | 600-1000 | `list_pipelines()`, `get_pipeline()`, `save_pipeline()` |
| Execution | 1000-1350 | `AutonomousExecutor` integration |
| Memory | 1350-1600 | `MemoryStore` search/display |
| Costs | 1600-2000 | `CostTracker`, `BudgetManager`, `CostEstimator` |
| Browser | 2000-2800 | `BrowserClient` automation |
| Deploy | 2800-3350 | GitHub, Streamlit Cloud, Vercel |
| Settings | 3350-3500 | API key management |

### Session State

Key `st.session_state` variables:

```python
st.session_state.current_project      # Active project name
st.session_state.current_task         # Active task description
st.session_state.messages             # Chat history
st.session_state.execution_history    # Autonomous execution log
st.session_state.browser_client       # BrowserClient instance
st.session_state.deploy_registry      # ProductRegistry instance
```

## Module Deep Dives

### LLM Providers (`llm/`)

All providers inherit from `LLMProvider`:

```python
class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str: ...

    @abstractmethod
    def stream(self, messages: List[Dict], **kwargs) -> Iterator[str]: ...
```

Factory function: `get_llm_provider(model_id: str) -> LLMProvider`

Model IDs follow pattern: `provider/model-name` (e.g., `anthropic/claude-3-5-sonnet`)

### Pipelines (`pipelines/`)

**NodeType enum**: `LLM`, `CONDITIONAL`, `TRANSFORM`, `AGGREGATE`, `BROWSER`, `GITHUB`

**NodeConfig dataclass**: Configuration for each node including model, prompt template, edges.

**Key functions**:
- `create_llm_node()` - Creates LLM execution node
- `create_browser_node()` - Creates browser automation node
- `create_github_node()` - Creates GitHub operation node
- `create_node()` - Factory that routes to specific creators

**Pipeline execution flow**:
1. `Pipeline.compile()` - Builds LangGraph
2. `Pipeline.run(input)` - Executes graph
3. Results stored in `PipelineState`

### Memory System (`memory/`)

Uses Pinecone for vector storage + OpenAI for embeddings.

```python
store = MemoryStore()
store.set_project("my-project")

# Store
store.log_interaction(role="user", content="...", metadata={})

# Search
results = store.search("query", n_results=5)
```

Namespaces: Each project gets a Pinecone namespace for isolation.

### Cost System (`costs/`)

**CostTracker**: Records actual costs per API call
**CostEstimator**: Pre-execution estimates based on token counts
**BudgetManager**: Limits, alerts, period tracking (hourly/daily/weekly)
**PricingManager**: Model pricing data (input/output per 1K tokens)

### Autonomous Execution (`execution/`)

```python
executor = AutonomousExecutor(
    model="anthropic/claude-3-5-sonnet",
    max_iterations=10,
    budget_limit=1.0,
)

result = executor.run(
    goal="Build a web scraper",
    context="...",
    constraints=["Use Python", "No external APIs"],
)
```

Features:
- Auto-retry on failure
- Self-debugging (analyzes errors, fixes code)
- Memory injection (pulls relevant context from Pinecone)
- Code execution in sandbox

### Browser Automation (`browser/`)

```python
client = BrowserClient()
await client.initialize()

# Navigate and interact
await client.navigate("https://example.com")
content = await client.get_page_content()
await client.click("#submit")
await client.type_text("#search", "query")

# Screenshot
screenshot = await client.screenshot()

await client.close()
```

Built on Playwright. Supports headless/headful modes.

### Integrations (`integrations/`)

All use singleton pattern:

```python
# GitHub
client = get_github_client()
repos = client.list_repos()
client.push_file(repo, path, content, message)

# Vercel
client = get_vercel_client()
projects = client.list_projects()
deployment = client.deploy(project_id, files)
```

## Common Patterns

### Singleton Clients

```python
_client: Optional[MyClient] = None

def get_client() -> MyClient:
    global _client
    if _client is None:
        _client = MyClient()
    return _client

def reset_client():
    global _client
    _client = None
```

### Error Handling

```python
try:
    result = api_call()
except Exception as e:
    st.error(f"Failed to {action}: {e}")
    st.caption("Try: [specific recovery suggestion]")
```

### Streamlit Patterns

```python
# Loading state
with st.spinner("Processing..."):
    result = long_operation()

# Columns layout
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Label", value)

# Expanders for details
with st.expander("Details"):
    st.json(data)
```

## Common Tasks

### Adding a New LLM Provider

1. Create `ai_orchestrator/llm/newprovider.py`:
```python
class NewProvider(LLMProvider):
    def chat(self, messages, **kwargs) -> str: ...
    def stream(self, messages, **kwargs) -> Iterator[str]: ...
```

2. Register in `ai_orchestrator/llm/__init__.py`:
```python
from .newprovider import NewProvider
PROVIDERS["newprovider"] = NewProvider
```

3. Add pricing in `ai_orchestrator/costs/pricing.py`:
```python
MODEL_PRICING["newprovider/model-name"] = ModelPricing(...)
```

4. Add API key handling in `ai_orchestrator/config.py`:
```python
def get_api_keys():
    return {
        ...
        "newprovider": os.getenv("NEWPROVIDER_API_KEY"),
    }
```

### Adding a New Pipeline Node Type

1. Add to `NodeType` enum in `pipelines/nodes.py`:
```python
class NodeType(str, Enum):
    ...
    NEWTYPE = "newtype"
```

2. Add fields to `NodeConfig`:
```python
@dataclass
class NodeConfig:
    ...
    newtype_field: Optional[str] = None
```

3. Create node factory:
```python
def create_newtype_node(config: NodeConfig):
    def node_fn(state: PipelineState) -> PipelineState:
        # Implementation
        return state
    return node_fn
```

4. Register in `create_node()`:
```python
def create_node(config: NodeConfig):
    if config.node_type == NodeType.NEWTYPE:
        return create_newtype_node(config)
    ...
```

5. Add UI in `app.py` (Pipelines tab, node configuration section).

### Adding a New Integration

1. Create `ai_orchestrator/integrations/newservice.py`:
```python
@dataclass
class NewServiceClient:
    def __init__(self):
        self.api_key = os.getenv("NEWSERVICE_API_KEY")

    def operation(self) -> Result: ...

_client: Optional[NewServiceClient] = None

def get_newservice_client() -> NewServiceClient:
    global _client
    if _client is None:
        _client = NewServiceClient()
    return _client
```

2. Export in `ai_orchestrator/integrations/__init__.py`

3. Add UI in `app.py` (appropriate tab)

## Testing

### Syntax Check

```bash
python3 -m py_compile app.py
python3 -m py_compile ai_orchestrator/**/*.py
```

### Import Test

```bash
python3 -c "import ai_orchestrator; print('OK')"
```

### Run UI

```bash
streamlit run app.py
# Check all 8 tabs load without errors
```

## Known Issues / Areas for Improvement

1. **Error handling**: Some bare `except: pass` blocks swallow errors silently
2. **Loading states**: Some long operations lack spinners
3. **Type hints**: Not all functions have complete type annotations
4. **Tests**: No automated test suite (manual verification only)

## Dependencies

Key packages:
- `streamlit` - UI framework
- `anthropic`, `openai`, `google-generativeai` - LLM SDKs
- `langgraph` - Pipeline execution
- `pinecone` - Vector storage
- `playwright` - Browser automation
- `PyGithub` - GitHub API
- `requests` - HTTP client (Vercel, Perplexity)
