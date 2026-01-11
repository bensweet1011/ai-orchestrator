# AI Orchestrator

Multi-LLM orchestration platform with autonomous execution, cross-session memory, and cloud deployment capabilities.

## Features

- **8 LLM Providers**: Claude (Anthropic), GPT-4 (OpenAI), Gemini (Google), Grok (xAI), Perplexity, DeepSeek, Mistral, Groq
- **Visual Pipeline Builder**: Create multi-step LLM workflows with LangGraph
- **Autonomous Execution**: Auto-retry, self-debugging, and code execution
- **Cross-Session Memory**: Pinecone-backed semantic search for persistent context
- **Cost Intelligence**: Real-time tracking, budgets, alerts, and estimation
- **Browser Automation**: Playwright-powered web scraping and interaction
- **Cloud Deployment**: Deploy to Streamlit Cloud or Vercel with one click
- **Document Integration**: Notion and Google Docs syncing

## Quick Start

### 1. Clone and Install

```bash
cd _system
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the `_system` directory:

```bash
# Required - at least one LLM provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional LLM providers
GOOGLE_API_KEY=...
XAI_API_KEY=...
PERPLEXITY_API_KEY=...
DEEPSEEK_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=...

# Optional - Cross-session memory
PINECONE_API_KEY=...
PINECONE_INDEX=ai-orchestrator

# Optional - Document integrations
NOTION_API_KEY=...
GOOGLE_DOCS_CREDENTIALS_PATH=./credentials.json

# Optional - Deployment
GITHUB_TOKEN=ghp_...
VERCEL_TOKEN=...
```

### 3. Run the Application

```bash
streamlit run app.py
```

The UI opens at `http://localhost:8501`.

## UI Overview

The application has 8 main tabs:

| Tab | Purpose |
|-----|---------|
| **Chat** | Interactive LLM conversations with model switching |
| **Pipelines** | Visual pipeline builder and execution |
| **Execution** | Autonomous mode with auto-retry and debugging |
| **Memory** | Semantic search and cross-session context |
| **Costs** | Budget management, tracking, and estimation |
| **Browser** | Web automation and scraping |
| **Deploy** | GitHub, Streamlit Cloud, and Vercel deployment |
| **Settings** | API keys, model configuration, preferences |

## Core Concepts

### Projects

Organize work into projects. Each project has its own:
- Memory namespace
- Pipeline collection
- Cost tracking
- Browser session history

Switch projects using the dropdown in the sidebar.

### Pipelines

Build multi-step LLM workflows:

1. **Create a pipeline** in the Pipelines tab
2. **Add nodes** (LLM, Transform, Conditional, Aggregate, Browser, GitHub)
3. **Connect nodes** to define execution flow
4. **Run** with input text or variables

### Autonomous Execution

Enable "Autonomous Mode" in the Execution tab for:
- **Auto-retry** on failures (configurable attempts)
- **Self-debugging** when code execution fails
- **Cross-session memory** injection for context

### Memory System

Requires Pinecone configuration. Features:
- **Automatic logging** of conversations and results
- **Semantic search** across all sessions
- **Context injection** into new conversations

## API Keys Setup

| Provider | Get Key | Required For |
|----------|---------|--------------|
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | Claude models |
| OpenAI | [platform.openai.com](https://platform.openai.com) | GPT-4, embeddings |
| Google | [makersuite.google.com](https://makersuite.google.com) | Gemini models |
| Pinecone | [pinecone.io](https://www.pinecone.io) | Memory system |
| GitHub | [github.com/settings/tokens](https://github.com/settings/tokens) | Deployment |
| Vercel | [vercel.com/account/tokens](https://vercel.com/account/tokens) | Vercel deployment |

## Usage Examples

### Simple Chat

1. Go to **Chat** tab
2. Select a model from the dropdown
3. Type a message and click Send

### Create a Pipeline

1. Go to **Pipelines** > **Create**
2. Name your pipeline (e.g., "Research Synthesis")
3. Add nodes:
   - LLM Node: Perplexity (research prompt)
   - LLM Node: Claude (synthesis prompt)
   - LLM Node: GPT-4 (editing prompt)
4. Connect: Node 1 → Node 2 → Node 3
5. Save and run

### Autonomous Task

1. Go to **Execution** tab
2. Enable "Autonomous Mode"
3. Set task goal and constraints
4. Configure max iterations and budget
5. Click "Start Execution"

### Deploy to Streamlit Cloud

1. Go to **Deploy** > **Streamlit Cloud**
2. Select GitHub repository
3. Configure app entry point
4. Click "Deploy"

## File Structure

```
_system/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
└── ai_orchestrator/          # Core package
    ├── config.py             # Configuration management
    ├── llm/                  # LLM provider implementations
    ├── pipelines/            # LangGraph pipeline system
    ├── memory/               # Pinecone memory store
    ├── costs/                # Cost tracking and budgets
    ├── execution/            # Autonomous execution loop
    ├── browser/              # Playwright automation
    ├── integrations/         # External service connectors
    └── deploy/               # Deployment utilities
```

## Troubleshooting

### "No API keys configured"

Ensure your `.env` file exists and has at least one LLM provider key set.

### "Memory search failed"

Check that:
1. `PINECONE_API_KEY` is set
2. `PINECONE_INDEX` matches your Pinecone index name
3. `OPENAI_API_KEY` is set (required for embeddings)

### "Pipeline execution failed"

1. Check the error message in the UI
2. Verify all nodes have valid configurations
3. Ensure required API keys for each model are set

### Browser automation not working

Install Playwright browsers:
```bash
playwright install chromium
```

## Development

### Adding a New LLM Provider

1. Create provider class in `ai_orchestrator/llm/`
2. Implement the `LLMProvider` interface
3. Register in `ai_orchestrator/llm/__init__.py`
4. Add pricing in `ai_orchestrator/costs/pricing.py`

### Adding a New Pipeline Node Type

1. Add to `NodeType` enum in `pipelines/nodes.py`
2. Implement node execution logic
3. Add UI for node configuration in `app.py`

## License

MIT License - See LICENSE file for details.
