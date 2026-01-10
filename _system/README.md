# AI Orchestrator - Setup & Run

## Quick Start

### 1. Install dependencies
```bash
cd ~/orchestrator-workspace/_system
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file or export these variables:

```bash
# Required
export PINECONE_API_KEY="your-pinecone-key"
export OPENAI_API_KEY="your-openai-key"  # Required for embeddings

# LLMs (at least one required)
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export XAI_API_KEY="your-xai-key"
export PERPLEXITY_API_KEY="your-perplexity-key"
```

### 3. Create Pinecone index

Go to https://app.pinecone.io and create an index:
- Name: `llm-logs`
- Dimensions: `1536`
- Metric: `cosine`

### 4. Run the app
```bash
cd ~/orchestrator-workspace/_system
streamlit run app.py
```

The UI will open at http://localhost:8501

## Usage

### Basic
1. Select models in sidebar
2. Choose execution mode (single, parallel, sequential, chain)
3. Type command in chat input
4. View results

### Execution Modes
- **Single**: Uses first selected model
- **Parallel**: Runs all selected models simultaneously, Claude synthesizes best result
- **Sequential**: Runs models one after another, shows all outputs
- **Chain**: Passes each model's output to the next model

### Memory
- All interactions are automatically logged to Pinecone
- Search past interactions in sidebar
- Generate project reports

### Projects
- Create projects to organize work
- Each project's history is searchable
- Reports show what was done in each project
