AI Orchestrator
A multi-LLM orchestration platform that lets you sequence and coordinate multiple AI models (Claude, GPT-4, Gemini, and others) through a single UI. Instead of manually prompting models one at a time, you define pipelines where each step can use a different LLM, with configurable execution order, conditional routing, and autonomous error handling.
Built with LangGraph, Pinecone, and Streamlit.

What It Does
Most AI workflows today are single-model, single-prompt interactions. Complex tasks that benefit from different models' strengths (one for reasoning, another for code generation, a third for synthesis) require manual copy-paste between tools. The AI Orchestrator eliminates that friction.
Core Capabilities
Configurable Multi-LLM Pipelines -- Define which models handle each step, in what order, and under what conditions. No hardcoded flows; you specify the routing logic per task.
Autonomous Execution -- Once a pipeline is launched, the system executes each step, handles handoffs between models, and debugs errors automatically. It only surfaces to the user at defined checkpoints or when genuinely stuck.
Persistent Memory and Synthesis -- Pipeline outputs, intermediate results, and execution logs are stored in Pinecone. When multiple models contribute to a task, Claude serves as the default arbiter for synthesizing their outputs into a coherent result.
Streamlit UI -- All interaction happens through a browser-based interface. No CLI or Python scripting required to run pipelines.

Architecture
┌──────────────────────────────────────────────────────┐
│└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│              LangGraph Pipeline Engine                │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  Step 1   │→│  Step 2   │→│  Step N   │           │
│  │ (Claude)  │ │ (GPT-4)  │ │ (Gemini)  │           │
│  └──────────┘  └──────────┘  └──────────┘           │
│                                                      │
│  Conditional routing · Auto-nd monitoring interfacePipeline EngineLangGraphMulti-step LLM workflow orchestrationMemory / StoragePineconeVector-based storage for outputs, logs, and synthesisLLM ProvidersAnthropic, OpenAI, GoogleConfigurable per pipeline stepLanguagePythonEnd-to-end implementation

Design Decisions
Why multi-LLM instead of picking one?
Different models have different strengths. Claude tends to excel at nuanced reasoning and long-form synthesis. GPT-4 is strong at structured code generation. Gemini handles certain multimodal tasks well. Rather than force every task through one model, the orchestrator lets you assign models to the steps where they perform best.
Why LangGraph?
LangGraph provides native support for stateful, multi-step agent workflows with conditional branching. It handles the graph execution, state management, and error propagation that would otherwise require significant custom infrastructure.
Why Pinecone for memory?
Pipeline executions generate structured outputs that need to be retrievable across runs. Pinecone's vector storage enables semantic retrieval of past results, which is especially useful when a new pipeline needs context from previous executions.
Why a UI-first approach?
The goal is to command LLMs, not write code to invoke them. The Streamlit interface means pipeline configuration and execution happen entirely through the browser, with the system handling all interactions with dev tools, terminals, and APIs behind the scenes.

Project Status
This is an active build. Blocks 1 through 4 are complete (~9,000 lines), covering the Streamlit UI, LangGraph pipeline engine, autonomous execution loop with auto-debugging, and Pinecone-based memory and synthesis. Remaining work includes Notion integration, cost intelligence tracking, and browser automation.

About
Built by Ben Sweet as both a working tool and a demonstration of multi-LLM orchestration architecture. The project reflects a conviction that the future of applied AI is not about picking the best single model, but about composing multiple models into systems that leverage each one's strengths.

License
MIT
