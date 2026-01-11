"""
AI Orchestrator - Streamlit UI
Your command center for orchestrating LLMs.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file (must be before other imports)
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import streamlit as st
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from ai_orchestrator.config import (
    MODELS,
    get_available_models,
    check_required_keys,
    PROJECTS_DIR,
)
from ai_orchestrator.core.llm_clients import get_clients, LLMResponse
from ai_orchestrator.core.memory import get_memory
from ai_orchestrator.pipelines import (
    BasePipeline,
    NodeConfig,
    NodeType,
    list_pipelines,
    get_pipeline,
    save_pipeline,
    delete_pipeline,
    get_pipeline_templates,
    create_from_template,
)
from ai_orchestrator.execution import (
    AutonomousExecutor,
    ExecutionMode,
    CheckpointManager,
    EscalationManager,
)
from ai_orchestrator.memory import (
    SynthesisType,
    get_memory_store,
    get_synthesizer,
)

# Page config
st.set_page_config(
    page_title="AI Orchestrator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_project" not in st.session_state:
    st.session_state.current_project = "default"
if "current_task" not in st.session_state:
    st.session_state.current_task = "general"
if "selected_models" not in st.session_state:
    st.session_state.selected_models = ["claude"]
if "execution_mode" not in st.session_state:
    st.session_state.execution_mode = "single"
if "last_outputs" not in st.session_state:
    st.session_state.last_outputs = {}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"
if "pipeline_config" not in st.session_state:
    st.session_state.pipeline_config = {}
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "autonomous_result" not in st.session_state:
    st.session_state.autonomous_result = None
if "browser_session" not in st.session_state:
    st.session_state.browser_session = None
if "browser_queue" not in st.session_state:
    st.session_state.browser_queue = None
if "browser_history" not in st.session_state:
    st.session_state.browser_history = []


def check_setup() -> tuple[bool, List[str]]:
    """Check if system is properly configured."""
    issues = []
    keys = check_required_keys()

    if not keys["pinecone"]:
        issues.append("PINECONE_API_KEY not set")
    if not keys["openai"]:
        issues.append("OPENAI_API_KEY not set (required for embeddings)")
    if not keys["any_llm"]:
        issues.append(
            "No LLM API keys set (need at least one of: ANTHROPIC, OPENAI, GOOGLE)"
        )

    return len(issues) == 0, issues


def call_single_model(
    prompt: str, model: str, system: str = None
) -> tuple[str, LLMResponse, int]:
    """Call a single model and return (model_name, response, latency_ms)."""
    clients = get_clients()
    start = time.time()
    response = clients.call(prompt, model=model, system=system)
    latency = int((time.time() - start) * 1000)
    return model, response, latency


def call_models_parallel(
    prompt: str, models: List[str], system: str = None
) -> Dict[str, tuple[LLMResponse, int]]:
    """Call multiple models in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(call_single_model, prompt, model, system): model
            for model in models
        }
        for future in futures:
            model = futures[future]
            try:
                _, response, latency = future.result()
                results[model] = (response, latency)
            except Exception as e:
                results[model] = (None, str(e))

    return results


def call_models_sequential(
    prompt: str, models: List[str], system: str = None, pass_output: bool = False
) -> Dict[str, tuple[LLMResponse, int]]:
    """Call models sequentially, optionally passing output to next."""
    results = {}
    current_prompt = prompt

    for model in models:
        try:
            _, response, latency = call_single_model(current_prompt, model, system)
            results[model] = (response, latency)

            if pass_output and response:
                current_prompt = f"Previous output:\n{response.content}\n\nBased on the above, {prompt}"
        except Exception as e:
            results[model] = (None, str(e))
            if pass_output:
                break

    return results


def synthesize_outputs(
    outputs: Dict[str, tuple[LLMResponse, int]], original_prompt: str
) -> str:
    """Use Claude to synthesize/arbitrate multiple outputs."""
    clients = get_clients()

    output_text = ""
    for model, (response, latency) in outputs.items():
        if response:
            output_text += f"\n\n=== {model.upper()} ===\n{response.content}"

    synthesis_prompt = f"""You are an expert arbiter. Multiple LLMs responded to this prompt:

ORIGINAL PROMPT: {original_prompt}

OUTPUTS:{output_text}

Your task:
1. Evaluate each output for accuracy, completeness, and quality
2. Identify the best response OR synthesize the best parts from multiple responses
3. Provide the optimal final answer

If one response is clearly best, return it (possibly with minor improvements).
If multiple have complementary strengths, synthesize them into a superior response.

FINAL ANSWER:"""

    response = clients.call(synthesis_prompt, model="claude")
    return response.content


def log_interaction(
    prompt: str, response: LLMResponse, latency_ms: int, metadata: dict = None
):
    """Log interaction to Pinecone."""
    try:
        memory = get_memory()
        memory.set_context(
            project=st.session_state.current_project, task=st.session_state.current_task
        )
        memory.log(prompt, response, latency_ms, metadata=metadata)
    except Exception as e:
        st.warning(f"Failed to log to memory: {e}")


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🎯 AI Orchestrator")

    # Status check
    setup_ok, issues = check_setup()
    if not setup_ok:
        st.error("Setup Issues:")
        for issue in issues:
            st.write(f"• {issue}")
        st.stop()
    else:
        st.success("System ready")

    st.divider()

    # Project & Task
    st.subheader("Context")

    existing_projects = ["default"] + [
        p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()
    ]

    project = st.selectbox(
        "Project",
        options=existing_projects,
        index=(
            existing_projects.index(st.session_state.current_project)
            if st.session_state.current_project in existing_projects
            else 0
        ),
    )
    st.session_state.current_project = project

    new_project = st.text_input("Or create new project:")
    if new_project and st.button("Create Project"):
        project_path = PROJECTS_DIR / new_project
        project_path.mkdir(exist_ok=True)
        st.session_state.current_project = new_project
        st.rerun()

    task = st.text_input("Task label:", value=st.session_state.current_task)
    st.session_state.current_task = task

    st.divider()

    # Model Selection
    st.subheader("Available Models")

    available = get_available_models()
    if not available:
        st.warning("No LLM API keys configured")

    for model_name in available:
        model_info = MODELS[model_name]
        st.caption(f"• **{model_name}**: {model_info.description[:50]}...")

    st.divider()

    # Memory Search
    st.subheader("Memory")

    search_query = st.text_input("Search past interactions:")
    if search_query and st.button("Search"):
        try:
            memory = get_memory()
            results = memory.search(
                search_query, n_results=5, project=st.session_state.current_project
            )
            if results:
                for r in results:
                    with st.expander(
                        f"{r['metadata'].get('timestamp', '')[:10]} - {r['metadata'].get('task', '')}"
                    ):
                        st.write(f"**Model:** {r['metadata'].get('model', 'N/A')}")
                        st.write(
                            f"**Prompt:** {r['metadata'].get('prompt_preview', 'N/A')}"
                        )
                        st.write(
                            f"**Response:** {r['metadata'].get('response_preview', 'N/A')}"
                        )
            else:
                st.info("No matching interactions found")
        except Exception as e:
            st.error(f"Search failed: {e}")


# =============================================================================
# MAIN AREA - TABS
# =============================================================================

tab_chat, tab_pipelines, tab_autonomous, tab_memory, tab_cost, tab_integrations, tab_browser = st.tabs(
    ["💬 Chat", "🔧 Pipelines", "🤖 Autonomous", "🧠 Memory", "💰 Cost", "🔗 Integrations", "🌐 Browser"]
)

# =============================================================================
# CHAT TAB
# =============================================================================

with tab_chat:
    st.header(f"Project: {st.session_state.current_project}")

    # Model and mode selection for chat
    col1, col2 = st.columns(2)
    with col1:
        selected = st.multiselect(
            "Select models:",
            options=available,
            default=(
                st.session_state.selected_models
                if all(m in available for m in st.session_state.selected_models)
                else [available[0]] if available else []
            ),
        )
        st.session_state.selected_models = (
            selected
            if selected
            else (
                ["claude"]
                if "claude" in available
                else [available[0]] if available else []
            )
        )

    with col2:
        mode = st.selectbox(
            "Execution mode:",
            options=["single", "parallel", "sequential", "sequential_chain"],
            format_func=lambda x: {
                "single": "Single (first selected)",
                "parallel": "Parallel (all at once, pick best)",
                "sequential": "Sequential (one after another)",
                "sequential_chain": "Chain (pass output to next)",
            }[x],
            index=["single", "parallel", "sequential", "sequential_chain"].index(
                st.session_state.execution_mode
            ),
        )
        st.session_state.execution_mode = mode

    st.divider()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    prompt = st.chat_input("Enter your command...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            models = st.session_state.selected_models
            mode = st.session_state.execution_mode

            if mode == "single":
                model = models[0]
                with st.spinner(f"Calling {model}..."):
                    try:
                        _, response, latency = call_single_model(prompt, model)
                        st.markdown(response.content)
                        st.caption(f"*{model} | {latency}ms*")
                        log_interaction(prompt, response, latency)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response.content}
                        )
                        st.session_state.last_outputs = {model: response.content}
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif mode == "parallel":
                with st.spinner(f"Calling {len(models)} models in parallel..."):
                    try:
                        results = call_models_parallel(prompt, models)

                        st.subheader("Individual Outputs")
                        for model, (response, latency) in results.items():
                            with st.expander(
                                f"{model} ({latency}ms)"
                                if response
                                else f"{model} (failed)"
                            ):
                                if response:
                                    st.markdown(response.content)
                                    log_interaction(
                                        prompt, response, latency, {"mode": "parallel"}
                                    )
                                else:
                                    st.error(f"Failed: {latency}")

                        successful = {
                            m: r for m, r in results.items() if r[0] is not None
                        }
                        if len(successful) > 1:
                            st.subheader("Synthesized Result")
                            with st.spinner(
                                "Claude is synthesizing the best answer..."
                            ):
                                synthesis = synthesize_outputs(successful, prompt)
                                st.markdown(synthesis)
                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": f"**Synthesized from {len(successful)} models:**\n\n{synthesis}",
                                    }
                                )
                        elif len(successful) == 1:
                            model, (response, _) = list(successful.items())[0]
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response.content}
                            )

                        st.session_state.last_outputs = {
                            m: r[0].content for m, r in results.items() if r[0]
                        }
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif mode in ["sequential", "sequential_chain"]:
                pass_output = mode == "sequential_chain"

                with st.spinner(f"Calling {len(models)} models sequentially..."):
                    try:
                        results = call_models_sequential(
                            prompt, models, pass_output=pass_output
                        )

                        for model, (response, latency) in results.items():
                            with st.expander(
                                f"{model} ({latency}ms)"
                                if response
                                else f"{model} (failed)"
                            ):
                                if response:
                                    st.markdown(response.content)
                                    log_interaction(
                                        prompt,
                                        response,
                                        latency,
                                        {
                                            "mode": mode,
                                            "sequence_position": list(
                                                results.keys()
                                            ).index(model),
                                        },
                                    )
                                else:
                                    st.error(f"Failed: {latency}")

                        last_response = None
                        for model, (response, _) in results.items():
                            if response:
                                last_response = response.content

                        if last_response:
                            st.subheader("Final Output")
                            st.markdown(last_response)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": last_response}
                            )

                        st.session_state.last_outputs = {
                            m: r[0].content for m, r in results.items() if r[0]
                        }
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Quick actions
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("New Session"):
            try:
                memory = get_memory()
                memory.new_session()
                st.success("New session started")
            except Exception as e:
                st.error(f"Failed: {e}")

    with col3:
        if st.session_state.last_outputs:
            if st.button("Copy Last Output"):
                st.code(list(st.session_state.last_outputs.values())[-1])


# =============================================================================
# PIPELINES TAB
# =============================================================================

with tab_pipelines:
    st.header("LangGraph Pipelines")
    st.caption(
        "Build multi-step LLM workflows with user-defined model selection per node"
    )

    pipeline_subtab = st.radio(
        "Mode:",
        ["Run Pipeline", "Create Pipeline", "Manage Pipelines"],
        horizontal=True,
    )

    st.divider()

    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    if pipeline_subtab == "Run Pipeline":
        st.subheader("Execute a Pipeline")

        # Get available pipelines and templates
        saved_pipelines = list_pipelines()
        templates = get_pipeline_templates()

        pipeline_source = st.radio(
            "Pipeline source:", ["Saved Pipelines", "Templates"], horizontal=True
        )

        if pipeline_source == "Saved Pipelines":
            if not saved_pipelines:
                st.info("No saved pipelines yet. Create one or use a template.")
            else:
                pipeline_names = [p["name"] for p in saved_pipelines]
                selected_pipeline = st.selectbox(
                    "Select pipeline:", options=pipeline_names
                )

                if selected_pipeline:
                    pipeline = get_pipeline(selected_pipeline)
                    if pipeline:
                        st.write(f"**Description:** {pipeline.description}")
                        st.write(f"**Nodes:** {' → '.join(pipeline.get_node_order())}")

                        # Show node LLM assignments
                        st.write("**Model Assignments:**")
                        for node_name, config in pipeline.nodes.items():
                            st.write(f"  • {node_name}: `{config.llm}`")

                        # Input
                        pipeline_input = st.text_area(
                            "Input text:",
                            height=150,
                            placeholder="Enter the text to process through the pipeline...",
                        )

                        if st.button("Run Pipeline", type="primary"):
                            if pipeline_input:
                                with st.spinner("Executing pipeline..."):
                                    result = pipeline.run(pipeline_input)
                                    st.session_state.pipeline_results = result

                                if result.success:
                                    st.success(
                                        f"Pipeline completed in {result.total_latency_ms}ms"
                                    )
                                else:
                                    st.error("Pipeline completed with errors")

                                # Show results
                                st.subheader("Results")

                                # Show each node's output
                                for node_name in pipeline.get_node_order():
                                    if node_name in result.node_outputs:
                                        output = result.node_outputs[node_name]
                                        with st.expander(
                                            f"**{node_name}** ({output['model']}, {output['latency_ms']}ms)",
                                            expanded=(
                                                node_name == pipeline.finish_points[0]
                                            ),
                                        ):
                                            st.markdown(output["content"])

                                # Show final output
                                st.subheader("Final Output")
                                st.markdown(result.final_output)

                                # Token usage
                                if result.total_tokens:
                                    st.caption(
                                        f"Tokens: {result.total_tokens.get('input_tokens', 0)} in, "
                                        f"{result.total_tokens.get('output_tokens', 0)} out"
                                    )

                                # Errors
                                if result.errors:
                                    st.error("Errors:")
                                    for err in result.errors:
                                        st.write(f"  • {err['node']}: {err['error']}")
                            else:
                                st.warning("Please enter input text")

        else:  # Templates
            template_names = [t["name"] for t in templates]
            selected_template = st.selectbox("Select template:", options=template_names)

            if selected_template:
                template = next(t for t in templates if t["name"] == selected_template)
                st.write(f"**Description:** {template['description']}")

                template_data = template["template"]
                node_names = list(template_data["nodes"].keys())

                st.write("**Configure LLMs for each node:**")

                llm_assignments = {}
                cols = st.columns(len(node_names))
                for i, node_name in enumerate(node_names):
                    with cols[i]:
                        llm_assignments[node_name] = st.selectbox(
                            f"{node_name}:",
                            options=available,
                            key=f"template_llm_{node_name}",
                        )

                pipeline_input = st.text_area(
                    "Input text:",
                    height=150,
                    placeholder="Enter the text to process...",
                    key="template_input",
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Run from Template", type="primary"):
                        if pipeline_input and all(llm_assignments.values()):
                            try:
                                pipeline = create_from_template(
                                    selected_template, llm_assignments
                                )
                                with st.spinner("Executing pipeline..."):
                                    result = pipeline.run(pipeline_input)
                                    st.session_state.pipeline_results = result

                                if result.success:
                                    st.success(
                                        f"Completed in {result.total_latency_ms}ms"
                                    )
                                else:
                                    st.error("Completed with errors")

                                for node_name in pipeline.get_node_order():
                                    if node_name in result.node_outputs:
                                        output = result.node_outputs[node_name]
                                        with st.expander(
                                            f"**{node_name}** ({output['model']})"
                                        ):
                                            st.markdown(output["content"])

                                st.subheader("Final Output")
                                st.markdown(result.final_output)

                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.warning(
                                "Please enter input and select LLMs for all nodes"
                            )

                with col2:
                    save_name = st.text_input("Save as:", placeholder="my_pipeline")
                    if st.button("Save Configuration"):
                        if save_name and all(llm_assignments.values()):
                            try:
                                pipeline = create_from_template(
                                    selected_template,
                                    llm_assignments,
                                    new_name=save_name,
                                )
                                save_pipeline(pipeline)
                                st.success(f"Saved as '{save_name}'")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

    # -------------------------------------------------------------------------
    # CREATE PIPELINE
    # -------------------------------------------------------------------------
    elif pipeline_subtab == "Create Pipeline":
        st.subheader("Build a Custom Pipeline")

        # Pipeline metadata
        pipeline_name = st.text_input(
            "Pipeline name:", placeholder="my_custom_pipeline"
        )
        pipeline_desc = st.text_input(
            "Description:", placeholder="What does this pipeline do?"
        )

        st.divider()

        # Node builder
        st.write("**Define Nodes:**")

        if "builder_nodes" not in st.session_state:
            st.session_state.builder_nodes = []

        # Add node form
        with st.expander(
            "Add New Node", expanded=len(st.session_state.builder_nodes) == 0
        ):
            node_name = st.text_input(
                "Node name:", placeholder="summarize", key="new_node_name"
            )
            node_llm = st.selectbox("LLM:", options=available, key="new_node_llm")
            node_system = st.text_area(
                "System prompt:",
                placeholder="Instructions for this node...",
                key="new_node_system",
            )
            node_input = st.selectbox(
                "Input from:",
                options=["input (user text)"]
                + [n["name"] for n in st.session_state.builder_nodes],
                key="new_node_input",
            )

            if st.button("Add Node"):
                if node_name and node_llm:
                    input_key = (
                        "input" if node_input == "input (user text)" else node_input
                    )
                    st.session_state.builder_nodes.append(
                        {
                            "name": node_name,
                            "llm": node_llm,
                            "system_prompt": node_system,
                            "input_key": input_key,
                        }
                    )
                    st.rerun()
                else:
                    st.warning("Node name and LLM are required")

        # Show existing nodes
        if st.session_state.builder_nodes:
            st.write("**Current Nodes:**")
            for i, node in enumerate(st.session_state.builder_nodes):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(
                        f"**{i+1}. {node['name']}** ({node['llm']}) ← {node['input_key']}"
                    )
                with col2:
                    if st.button("⬆", key=f"up_{i}") and i > 0:
                        (
                            st.session_state.builder_nodes[i],
                            st.session_state.builder_nodes[i - 1],
                        ) = (
                            st.session_state.builder_nodes[i - 1],
                            st.session_state.builder_nodes[i],
                        )
                        st.rerun()
                with col3:
                    if st.button("🗑", key=f"del_{i}"):
                        st.session_state.builder_nodes.pop(i)
                        st.rerun()

            st.divider()

            # Save pipeline
            if st.button("Create Pipeline", type="primary"):
                if pipeline_name and len(st.session_state.builder_nodes) >= 1:
                    try:
                        pipeline = BasePipeline(
                            name=pipeline_name, description=pipeline_desc
                        )

                        for node in st.session_state.builder_nodes:
                            pipeline.add_node(
                                NodeConfig(
                                    name=node["name"],
                                    node_type=NodeType.LLM,
                                    llm=node["llm"],
                                    system_prompt=node["system_prompt"],
                                    input_key=node["input_key"],
                                )
                            )

                        # Add sequential edges
                        for i in range(len(st.session_state.builder_nodes) - 1):
                            pipeline.add_edge(
                                st.session_state.builder_nodes[i]["name"],
                                st.session_state.builder_nodes[i + 1]["name"],
                            )

                        pipeline.set_entry_point(
                            st.session_state.builder_nodes[0]["name"]
                        )
                        pipeline.set_finish_point(
                            st.session_state.builder_nodes[-1]["name"]
                        )

                        # Validate and save
                        pipeline.compile()
                        save_pipeline(pipeline)

                        st.success(f"Pipeline '{pipeline_name}' created!")
                        st.session_state.builder_nodes = []
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error creating pipeline: {e}")
                else:
                    st.warning("Pipeline needs a name and at least one node")

            if st.button("Clear All"):
                st.session_state.builder_nodes = []
                st.rerun()

    # -------------------------------------------------------------------------
    # MANAGE PIPELINES
    # -------------------------------------------------------------------------
    else:
        st.subheader("Manage Saved Pipelines")

        saved = list_pipelines()

        if not saved:
            st.info("No saved pipelines yet.")
        else:
            for p in saved:
                with st.expander(
                    f"**{p['name']}** - {p['description'][:50]}..."
                    if p["description"]
                    else f"**{p['name']}**"
                ):
                    st.write(f"**Nodes:** {', '.join(p['nodes'])}")
                    if p.get("created_at"):
                        st.caption(f"Created: {p['created_at'][:10]}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("View", key=f"view_{p['name']}"):
                            pipeline = get_pipeline(p["name"])
                            if pipeline:
                                st.code(pipeline.visualize())
                    with col2:
                        if st.button("Duplicate", key=f"dup_{p['name']}"):
                            from ai_orchestrator.pipelines import duplicate_pipeline

                            new_name = f"{p['name']}_copy"
                            duplicate_pipeline(p["name"], new_name)
                            st.success(f"Duplicated as '{new_name}'")
                            st.rerun()
                    with col3:
                        if st.button("Delete", key=f"del_{p['name']}"):
                            delete_pipeline(p["name"])
                            st.success(f"Deleted '{p['name']}'")
                            st.rerun()


# =============================================================================
# AUTONOMOUS EXECUTION TAB
# =============================================================================

with tab_autonomous:
    st.header("Autonomous Execution")
    st.caption("Run pipelines with auto-retry, debugging, and checkpoints")

    auto_subtab = st.radio(
        "Mode:",
        ["Run Autonomous", "View Escalations", "Manage Checkpoints"],
        horizontal=True,
        key="auto_subtab",
    )

    st.divider()

    # -------------------------------------------------------------------------
    # RUN AUTONOMOUS
    # -------------------------------------------------------------------------
    if auto_subtab == "Run Autonomous":
        st.subheader("Autonomous Pipeline Execution")

        # Pipeline selection
        saved_pipelines = list_pipelines()
        templates = get_pipeline_templates()

        if not saved_pipelines and not templates:
            st.info("No pipelines available. Create one in the Pipelines tab first.")
        else:
            # Source selection
            auto_source = st.radio(
                "Pipeline source:",
                ["Saved Pipelines", "Templates"],
                horizontal=True,
                key="auto_source",
            )

            if auto_source == "Saved Pipelines" and saved_pipelines:
                pipeline_names = [p["name"] for p in saved_pipelines]
                selected_auto_pipeline = st.selectbox(
                    "Select pipeline:",
                    options=pipeline_names,
                    key="auto_pipeline_select",
                )
                pipeline = (
                    get_pipeline(selected_auto_pipeline)
                    if selected_auto_pipeline
                    else None
                )

            elif auto_source == "Templates" and templates:
                template_names = [t["name"] for t in templates]
                selected_template = st.selectbox(
                    "Select template:",
                    options=template_names,
                    key="auto_template_select",
                )

                if selected_template:
                    template = next(
                        t for t in templates if t["name"] == selected_template
                    )
                    template_data = template["template"]
                    node_names = list(template_data["nodes"].keys())

                    st.write("**Configure LLMs for each node:**")
                    llm_assignments = {}
                    cols = st.columns(len(node_names))
                    for i, node_name in enumerate(node_names):
                        with cols[i]:
                            llm_assignments[node_name] = st.selectbox(
                                f"{node_name}:",
                                options=available,
                                key=f"auto_template_llm_{node_name}",
                            )

                    if all(llm_assignments.values()):
                        pipeline = create_from_template(
                            selected_template, llm_assignments
                        )
                    else:
                        pipeline = None
                else:
                    pipeline = None
            else:
                pipeline = None

            if pipeline:
                st.write(f"**Pipeline:** {pipeline.name}")
                st.write(f"**Nodes:** {' → '.join(pipeline.get_node_order())}")

                # Execution configuration
                st.subheader("Execution Settings")

                col1, col2, col3 = st.columns(3)
                with col1:
                    exec_mode = st.selectbox(
                        "Execution mode:",
                        options=["autonomous", "supervised", "manual"],
                        format_func=lambda x: {
                            "autonomous": "Autonomous (full auto-retry)",
                            "supervised": "Supervised (checkpoints require approval)",
                            "manual": "Manual (no auto-retry)",
                        }[x],
                        key="auto_exec_mode",
                    )

                with col2:
                    max_retries = st.number_input(
                        "Max retries per node:",
                        min_value=0,
                        max_value=10,
                        value=3,
                        key="auto_max_retries",
                    )

                with col3:
                    auto_debug = st.checkbox(
                        "Enable auto-debugging", value=True, key="auto_debug"
                    )

                # Checkpoint nodes
                checkpoint_nodes = st.multiselect(
                    "Nodes requiring approval (checkpoints):",
                    options=list(pipeline.nodes.keys()),
                    key="auto_checkpoint_nodes",
                )

                st.divider()

                # Input
                auto_input = st.text_area(
                    "Input text:",
                    height=150,
                    placeholder="Enter text to process autonomously...",
                    key="auto_input",
                )

                if st.button("Run Autonomous", type="primary", key="run_auto"):
                    if auto_input:
                        # Create executor
                        executor = AutonomousExecutor(
                            pipeline=pipeline,
                            max_retries=max_retries,
                            checkpoint_nodes=checkpoint_nodes,
                            auto_debug=auto_debug,
                        )

                        exec_mode_enum = {
                            "autonomous": ExecutionMode.AUTONOMOUS,
                            "supervised": ExecutionMode.SUPERVISED,
                            "manual": ExecutionMode.MANUAL,
                        }[exec_mode]

                        with st.spinner("Executing autonomously..."):
                            result = executor.run(
                                auto_input, execution_mode=exec_mode_enum
                            )
                            st.session_state.autonomous_result = result

                        # Show results
                        if result.success:
                            st.success(
                                f"Pipeline completed successfully in {result.total_latency_ms}ms"
                            )
                        else:
                            st.error("Pipeline failed - see details below")

                        # Results tabs
                        result_tab1, result_tab2, result_tab3 = st.tabs(
                            ["Output", "Retry History", "Execution Trace"]
                        )

                        with result_tab1:
                            st.subheader("Node Outputs")
                            for node_name in pipeline.get_node_order():
                                if node_name in result.node_outputs:
                                    output = result.node_outputs[node_name]
                                    with st.expander(
                                        f"**{node_name}** ({output.get('model', 'N/A')})",
                                        expanded=(
                                            node_name == pipeline.finish_points[0]
                                        ),
                                    ):
                                        st.markdown(output.get("content", ""))
                                        st.caption(
                                            f"Latency: {output.get('latency_ms', 0)}ms"
                                        )

                            st.subheader("Final Output")
                            st.markdown(result.final_output)

                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Total Latency", f"{result.total_latency_ms}ms"
                                )
                            with col2:
                                st.metric(
                                    "Tokens",
                                    f"{result.total_tokens.get('input_tokens', 0) + result.total_tokens.get('output_tokens', 0)}",
                                )
                            with col3:
                                st.metric("Checkpoints", result.checkpoints_created)

                        with result_tab2:
                            if result.retry_attempts:
                                st.subheader(
                                    f"Retry Attempts ({len(result.retry_attempts)})"
                                )
                                for attempt in result.retry_attempts:
                                    status = "✅" if attempt.success else "❌"
                                    with st.expander(
                                        f"{status} {attempt.node_name} - Attempt #{attempt.attempt_number}"
                                    ):
                                        st.write(
                                            f"**Error Type:** {attempt.error_type.value}"
                                        )
                                        st.write(
                                            f"**Error:** {attempt.error_message[:200]}..."
                                        )
                                        if attempt.fix_type:
                                            st.write(
                                                f"**Fix Applied:** {attempt.fix_type.value}"
                                            )
                                        if attempt.fix_details:
                                            st.write(
                                                f"**Fix Details:** {attempt.fix_details}"
                                            )
                                        st.write(f"**Latency:** {attempt.latency_ms}ms")
                            else:
                                st.info(
                                    "No retry attempts - execution succeeded on first try!"
                                )

                            if result.escalations:
                                st.subheader(f"Escalations ({len(result.escalations)})")
                                for esc in result.escalations:
                                    st.error(f"**{esc.node_name}**: {esc.reason}")
                                    st.write(f"**Error:** {esc.error_summary[:300]}...")
                                    st.write("**Suggested Actions:**")
                                    for action in esc.suggested_actions[:5]:
                                        st.write(f"  • {action}")

                        with result_tab3:
                            st.subheader("Execution Trace")
                            for event in result.trace_log:
                                timestamp = event.get("timestamp", "")[:19]
                                event_type = event.get("event", "")
                                data = event.get("data", {})

                                icon = {
                                    "execution_start": "🚀",
                                    "node_start": "▶️",
                                    "node_success": "✅",
                                    "node_error": "❌",
                                    "retry_strategy": "🔄",
                                    "debug_analysis": "🔍",
                                    "fix_applied": "🔧",
                                    "checkpoint_saved": "💾",
                                    "escalation_created": "🚨",
                                    "execution_complete": "🏁",
                                }.get(event_type, "•")

                                st.write(f"{icon} **{timestamp}** - {event_type}")
                                if data:
                                    st.caption(str(data)[:100])

                    else:
                        st.warning("Please enter input text")

    # -------------------------------------------------------------------------
    # VIEW ESCALATIONS
    # -------------------------------------------------------------------------
    elif auto_subtab == "View Escalations":
        st.subheader("Escalations")

        escalation_mgr = EscalationManager()
        escalations = escalation_mgr.list_escalations(unresolved_only=False, limit=20)

        if not escalations:
            st.info("No escalations yet.")
        else:
            # Filter
            show_unresolved = st.checkbox("Show unresolved only", value=True)
            filtered = [e for e in escalations if not show_unresolved or not e.resolved]

            st.write(f"**Showing {len(filtered)} escalations**")

            for esc in filtered:
                status = "✅ Resolved" if esc.resolved else "❌ Unresolved"
                with st.expander(
                    f"**{esc.node_name}** - {esc.timestamp[:10]} ({status})"
                ):
                    st.write(f"**Pipeline:** {esc.pipeline_id}")
                    st.write(f"**Reason:** {esc.reason}")
                    st.write(f"**Error:** {esc.error_summary[:500]}")

                    st.write("**Suggested Actions:**")
                    for action in esc.suggested_actions:
                        st.write(f"  • {action}")

                    if not esc.resolved:
                        resolution = st.text_input(
                            "Resolution notes:",
                            key=f"resolve_{esc.escalation_id}",
                        )
                        if st.button(
                            "Mark Resolved", key=f"resolve_btn_{esc.escalation_id}"
                        ):
                            escalation_mgr.resolve(
                                esc.escalation_id,
                                resolution or "Manually resolved",
                            )
                            st.success("Escalation resolved")
                            st.rerun()

    # -------------------------------------------------------------------------
    # MANAGE CHECKPOINTS
    # -------------------------------------------------------------------------
    else:
        st.subheader("Checkpoints")

        checkpoint_mgr = CheckpointManager()

        # Get all pipeline IDs from saved pipelines
        saved = list_pipelines()
        pipeline_ids = list(set(p["name"] for p in saved))

        if not pipeline_ids:
            st.info("No pipelines with checkpoints yet.")
        else:
            selected_pipeline_id = st.selectbox(
                "Filter by pipeline:",
                options=["All"] + pipeline_ids,
                key="checkpoint_pipeline_filter",
            )

            filter_id = None if selected_pipeline_id == "All" else selected_pipeline_id
            checkpoints = checkpoint_mgr.list_checkpoints(pipeline_id=filter_id)

            if not checkpoints:
                st.info("No checkpoints found.")
            else:
                # Summary
                summary = checkpoint_mgr.get_checkpoint_summary(filter_id or "")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", summary.get("total", 0))
                with col2:
                    st.metric("Approved", summary.get("approved", 0))
                with col3:
                    st.metric("Pending", summary.get("pending", 0))
                with col4:
                    st.metric("Rejected", summary.get("rejected", 0))

                st.divider()

                for ckpt in checkpoints[:20]:
                    status = (
                        "✅ Approved"
                        if ckpt.approved
                        else ("❌ Rejected" if ckpt.approved is False else "⏳ Pending")
                    )
                    with st.expander(
                        f"**{ckpt.node_name}** - {ckpt.timestamp[:16]} ({status})"
                    ):
                        st.write(f"**Checkpoint ID:** {ckpt.checkpoint_id}")
                        st.write(f"**Type:** {ckpt.checkpoint_type}")

                        if ckpt.approved is None:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(
                                    "Approve", key=f"approve_{ckpt.checkpoint_id}"
                                ):
                                    checkpoint_mgr.approve(ckpt.checkpoint_id)
                                    st.success("Checkpoint approved")
                                    st.rerun()
                            with col2:
                                reject_reason = st.text_input(
                                    "Rejection reason:",
                                    key=f"reject_reason_{ckpt.checkpoint_id}",
                                )
                                if st.button(
                                    "Reject", key=f"reject_{ckpt.checkpoint_id}"
                                ):
                                    checkpoint_mgr.reject(
                                        ckpt.checkpoint_id,
                                        reject_reason or "Rejected by user",
                                    )
                                    st.warning("Checkpoint rejected")
                                    st.rerun()


# =============================================================================
# MEMORY TAB
# =============================================================================

with tab_memory:
    st.header("Cross-Session Memory")
    st.caption("Search, browse, and synthesize outputs across pipeline runs")

    memory_subtab = st.radio(
        "Mode:",
        ["Search Memory", "Pipeline History", "Synthesis", "Namespace Stats"],
        horizontal=True,
        key="memory_subtab",
    )

    st.divider()

    # -------------------------------------------------------------------------
    # SEARCH MEMORY
    # -------------------------------------------------------------------------
    if memory_subtab == "Search Memory":
        st.subheader("Semantic Search")

        search_query = st.text_input(
            "Search query:",
            placeholder="Enter a query to search past outputs...",
            key="memory_search_query",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            search_limit = st.number_input(
                "Max results:",
                min_value=1,
                max_value=50,
                value=10,
                key="memory_search_limit",
            )
        with col2:
            min_score = st.slider(
                "Min similarity:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                key="memory_min_score",
            )
        with col3:
            success_only = st.checkbox(
                "Successful runs only", value=True, key="memory_success_only"
            )

        if st.button("Search", type="primary", key="memory_search_btn"):
            if search_query:
                try:
                    store = get_memory_store()
                    store.set_project(st.session_state.current_project)

                    results = store.search_pipeline_outputs(
                        query=search_query,
                        project=st.session_state.current_project,
                        success_only=success_only,
                        limit=search_limit,
                        min_score=min_score,
                    )

                    if results:
                        st.success(f"Found {len(results)} results")

                        for i, result in enumerate(results, 1):
                            meta = result["metadata"]
                            with st.expander(
                                f"**{i}. {meta.get('pipeline_name', 'Unknown')}** "
                                f"(score: {result['score']:.2f})"
                            ):
                                st.write(f"**Date:** {result['timestamp'][:10]}")
                                st.write(
                                    f"**Pipeline:** {meta.get('pipeline_name', 'N/A')}"
                                )
                                st.write(
                                    f"**Latency:** {meta.get('latency_ms', 'N/A')}ms"
                                )

                                if meta.get("input_preview"):
                                    st.write("**Input:**")
                                    st.caption(meta["input_preview"][:300])

                                st.write("**Output:**")
                                st.markdown(result["content_preview"])

                                if meta.get("nodes"):
                                    st.write(f"**Nodes:** {meta['nodes']}")
                    else:
                        st.info("No matching results found")

                except Exception as e:
                    st.error(f"Search failed: {e}")
            else:
                st.warning("Please enter a search query")

    # -------------------------------------------------------------------------
    # PIPELINE HISTORY
    # -------------------------------------------------------------------------
    elif memory_subtab == "Pipeline History":
        st.subheader("Pipeline Run History")

        # Get pipelines for filtering
        saved = list_pipelines()
        pipeline_names = ["All"] + [p["name"] for p in saved]

        col1, col2 = st.columns(2)
        with col1:
            filter_pipeline = st.selectbox(
                "Filter by pipeline:",
                options=pipeline_names,
                key="memory_history_pipeline",
            )
        with col2:
            history_limit = st.number_input(
                "Max results:",
                min_value=5,
                max_value=100,
                value=20,
                key="memory_history_limit",
            )

        try:
            store = get_memory_store()
            store.set_project(st.session_state.current_project)

            pipeline_filter = None if filter_pipeline == "All" else filter_pipeline
            runs = store.get_pipeline_history(
                pipeline_name=pipeline_filter,
                project=st.session_state.current_project,
                limit=history_limit,
            )

            if runs:
                st.write(f"**Showing {len(runs)} runs**")

                for run in runs:
                    status = "✅" if run.success else "❌"
                    with st.expander(
                        f"{status} **{run.pipeline_name}** - {run.timestamp[:16]} ({run.total_latency_ms}ms)"
                    ):
                        st.write(f"**Run ID:** {run.id}")
                        st.write(f"**Session:** {run.session_id[:16]}...")

                        st.write("**Input:**")
                        st.caption(run.input_text[:500])

                        st.write("**Output:**")
                        st.markdown(run.final_output[:1000])

                        st.write(
                            f"**Tokens:** {run.total_tokens.get('input_tokens', 0)} in, "
                            f"{run.total_tokens.get('output_tokens', 0)} out"
                        )

                        if run.node_outputs:
                            st.write("**Node Outputs:**")
                            for node_name, output in run.node_outputs.items():
                                st.caption(
                                    f"  • {node_name}: {output.get('content', '')[:100]}..."
                                )

                        if run.errors:
                            st.error(f"Errors: {len(run.errors)}")
            else:
                st.info("No pipeline runs found for this project")

        except Exception as e:
            st.error(f"Failed to load history: {e}")

    # -------------------------------------------------------------------------
    # SYNTHESIS
    # -------------------------------------------------------------------------
    elif memory_subtab == "Synthesis":
        st.subheader("Multi-Run Synthesis")
        st.caption("Combine, compare, or extract patterns from multiple pipeline runs")

        try:
            store = get_memory_store()
            store.set_project(st.session_state.current_project)

            # Get recent runs for selection
            runs = store.get_pipeline_history(
                project=st.session_state.current_project,
                limit=30,
                success_only=True,
            )

            if runs:
                # Create selection options
                run_options = {
                    f"{r.pipeline_name} - {r.timestamp[:16]} ({r.id})": r.id
                    for r in runs
                }

                selected_runs = st.multiselect(
                    "Select runs to synthesize:",
                    options=list(run_options.keys()),
                    key="synthesis_runs",
                )

                synthesis_type = st.selectbox(
                    "Synthesis type:",
                    options=["combine", "compare", "patterns", "best_of", "timeline"],
                    format_func=lambda x: {
                        "combine": "Combine - Merge outputs into unified result",
                        "compare": "Compare - Analyze differences and similarities",
                        "patterns": "Patterns - Extract common themes",
                        "best_of": "Best Of - Select and enhance best output",
                        "timeline": "Timeline - Chronological summary",
                    }[x],
                    key="synthesis_type",
                )

                custom_instructions = st.text_area(
                    "Custom instructions (optional):",
                    placeholder="Add specific instructions for the synthesis...",
                    key="synthesis_instructions",
                )

                if st.button("Synthesize", type="primary", key="synthesis_btn"):
                    if len(selected_runs) >= 2:
                        run_ids = [run_options[r] for r in selected_runs]

                        with st.spinner("Synthesizing outputs..."):
                            synthesizer = get_synthesizer()
                            result = synthesizer.synthesize(
                                run_ids=run_ids,
                                synthesis_type=SynthesisType(synthesis_type),
                                instructions=(
                                    custom_instructions if custom_instructions else None
                                ),
                                project=st.session_state.current_project,
                            )

                        st.success(f"Synthesis complete ({result.latency_ms}ms)")

                        st.subheader("Synthesized Output")
                        st.markdown(result.synthesized_output)

                        st.caption(
                            f"Model: {result.model_used} | "
                            f"Sources: {result.source_count} | "
                            f"Tokens: {result.token_usage.get('input_tokens', 0)} in, "
                            f"{result.token_usage.get('output_tokens', 0)} out"
                        )
                    else:
                        st.warning("Please select at least 2 runs to synthesize")

                # Quick patterns button
                st.divider()
                st.write("**Quick Actions:**")
                if st.button("Extract Patterns from Recent Runs", key="quick_patterns"):
                    with st.spinner("Analyzing patterns..."):
                        synthesizer = get_synthesizer()
                        result = synthesizer.extract_patterns(
                            project=st.session_state.current_project,
                            limit=10,
                        )
                    st.subheader("Pattern Analysis")
                    st.markdown(result.synthesized_output)

            else:
                st.info("No successful pipeline runs found to synthesize")

        except Exception as e:
            st.error(f"Synthesis error: {e}")

    # -------------------------------------------------------------------------
    # NAMESPACE STATS
    # -------------------------------------------------------------------------
    else:
        st.subheader("Memory Namespace Statistics")

        try:
            store = get_memory_store()

            # Current project stats
            stats = store.get_namespace_stats(st.session_state.current_project)

            st.write(f"**Current Project:** {st.session_state.current_project}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Namespace", stats.get("namespace", "N/A")[:16])
            with col2:
                st.metric("Vectors", stats.get("vector_count", 0))
            with col3:
                st.metric("Local Records", stats.get("local_records", 0))

            if stats.get("error"):
                st.warning(f"Stats error: {stats['error']}")

            st.divider()

            # List all namespaces
            st.write("**All Project Namespaces:**")
            namespaces = store.list_namespaces()

            if namespaces:
                for ns in namespaces:
                    with st.expander(f"**{ns.project}** ({ns.full_namespace})"):
                        st.write(f"Created: {ns.created_at[:10]}")

                        ns_stats = store.get_namespace_stats(ns.project)
                        st.write(f"Vectors: {ns_stats.get('vector_count', 0)}")
                        st.write(f"Local Records: {ns_stats.get('local_records', 0)}")

                        if st.button("Clear Memory", key=f"clear_{ns.project}"):
                            if store.clear_namespace(ns.project):
                                st.success(f"Cleared memory for {ns.project}")
                                st.rerun()
                            else:
                                st.error("Failed to clear namespace")
            else:
                st.info("No namespaces created yet")

        except Exception as e:
            st.error(f"Failed to load namespace stats: {e}")

# =============================================================================
# COST INTELLIGENCE TAB
# =============================================================================

with tab_cost:
    st.header("Cost Intelligence")

    from ai_orchestrator.cost import (
        get_cost_analytics,
        get_budget_manager,
        get_cost_estimator,
        get_cost_tracker,
        get_pricing_manager,
    )

    cost_mode = st.radio(
        "Mode:",
        ["Dashboard", "Budget", "Estimates", "History", "Export"],
        horizontal=True,
    )

    project = st.session_state.current_project

    if cost_mode == "Dashboard":
        st.subheader("Cost Dashboard")

        try:
            analytics = get_cost_analytics(project)
            stats = analytics.get_summary_stats()

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                daily_change = stats.get("daily_change", 0)
                delta_str = f"{daily_change:+.1%}" if daily_change else None
                st.metric("Today's Spend", f"${stats.get('today_cost', 0):.4f}", delta=delta_str)
            with col2:
                st.metric("Week Spend", f"${stats.get('week_cost', 0):.4f}")
            with col3:
                st.metric("Month Spend", f"${stats.get('month_cost', 0):.4f}")
            with col4:
                st.metric("Total Tokens", f"{stats.get('today_tokens', 0):,}")

            st.divider()

            # Spend trend chart
            try:
                import plotly.express as px
                import pandas as pd

                trend_data = analytics.get_cost_trend(days=30)
                if trend_data:
                    df = pd.DataFrame(trend_data)
                    fig = px.line(
                        df,
                        x="date",
                        y="cost",
                        title="Daily Spending (Last 30 Days)",
                        labels={"date": "Date", "cost": "Cost (USD)"},
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No cost data available yet. Run some pipelines to see trends.")

            except ImportError:
                st.warning("Install plotly for charts: pip install plotly")

            # Cost breakdown
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Cost by Model")
                model_costs = analytics.get_spend_by_model()
                if model_costs:
                    try:
                        import plotly.express as px

                        model_df = pd.DataFrame([
                            {"model": k, "cost": v["cost"]}
                            for k, v in model_costs.items()
                        ])
                        fig = px.pie(model_df, values="cost", names="model", title="Cost by Model")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        for model, data in model_costs.items():
                            st.write(f"**{model}**: ${data['cost']:.4f} ({data['calls']} calls)")
                else:
                    st.info("No cost data by model yet")

            with col2:
                st.subheader("Cost by Pipeline")
                pipeline_costs = analytics.get_spend_by_pipeline()
                if pipeline_costs:
                    try:
                        import plotly.express as px

                        pipeline_df = pd.DataFrame([
                            {"pipeline": k, "cost": v["cost"]}
                            for k, v in pipeline_costs.items()
                        ])
                        fig = px.bar(pipeline_df, x="pipeline", y="cost", title="Cost by Pipeline")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        for pipeline, data in pipeline_costs.items():
                            st.write(f"**{pipeline}**: ${data['cost']:.4f} ({data['runs']} runs)")
                else:
                    st.info("No cost data by pipeline yet")

        except Exception as e:
            st.error(f"Failed to load cost dashboard: {e}")

    elif cost_mode == "Budget":
        st.subheader("Budget Management")

        try:
            budget_mgr = get_budget_manager(project)
            config = budget_mgr.get_config()
            status = budget_mgr.get_status()

            # Budget configuration
            st.write("**Set Budget Limits**")
            col1, col2, col3 = st.columns(3)
            with col1:
                daily = st.number_input(
                    "Daily Limit ($)",
                    min_value=0.0,
                    value=config.daily_limit if config and config.daily_limit else 0.0,
                    step=1.0,
                )
            with col2:
                weekly = st.number_input(
                    "Weekly Limit ($)",
                    min_value=0.0,
                    value=config.weekly_limit if config and config.weekly_limit else 0.0,
                    step=5.0,
                )
            with col3:
                monthly = st.number_input(
                    "Monthly Limit ($)",
                    min_value=0.0,
                    value=config.monthly_limit if config and config.monthly_limit else 0.0,
                    step=10.0,
                )

            col1, col2 = st.columns(2)
            with col1:
                alerts_enabled = st.checkbox(
                    "Enable Budget Alerts",
                    value=config.alerts_enabled if config else True,
                )
            with col2:
                enforce = st.checkbox(
                    "Enforce Limits (block execution)",
                    value=config.enforce_limits if config else False,
                )

            if st.button("Save Budget Settings", type="primary"):
                budget_mgr.set_budget(
                    daily=daily if daily > 0 else None,
                    weekly=weekly if weekly > 0 else None,
                    monthly=monthly if monthly > 0 else None,
                    enforce=enforce,
                    alerts_enabled=alerts_enabled,
                )
                st.success("Budget settings saved!")
                st.rerun()

            st.divider()

            # Budget status
            st.write("**Current Status**")

            if status.daily_limit:
                pct = status.daily_percentage or 0
                st.progress(min(pct, 1.0), text=f"Daily: ${status.daily_usage:.4f} / ${status.daily_limit:.2f} ({pct:.1%})")

            if status.weekly_limit:
                pct = status.weekly_percentage or 0
                st.progress(min(pct, 1.0), text=f"Weekly: ${status.weekly_usage:.4f} / ${status.weekly_limit:.2f} ({pct:.1%})")

            if status.monthly_limit:
                pct = status.monthly_percentage or 0
                st.progress(min(pct, 1.0), text=f"Monthly: ${status.monthly_usage:.4f} / ${status.monthly_limit:.2f} ({pct:.1%})")

            if status.projected_monthly_spend > 0:
                st.info(f"Projected monthly spend: ${status.projected_monthly_spend:.2f}")

            # Alerts
            st.divider()
            st.write("**Budget Alerts**")
            alerts = budget_mgr.get_alerts(unacknowledged_only=True)
            if alerts:
                for alert in alerts:
                    icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "exceeded": "🛑"}.get(
                        alert.alert_level.value, "•"
                    )
                    with st.expander(f"{icon} {alert.message}"):
                        st.write(f"**Period:** {alert.period.value}")
                        st.write(f"**Usage:** ${alert.current_usage:.4f} / ${alert.limit:.2f}")
                        st.write(f"**Time:** {alert.timestamp[:16]}")
                        if st.button("Acknowledge", key=f"ack_{alert.id}"):
                            budget_mgr.acknowledge_alert(alert.id)
                            st.rerun()
            else:
                st.success("No unacknowledged alerts")

            if st.button("Clear All Alerts"):
                budget_mgr.clear_alerts()
                st.success("Alerts cleared")
                st.rerun()

        except Exception as e:
            st.error(f"Failed to load budget: {e}")

    elif cost_mode == "Estimates":
        st.subheader("Cost Estimation")

        try:
            estimator = get_cost_estimator()
            pricing_mgr = get_pricing_manager()

            est_type = st.radio("Estimate:", ["Single LLM Call", "Pipeline"], horizontal=True)

            if est_type == "Single LLM Call":
                col1, col2 = st.columns([3, 1])
                with col1:
                    prompt = st.text_area("Prompt:", height=150, key="estimate_prompt")
                with col2:
                    model = st.selectbox("Model:", options=available, key="estimate_model")
                    max_tokens = st.number_input("Max Tokens:", value=4096, step=256)

                system = st.text_input("System Prompt (optional):")

                if st.button("Estimate Cost", type="primary"):
                    if prompt:
                        estimate = estimator.estimate_llm_call(
                            prompt=prompt,
                            model=model,
                            system_prompt=system if system else None,
                            max_tokens=max_tokens,
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Low Estimate", f"${estimate.estimated_cost_low:.6f}")
                        with col2:
                            st.metric("Likely Estimate", f"${estimate.estimated_cost_likely:.6f}")
                        with col3:
                            st.metric("High Estimate", f"${estimate.estimated_cost_high:.6f}")

                        st.write(f"**Input Tokens:** {estimate.estimated_input_tokens:,}")
                        st.write(f"**Output Tokens (likely):** {estimate.estimated_output_tokens_likely:,}")

                        if estimate.warnings:
                            for warning in estimate.warnings:
                                st.warning(warning)
                    else:
                        st.warning("Enter a prompt to estimate")

            else:  # Pipeline
                pipelines = list_pipelines(project)
                if pipelines:
                    selected_pipeline = st.selectbox("Pipeline:", options=pipelines)
                    input_text = st.text_area("Input Text:", height=150, key="pipeline_estimate_input")

                    if st.button("Estimate Pipeline Cost", type="primary"):
                        if input_text:
                            pipeline = get_pipeline(selected_pipeline, project)
                            estimate = estimator.estimate_pipeline(pipeline, input_text)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Low Estimate", f"${estimate.estimated_cost_low:.6f}")
                            with col2:
                                st.metric("Likely Estimate", f"${estimate.estimated_cost_likely:.6f}")
                            with col3:
                                st.metric("High Estimate", f"${estimate.estimated_cost_high:.6f}")

                            st.divider()
                            st.write("**Cost by Node:**")
                            for node_est in estimate.node_estimates:
                                with st.expander(f"**{node_est.node_name}** ({node_est.model})"):
                                    st.write(f"Input Tokens: {node_est.input_tokens:,}")
                                    st.write(f"Output Tokens (likely): {node_est.output_tokens_likely:,}")
                                    st.write(f"Cost (likely): ${node_est.cost_likely:.6f}")

                            if estimate.warnings:
                                for warning in estimate.warnings:
                                    st.warning(warning)
                        else:
                            st.warning("Enter input text to estimate")
                else:
                    st.info("No pipelines created yet")

            # Show pricing table
            st.divider()
            st.write("**Current Pricing (per 1K tokens)**")
            all_pricing = pricing_mgr.get_all_pricing()
            pricing_data = []
            for model_id, pricing in all_pricing.items():
                pricing_data.append({
                    "Model": model_id,
                    "Provider": pricing.provider,
                    "Input": f"${pricing.input_cost_per_1k:.4f}",
                    "Output": f"${pricing.output_cost_per_1k:.4f}",
                })
            if pricing_data:
                st.dataframe(pricing_data, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Failed to load estimator: {e}")

    elif cost_mode == "History":
        st.subheader("Cost History")

        try:
            tracker = get_cost_tracker(project)

            col1, col2 = st.columns(2)
            with col1:
                limit = st.slider("Records to show:", 10, 200, 50)
            with col2:
                pipelines = list_pipelines(project)
                filter_pipeline = st.selectbox(
                    "Filter by Pipeline:",
                    options=["All"] + pipelines,
                )

            records = tracker.get_recent_records(
                limit=limit,
                pipeline_name=filter_pipeline if filter_pipeline != "All" else None,
            )

            if records:
                st.write(f"**Showing {len(records)} records**")

                for record in records:
                    status = "✅" if record.success else "❌"
                    with st.expander(
                        f"{status} **{record.pipeline_name or 'Single Call'}** - "
                        f"${record.total_cost:.6f} - {record.timestamp[:16]}"
                    ):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Cost:** ${record.total_cost:.6f}")
                            if record.estimated_cost:
                                st.write(f"**Estimated:** ${record.estimated_cost:.6f}")
                                variance = record.estimate_variance or 0
                                st.write(f"**Variance:** {variance:+.1%}")
                        with col2:
                            st.write(f"**Input Tokens:** {record.total_input_tokens:,}")
                            st.write(f"**Output Tokens:** {record.total_output_tokens:,}")
                        with col3:
                            st.write(f"**Latency:** {record.latency_ms:,}ms")
                            st.write(f"**Models:** {', '.join(record.models_used)}")

                        if record.node_costs:
                            st.write("**Node Breakdown:**")
                            for node in record.node_costs:
                                st.write(
                                    f"- {node.node_name} ({node.model}): "
                                    f"${node.cost:.6f} ({node.input_tokens}+{node.output_tokens} tokens)"
                                )
            else:
                st.info("No cost records yet. Run some pipelines to see history.")

        except Exception as e:
            st.error(f"Failed to load cost history: {e}")

    elif cost_mode == "Export":
        st.subheader("Export Cost Data")

        try:
            from datetime import datetime, timedelta

            analytics = get_cost_analytics(project)

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=30),
                )
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())

            if st.button("Export to CSV", type="primary"):
                try:
                    csv_path = analytics.export_csv(
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat(),
                    )

                    with open(csv_path, "r") as f:
                        csv_content = f.read()

                    st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name=f"cost_export_{start_date}_{end_date}.csv",
                        mime="text/csv",
                    )
                    st.success(f"Export ready! {csv_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")

            st.divider()

            # Estimation accuracy
            st.write("**Estimation Accuracy**")
            accuracy = analytics.get_estimation_accuracy(
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )

            if accuracy.get("records_with_estimates", 0) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy Rate", f"{accuracy.get('accuracy_rate', 0):.1%}")
                with col2:
                    st.metric("Avg Variance", f"{accuracy.get('avg_variance', 0):+.1%}")
                with col3:
                    st.metric("Total Records", accuracy.get("total_records", 0))

                st.write(
                    f"- Within 20% estimate: {accuracy.get('accurate', 0)}\n"
                    f"- Over-estimated: {accuracy.get('over_estimated', 0)}\n"
                    f"- Under-estimated: {accuracy.get('under_estimated', 0)}"
                )
            else:
                st.info("No records with cost estimates in selected range")

        except Exception as e:
            st.error(f"Failed to load export: {e}")

# =============================================================================
# INTEGRATIONS TAB
# =============================================================================

with tab_integrations:
    st.header("External Integrations")

    from ai_orchestrator.config import get_integration_status

    integration_status = get_integration_status()

    # Status indicators
    col1, col2 = st.columns(2)
    with col1:
        if integration_status["notion"]:
            st.success("Notion: Connected")
        else:
            st.warning("Notion: Not configured (set NOTION_API_KEY)")

    with col2:
        if integration_status["google_docs"]:
            st.success("Google Docs: Credentials found")
        else:
            st.warning("Google Docs: Not configured (add google_credentials.json)")

    # Sub-tabs for each integration
    integ_subtab = st.radio(
        "Select Integration:",
        ["Notion", "Google Docs"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    # -------------------------------------------------------------------------
    # NOTION
    # -------------------------------------------------------------------------
    if integ_subtab == "Notion":
        if not integration_status["notion"]:
            st.info(
                "To use Notion integration, set the `NOTION_API_KEY` environment variable "
                "or add it to the `.env` file."
            )
        else:
            from ai_orchestrator.integrations import get_notion_client

            notion_action = st.radio(
                "Action:",
                ["Read Page", "Query Database", "Create Project Page", "Log Pipeline Run"],
                horizontal=True,
            )

            try:
                notion = get_notion_client()

                if notion_action == "Read Page":
                    st.subheader("Read Notion Page")

                    page_id = st.text_input(
                        "Page ID or URL:",
                        placeholder="https://notion.so/... or page ID",
                    )

                    if st.button("Read Page", type="primary") and page_id:
                        with st.spinner("Reading page..."):
                            try:
                                page = notion.get_page(page_id)
                                st.success(f"**{page['title']}**")
                                st.write(f"URL: {page['url']}")
                                st.divider()
                                st.markdown(page["content"])

                                # Option to use as pipeline input
                                if st.button("Use as Pipeline Input"):
                                    st.session_state["pipeline_input_from_notion"] = page[
                                        "content"
                                    ]
                                    st.info("Content saved. Go to Pipelines tab to use it.")
                            except Exception as e:
                                st.error(f"Failed to read page: {e}")

                elif notion_action == "Query Database":
                    st.subheader("Query Notion Database")

                    db_id = st.text_input(
                        "Database ID or URL:",
                        placeholder="https://notion.so/... or database ID",
                    )

                    limit = st.slider("Max results:", 5, 100, 20)

                    if st.button("Query Database", type="primary") and db_id:
                        with st.spinner("Querying database..."):
                            try:
                                results = notion.query_database(db_id, limit=limit)
                                st.success(f"Found {len(results)} pages")

                                for page in results:
                                    with st.expander(f"**{page['title']}**"):
                                        st.write(f"ID: `{page['id']}`")
                                        st.write(f"URL: {page['url']}")
                                        st.write(
                                            f"Last edited: {page['last_edited_time'][:10]}"
                                        )

                                        if st.button("Read Full", key=f"read_{page['id']}"):
                                            full_page = notion.get_page(page["id"])
                                            st.markdown(full_page["content"])
                            except Exception as e:
                                st.error(f"Failed to query database: {e}")

                elif notion_action == "Create Project Page":
                    st.subheader("Create Project Page")

                    db_id = st.text_input(
                        "Projects Database ID:",
                        placeholder="Database ID where project will be created",
                    )

                    project_name = st.text_input("Project Name:")
                    description = st.text_area("Description:", height=100)
                    status = st.selectbox(
                        "Initial Status:", ["Not Started", "In Progress", "Completed"]
                    )

                    if (
                        st.button("Create Project", type="primary")
                        and db_id
                        and project_name
                    ):
                        with st.spinner("Creating project page..."):
                            try:
                                result = notion.create_project_page(
                                    database_id=db_id,
                                    project_name=project_name,
                                    description=description,
                                    status=status,
                                )
                                st.success(f"Created project: **{project_name}**")
                                st.write(f"URL: {result['url']}")
                            except Exception as e:
                                st.error(f"Failed to create project: {e}")

                elif notion_action == "Log Pipeline Run":
                    st.subheader("Log Pipeline Run to Notion")

                    db_id = st.text_input(
                        "Pipeline Logs Database ID:",
                        placeholder="Database ID for pipeline logs",
                    )

                    # Check for recent pipeline results
                    if "pipeline_results" in st.session_state:
                        result = st.session_state.pipeline_results
                        st.info(
                            f"Recent pipeline result available (Success: {result.success})"
                        )

                        if st.button("Log Recent Result", type="primary") and db_id:
                            with st.spinner("Logging to Notion..."):
                                try:
                                    log_result = notion.log_pipeline_run(
                                        database_id=db_id,
                                        pipeline_name="UI Pipeline",
                                        input_text=st.session_state.get(
                                            "last_pipeline_input", "N/A"
                                        ),
                                        output_text=result.final_output,
                                        success=result.success,
                                        latency_ms=result.total_latency_ms,
                                        model_info=", ".join(
                                            output.get("model", "")
                                            for output in result.node_outputs.values()
                                        ),
                                    )
                                    st.success("Logged pipeline run to Notion!")
                                    st.write(f"URL: {log_result['url']}")
                                except Exception as e:
                                    st.error(f"Failed to log: {e}")
                    else:
                        st.info("No recent pipeline results. Run a pipeline first.")

            except Exception as e:
                st.error(f"Failed to initialize Notion client: {e}")

    # -------------------------------------------------------------------------
    # GOOGLE DOCS
    # -------------------------------------------------------------------------
    elif integ_subtab == "Google Docs":
        if not integration_status["google_docs"]:
            st.info(
                "To use Google Docs integration, add `google_credentials.json` "
                "to the `_system/` directory."
            )
        else:
            from ai_orchestrator.integrations import get_google_docs_client

            gdocs_action = st.radio(
                "Action:",
                ["Read Document", "Create Document", "Update Document", "List Documents"],
                horizontal=True,
            )

            try:
                gdocs = get_google_docs_client()

                if gdocs_action == "Read Document":
                    st.subheader("Read Google Doc")

                    doc_id = st.text_input(
                        "Document ID or URL:",
                        placeholder="https://docs.google.com/document/d/... or doc ID",
                    )

                    if st.button("Read Document", type="primary") and doc_id:
                        with st.spinner("Reading document..."):
                            try:
                                doc = gdocs.get_document(doc_id)
                                st.success(f"**{doc['title']}**")
                                st.write(f"URL: {doc['url']}")
                                st.divider()
                                st.text_area(
                                    "Content:", doc["content"], height=400, disabled=True
                                )

                                # Option to use as pipeline input
                                if st.button("Use as Pipeline Input"):
                                    st.session_state["pipeline_input_from_gdocs"] = doc[
                                        "content"
                                    ]
                                    st.info("Content saved. Go to Pipelines tab to use it.")
                            except Exception as e:
                                st.error(f"Failed to read document: {e}")

                elif gdocs_action == "Create Document":
                    st.subheader("Create Google Doc")

                    title = st.text_input("Document Title:")
                    content = st.text_area("Content:", height=200)

                    # Check for pipeline output to use
                    if "pipeline_results" in st.session_state:
                        if st.checkbox("Use recent pipeline output as content"):
                            content = st.session_state.pipeline_results.final_output
                            st.text_area(
                                "Pipeline Output Preview:",
                                content[:500] + "..." if len(content) > 500 else content,
                                disabled=True,
                            )

                    if st.button("Create Document", type="primary") and title:
                        with st.spinner("Creating document..."):
                            try:
                                result = gdocs.create_document(title=title, content=content)
                                st.success(f"Created document: **{title}**")
                                st.write(f"URL: {result['url']}")
                            except Exception as e:
                                st.error(f"Failed to create document: {e}")

                elif gdocs_action == "Update Document":
                    st.subheader("Update Google Doc")

                    doc_id = st.text_input("Document ID or URL:")
                    new_content = st.text_area("New Content:", height=200)

                    update_mode = st.radio(
                        "Update Mode:", ["Replace All", "Append"], horizontal=True
                    )

                    if st.button("Update Document", type="primary") and doc_id and new_content:
                        with st.spinner("Updating document..."):
                            try:
                                if update_mode == "Replace All":
                                    result = gdocs.update_document(doc_id, new_content)
                                else:
                                    result = gdocs.append_to_document(doc_id, new_content)

                                st.success("Document updated!")
                                st.write(f"URL: {result['url']}")
                            except Exception as e:
                                st.error(f"Failed to update document: {e}")

                elif gdocs_action == "List Documents":
                    st.subheader("List Google Docs")

                    limit = st.slider("Max results:", 5, 50, 20)

                    if st.button("List Documents", type="primary"):
                        with st.spinner("Listing documents..."):
                            try:
                                docs = gdocs.list_documents(limit=limit)
                                st.success(f"Found {len(docs)} documents")

                                for doc in docs:
                                    with st.expander(f"**{doc['title']}**"):
                                        st.write(f"ID: `{doc['id']}`")
                                        st.write(f"URL: {doc['url']}")
                                        st.write(f"Modified: {doc['modified_time'][:10]}")

                                        if st.button("Read", key=f"read_gdoc_{doc['id']}"):
                                            full_doc = gdocs.get_document(doc["id"])
                                            st.text_area(
                                                "Content:",
                                                full_doc["content"],
                                                height=200,
                                                disabled=True,
                                            )
                            except Exception as e:
                                st.error(f"Failed to list documents: {e}")

            except Exception as e:
                st.error(f"Failed to initialize Google Docs client: {e}")

# =============================================================================
# BROWSER ACTIONS TAB
# =============================================================================

with tab_browser:
    st.header("Browser Automation")
    st.caption("Automate web interactions with Playwright - Human-in-the-loop safety")

    # Check if playwright is available
    try:
        from ai_orchestrator.browser import (
            get_playwright_client,
            BrowserAction,
            BrowserActionType,
            ActionCategory,
            ApprovalStatus,
            ActionQueue,
            CredentialManager,
            create_action,
        )
        from ai_orchestrator.config import BROWSER_DATA_DIR, BROWSER_QUEUE_DIR

        playwright_available = True
    except ImportError as e:
        playwright_available = False
        st.error(
            f"Browser automation not available. Install dependencies: "
            f"`pip install playwright cryptography && playwright install chromium`"
        )
        st.stop()

    if playwright_available:
        browser_subtab = st.radio(
            "Mode:",
            ["Quick Actions", "Action Queue", "Credentials", "History"],
            horizontal=True,
            key="browser_subtab",
        )

        st.divider()

        # -------------------------------------------------------------------------
        # QUICK ACTIONS
        # -------------------------------------------------------------------------
        if browser_subtab == "Quick Actions":
            st.subheader("Execute Browser Actions")

            # Configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                headless = st.checkbox("Headless mode", value=True, key="browser_headless")
            with col2:
                dry_run = st.checkbox("Dry run (simulate submit)", value=True, key="browser_dry_run")
            with col3:
                take_screenshots = st.checkbox("Capture screenshots", value=True, key="browser_screenshots")

            st.divider()

            # Action builder
            action_type = st.selectbox(
                "Action type:",
                options=[t.value for t in BrowserActionType],
                format_func=lambda x: {
                    "navigate": "Navigate - Go to URL",
                    "fill": "Fill - Enter text in field",
                    "click": "Click - Click element",
                    "extract": "Extract - Get text from element",
                    "screenshot": "Screenshot - Capture page",
                    "wait": "Wait - Wait for element",
                    "select": "Select - Choose from dropdown",
                    "hover": "Hover - Hover over element",
                    "scroll": "Scroll - Scroll page",
                    "execute_js": "Execute JS - Run JavaScript",
                }.get(x, x),
                key="browser_action_type",
            )

            # Action-specific inputs
            target_url = None
            target_selector = None
            value = None

            if action_type == "navigate":
                target_url = st.text_input("URL:", placeholder="https://example.com", key="browser_url")
            elif action_type in ["fill", "select"]:
                target_selector = st.text_input(
                    "Selector (CSS/XPath):",
                    placeholder="#email, input[name='email']",
                    key="browser_selector",
                )
                value = st.text_input("Value to enter:", key="browser_value")
            elif action_type in ["click", "extract", "wait", "hover"]:
                target_selector = st.text_input(
                    "Selector (CSS/XPath):",
                    placeholder="button[type='submit']",
                    key="browser_selector2",
                )
            elif action_type == "execute_js":
                value = st.text_area("JavaScript code:", key="browser_js", height=100)
            elif action_type == "scroll":
                target_selector = st.text_input(
                    "Selector (optional, scrolls to element):",
                    placeholder="Leave empty to scroll page",
                    key="browser_scroll_selector",
                )

            description = st.text_input(
                "Description (optional):",
                placeholder="What this action does",
                key="browser_desc",
            )

            # Show category info
            if action_type in ["navigate", "extract", "screenshot", "wait", "hover", "scroll"]:
                st.success("Read-only action - auto-approved")
            elif action_type in ["fill", "select"]:
                st.info("Input action - reversible")
            elif action_type in ["click", "execute_js"]:
                st.warning("Submit action - requires approval if dry_run is off")

            if st.button("Execute Action", type="primary", key="browser_execute"):
                # Validation
                valid = True
                if action_type == "navigate" and not target_url:
                    st.error("URL is required for navigate action")
                    valid = False
                elif action_type in ["fill", "click", "extract", "wait", "select", "hover"] and not target_selector:
                    st.error("Selector is required for this action")
                    valid = False

                if valid:
                    try:
                        client = get_playwright_client(headless=headless, dry_run=dry_run)

                        # Start session if needed
                        if not client.is_active:
                            with st.spinner("Starting browser session..."):
                                session = client.start_session()
                                st.session_state.browser_session = session.to_dict()

                        # Create action
                        action = create_action(
                            action_type=action_type,
                            target_selector=target_selector or None,
                            target_url=target_url or None,
                            value=value or None,
                            description=description or f"{action_type} action",
                        )

                        # Auto-approve for execution
                        action.approval_status = ApprovalStatus.APPROVED

                        with st.spinner("Executing action..."):
                            result = client.execute_action(action, take_screenshots=take_screenshots)

                        # Store in history
                        st.session_state.browser_history.append({
                            "action": action.to_dict(),
                            "result": result.to_dict(),
                            "timestamp": result.timestamp,
                        })

                        if result.success:
                            st.success(f"Action completed in {result.latency_ms}ms")

                            if result.extracted_data:
                                st.subheader("Extracted Data")
                                st.text_area(
                                    "Content:",
                                    result.extracted_data,
                                    height=200,
                                    disabled=True,
                                    key="browser_result_data",
                                )

                            if result.screenshot_path:
                                st.subheader("Screenshot")
                                try:
                                    st.image(result.screenshot_path)
                                except Exception:
                                    st.caption(f"Screenshot saved: {result.screenshot_path}")

                            if result.page_url:
                                st.write(f"**Current URL:** {result.page_url}")
                        else:
                            st.error(f"Action failed: {result.error}")
                            if result.error_screenshot_path:
                                try:
                                    st.image(result.error_screenshot_path, caption="Error state")
                                except Exception:
                                    pass

                    except Exception as e:
                        st.error(f"Error: {e}")

            # Session control
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("End Session", key="browser_end_session"):
                    if st.session_state.browser_session:
                        try:
                            client = get_playwright_client()
                            client.end_session()
                            st.session_state.browser_session = None
                            st.success("Session ended")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error ending session: {e}")
            with col2:
                if st.session_state.browser_session:
                    session_id = st.session_state.browser_session.get("session_id", "unknown")
                    st.info(f"Active session: {session_id}")
                else:
                    st.caption("No active session")

        # -------------------------------------------------------------------------
        # ACTION QUEUE
        # -------------------------------------------------------------------------
        elif browser_subtab == "Action Queue":
            st.subheader("Action Approval Queue")
            st.caption("Review and approve actions before execution")

            BROWSER_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

            # Initialize or get queue
            if st.session_state.browser_queue is None:
                st.session_state.browser_queue = ActionQueue(BROWSER_QUEUE_DIR)

            queue = st.session_state.browser_queue
            pending = queue.get_pending()
            approved = queue.get_approved()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pending Approval", len(pending))
            with col2:
                st.metric("Ready to Execute", len(approved))
            with col3:
                stats = queue.get_stats()
                st.metric("Total Processed", stats.get("executed", 0))

            st.divider()

            # Pending actions
            if pending:
                st.subheader("Pending Approval")

                for entry in pending:
                    action = entry.action
                    with st.expander(
                        f"**{action.action_type.value}**: {action.description or action.target_url or action.target_selector}"
                    ):
                        st.write(f"**Category:** {action.category.value}")
                        st.write(f"**Created:** {entry.created_at[:16]}")

                        if entry.preview_screenshot:
                            try:
                                st.image(entry.preview_screenshot, caption="Preview")
                            except Exception:
                                pass

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Approve", key=f"approve_{action.action_id}", type="primary"):
                                queue.approve_action(action.action_id)
                                st.success("Approved!")
                                st.rerun()
                        with col2:
                            if st.button("Reject", key=f"reject_{action.action_id}"):
                                queue.reject_action(action.action_id, "Rejected by user")
                                st.warning("Rejected")
                                st.rerun()

                if st.button("Approve All Pending", key="approve_all_pending"):
                    count = queue.approve_all_pending()
                    st.success(f"Approved {count} actions")
                    st.rerun()
            else:
                st.info("No actions pending approval")

            # Execute approved actions
            st.divider()
            if approved:
                st.subheader("Execute Approved Actions")
                st.write(f"{len(approved)} action(s) ready to execute")

                if st.button("Execute All Approved", type="primary", key="execute_all_approved"):
                    try:
                        client = get_playwright_client()

                        if not client.is_active:
                            with st.spinner("Starting browser session..."):
                                session = client.start_session()
                                st.session_state.browser_session = session.to_dict()

                        with st.spinner("Executing actions..."):
                            for entry in approved:
                                result = client.execute_action(entry.action)
                                queue.mark_executed(entry.action.action_id)

                                # Store in history
                                st.session_state.browser_history.append({
                                    "action": entry.action.to_dict(),
                                    "result": result.to_dict(),
                                    "timestamp": result.timestamp,
                                })

                                if result.success:
                                    st.success(f"{entry.action.action_type.value}: OK")
                                else:
                                    st.error(f"{entry.action.action_type.value}: {result.error}")

                        st.rerun()
                    except Exception as e:
                        st.error(f"Error executing actions: {e}")
            else:
                st.caption("No approved actions to execute")

        # -------------------------------------------------------------------------
        # CREDENTIALS
        # -------------------------------------------------------------------------
        elif browser_subtab == "Credentials":
            st.subheader("Credential Management")
            st.caption("Securely store login credentials for browser automation")

            from ai_orchestrator.config import BROWSER_CREDENTIALS_DIR

            BROWSER_CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
            cred_mgr = CredentialManager(BROWSER_CREDENTIALS_DIR)

            # Check if master key is set
            status = cred_mgr.get_status()
            if not status["persistent"]:
                st.warning(
                    "BROWSER_MASTER_KEY not set. Credentials will not persist across restarts. "
                    "Add it to your .env file for persistent encrypted storage."
                )
            else:
                st.success("Credentials are encrypted and persistent")

            st.divider()

            # List existing credentials
            sites = cred_mgr.list_sites()

            if sites:
                st.subheader("Stored Credentials")
                for site_info in sites:
                    site_id = site_info["site_id"]
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        icons = []
                        if site_info["has_credentials"]:
                            icons.append("Password")
                        if site_info["has_state"]:
                            icons.append("Cookies")
                        st.write(f"**{site_id}** ({', '.join(icons)})")
                    with col2:
                        if st.button("Use", key=f"use_cred_{site_id}"):
                            st.info(f"Site '{site_id}' will be used for next browser session")
                    with col3:
                        if st.button("Delete", key=f"del_cred_{site_id}"):
                            cred_mgr.delete_credentials(site_id)
                            st.success(f"Deleted credentials for {site_id}")
                            st.rerun()
            else:
                st.info("No stored credentials")

            st.divider()

            # Add new credentials
            st.subheader("Add Credentials")

            site_id = st.text_input("Site identifier:", placeholder="github.com", key="cred_site")
            username = st.text_input("Username/Email:", key="cred_username")
            password = st.text_input("Password:", type="password", key="cred_password")

            if st.button("Save Credentials", type="primary", key="save_creds"):
                if site_id and (username or password):
                    cred_mgr.save_credentials(
                        site_id=site_id,
                        username=username or None,
                        password=password or None,
                    )
                    st.success(f"Credentials saved for {site_id}")
                    st.rerun()
                else:
                    st.warning("Site identifier and at least username or password required")

        # -------------------------------------------------------------------------
        # HISTORY
        # -------------------------------------------------------------------------
        else:
            st.subheader("Action History")

            history = st.session_state.browser_history

            if not history:
                st.info("No actions executed yet")
            else:
                st.write(f"**{len(history)} action(s) in history**")

                # Show most recent first, limit to 50
                for i, entry in enumerate(reversed(history[-50:])):
                    action = entry["action"]
                    result = entry["result"]

                    status_icon = "+" if result["success"] else "X"
                    action_desc = action.get("description") or action.get("target_url") or action.get("target_selector") or "N/A"

                    with st.expander(f"{status_icon} {action['action_type']}: {action_desc[:50]}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Timestamp:** {entry['timestamp'][:19]}")
                            st.write(f"**Latency:** {result['latency_ms']}ms")
                        with col2:
                            st.write(f"**Success:** {result['success']}")
                            if result.get("page_url"):
                                st.write(f"**URL:** {result['page_url'][:50]}...")

                        if result.get("extracted_data"):
                            st.write("**Extracted Data:**")
                            st.text_area(
                                "Data:",
                                result["extracted_data"][:1000],
                                disabled=True,
                                key=f"hist_data_{i}",
                                height=100,
                            )

                        if result.get("screenshot_path"):
                            try:
                                st.image(result["screenshot_path"], caption="Screenshot")
                            except Exception:
                                st.caption(f"Screenshot: {result['screenshot_path']}")

                        if result.get("error"):
                            st.error(f"Error: {result['error']}")

                if st.button("Clear History", key="clear_browser_history"):
                    st.session_state.browser_history = []
                    st.success("History cleared")
                    st.rerun()
