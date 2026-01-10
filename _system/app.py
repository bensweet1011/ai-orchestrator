"""
AI Orchestrator - Streamlit UI
Your command center for orchestrating LLMs.
"""

import streamlit as st
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

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

# Page config
st.set_page_config(
    page_title="AI Orchestrator", page_icon="🎯", layout="wide", initial_sidebar_state="expanded"
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


def check_setup() -> tuple[bool, List[str]]:
    """Check if system is properly configured."""
    issues = []
    keys = check_required_keys()

    if not keys["pinecone"]:
        issues.append("PINECONE_API_KEY not set")
    if not keys["openai"]:
        issues.append("OPENAI_API_KEY not set (required for embeddings)")
    if not keys["any_llm"]:
        issues.append("No LLM API keys set (need at least one of: ANTHROPIC, OPENAI, GOOGLE)")

    return len(issues) == 0, issues


def call_single_model(prompt: str, model: str, system: str = None) -> tuple[str, LLMResponse, int]:
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
            executor.submit(call_single_model, prompt, model, system): model for model in models
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
                current_prompt = (
                    f"Previous output:\n{response.content}\n\nBased on the above, {prompt}"
                )
        except Exception as e:
            results[model] = (None, str(e))
            if pass_output:
                break

    return results


def synthesize_outputs(outputs: Dict[str, tuple[LLMResponse, int]], original_prompt: str) -> str:
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


def log_interaction(prompt: str, response: LLMResponse, latency_ms: int, metadata: dict = None):
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

    existing_projects = ["default"] + [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]

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
                        st.write(f"**Prompt:** {r['metadata'].get('prompt_preview', 'N/A')}")
                        st.write(f"**Response:** {r['metadata'].get('response_preview', 'N/A')}")
            else:
                st.info("No matching interactions found")
        except Exception as e:
            st.error(f"Search failed: {e}")


# =============================================================================
# MAIN AREA - TABS
# =============================================================================

tab_chat, tab_pipelines = st.tabs(["💬 Chat", "🔧 Pipelines"])

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
            else (["claude"] if "claude" in available else [available[0]] if available else [])
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
                                f"{model} ({latency}ms)" if response else f"{model} (failed)"
                            ):
                                if response:
                                    st.markdown(response.content)
                                    log_interaction(prompt, response, latency, {"mode": "parallel"})
                                else:
                                    st.error(f"Failed: {latency}")

                        successful = {m: r for m, r in results.items() if r[0] is not None}
                        if len(successful) > 1:
                            st.subheader("Synthesized Result")
                            with st.spinner("Claude is synthesizing the best answer..."):
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
                        results = call_models_sequential(prompt, models, pass_output=pass_output)

                        for model, (response, latency) in results.items():
                            with st.expander(
                                f"{model} ({latency}ms)" if response else f"{model} (failed)"
                            ):
                                if response:
                                    st.markdown(response.content)
                                    log_interaction(
                                        prompt,
                                        response,
                                        latency,
                                        {
                                            "mode": mode,
                                            "sequence_position": list(results.keys()).index(model),
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
    st.caption("Build multi-step LLM workflows with user-defined model selection per node")

    pipeline_subtab = st.radio(
        "Mode:", ["Run Pipeline", "Create Pipeline", "Manage Pipelines"], horizontal=True
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
                selected_pipeline = st.selectbox("Select pipeline:", options=pipeline_names)

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
                                    st.success(f"Pipeline completed in {result.total_latency_ms}ms")
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
                                            expanded=(node_name == pipeline.finish_points[0]),
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
                            f"{node_name}:", options=available, key=f"template_llm_{node_name}"
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
                                pipeline = create_from_template(selected_template, llm_assignments)
                                with st.spinner("Executing pipeline..."):
                                    result = pipeline.run(pipeline_input)
                                    st.session_state.pipeline_results = result

                                if result.success:
                                    st.success(f"Completed in {result.total_latency_ms}ms")
                                else:
                                    st.error("Completed with errors")

                                for node_name in pipeline.get_node_order():
                                    if node_name in result.node_outputs:
                                        output = result.node_outputs[node_name]
                                        with st.expander(f"**{node_name}** ({output['model']})"):
                                            st.markdown(output["content"])

                                st.subheader("Final Output")
                                st.markdown(result.final_output)

                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.warning("Please enter input and select LLMs for all nodes")

                with col2:
                    save_name = st.text_input("Save as:", placeholder="my_pipeline")
                    if st.button("Save Configuration"):
                        if save_name and all(llm_assignments.values()):
                            try:
                                pipeline = create_from_template(
                                    selected_template, llm_assignments, new_name=save_name
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
        pipeline_name = st.text_input("Pipeline name:", placeholder="my_custom_pipeline")
        pipeline_desc = st.text_input("Description:", placeholder="What does this pipeline do?")

        st.divider()

        # Node builder
        st.write("**Define Nodes:**")

        if "builder_nodes" not in st.session_state:
            st.session_state.builder_nodes = []

        # Add node form
        with st.expander("Add New Node", expanded=len(st.session_state.builder_nodes) == 0):
            node_name = st.text_input("Node name:", placeholder="summarize", key="new_node_name")
            node_llm = st.selectbox("LLM:", options=available, key="new_node_llm")
            node_system = st.text_area(
                "System prompt:", placeholder="Instructions for this node...", key="new_node_system"
            )
            node_input = st.selectbox(
                "Input from:",
                options=["input (user text)"] + [n["name"] for n in st.session_state.builder_nodes],
                key="new_node_input",
            )

            if st.button("Add Node"):
                if node_name and node_llm:
                    input_key = "input" if node_input == "input (user text)" else node_input
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
                    st.write(f"**{i+1}. {node['name']}** ({node['llm']}) ← {node['input_key']}")
                with col2:
                    if st.button("⬆", key=f"up_{i}") and i > 0:
                        st.session_state.builder_nodes[i], st.session_state.builder_nodes[i - 1] = (
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
                        pipeline = BasePipeline(name=pipeline_name, description=pipeline_desc)

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

                        pipeline.set_entry_point(st.session_state.builder_nodes[0]["name"])
                        pipeline.set_finish_point(st.session_state.builder_nodes[-1]["name"])

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
