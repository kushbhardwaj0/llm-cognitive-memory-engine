import os
import sys
import time
import asyncio
import streamlit as st
import pandas as pd

# Ensure root directory is in sys.path for app imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.core.engine import CognitiveMemoryEngine
from app.memory.decay import calculate_retention

st.set_page_config(
    page_title="Cognitive Memory Engine - Chat & Live Memory Inspector",
    layout="wide",
)

# Helper to run async functions in Streamlit
def run_async(coro):
    return asyncio.run(coro)

# Initialize Engine in session state
if "engine" not in st.session_state:
    engine = CognitiveMemoryEngine(working_max_tokens=150)
    run_async(engine.initialize())
    st.session_state.engine = engine
    st.session_state.session_id = "live-demo-session"
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I am powered by Cognitive Memory Engine. Tell me facts about your architecture, and watch my Working, Episodic, and Semantic memory tiers update live in the right panel!",
        }
    ]

engine: CognitiveMemoryEngine = st.session_state.engine
session_id: str = st.session_state.session_id

# Header
st.title("Cognitive Memory Engine - Live Chat & 3-Tier Memory Inspector")
st.caption("Local-first agent cognitive architecture with Working (RAM), Episodic (SQLite), and Semantic (NetworkX Graph) memory.")

# Sidebar Controls
st.sidebar.header("Configuration")
st.sidebar.write(f"**Model:** `{settings.OLLAMA_MODEL}`")
st.sidebar.write(f"**Working Max Tokens:** `{engine.working_memory.max_tokens}`")

if st.sidebar.button("Run Consolidation Daemon Pass"):
    with st.spinner("Daemon consolidating un-processed SQLite episodes..."):
        count = run_async(engine.daemon.consolidate_once())
        st.sidebar.success(f"Consolidated {count} episodes into Knowledge Graph!")
        st.rerun()

if st.sidebar.button("Reset Chat & Memory State"):
    engine.working_memory.clear()
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "State reset. What would you like to discuss?",
        }
    ]
    st.rerun()

# Split Screen Layout: Left = Chat UI (60%), Right = Live Memory Inspector (40%)
col_chat, col_memory = st.columns([1.1, 0.9])

# =============================================================================
# LEFT PANEL: Chat Interface
# =============================================================================
with col_chat:
    st.subheader("Live Chat Interface")
    chat_container = st.container(height=520)

    with chat_container:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("Type your message to the agent...")
    if prompt:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Process through Cognitive Memory Engine
        with st.spinner("Cognitive Engine processing..."):
            response_text = run_async(engine.chat(session_id=session_id, user_message=prompt))

        # Append assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.rerun()

# =============================================================================
# RIGHT PANEL: Live 3-Tier Memory Inspector
# =============================================================================
with col_memory:
    st.subheader("Live 3-Tier Memory Inspector")
    
    memory_tabs = st.tabs([
        "1. Working Memory (RAM)",
        "2. Episodic Log (SQLite)",
        "3. Semantic Graph (Knowledge)",
    ])

    now = time.time()

    # -------------------------------------------------------------------------
    # Tier 1: Working Memory Buffer
    # -------------------------------------------------------------------------
    with memory_tabs[0]:
        st.markdown("#### Tier 1: Working Memory Sliding Window")
        st.caption("Active short-term conversation buffer in RAM. Pop/evicts when full.")

        current_tokens = engine.working_memory.total_tokens
        max_tokens = engine.working_memory.max_tokens
        usage_pct = min(1.0, current_tokens / max_tokens) if max_tokens > 0 else 0.0

        st.progress(usage_pct, text=f"Buffer Usage: {current_tokens} / {max_tokens} tokens ({int(usage_pct*100)}%)")

        active_msgs = engine.working_memory.get_messages()
        if active_msgs:
            active_data = [
                {
                    "Role": m.role.capitalize(),
                    "Content": m.content,
                    "Tokens": m.token_count,
                }
                for m in active_msgs
            ]
            st.dataframe(pd.DataFrame(active_data), use_container_width=True)
        else:
            st.info("Working memory buffer is currently empty.")

    # -------------------------------------------------------------------------
    # Tier 2: Episodic Memory Store (SQLite)
    # -------------------------------------------------------------------------
    with memory_tabs[1]:
        st.markdown("#### Tier 2: Episodic Memory Timeline (SQLite)")
        st.caption("Persistent raw event timeline log with consolidation tracking.")

        episodes_count = run_async(engine.episodic_memory.count_episodes(session_id=session_id))
        st.metric("Total Persistent SQLite Episodes", episodes_count)

        recent_episodes = run_async(engine.episodic_memory.get_recent_episodes(session_id=session_id, limit=15))
        if recent_episodes:
            ep_data = [
                {
                    "ID": ep.id[:8] + "...",
                    "Role": ep.role.capitalize(),
                    "Content": ep.content,
                    "Consolidated Status": "[Processed (1)]" if ep.consolidated else "[Pending (0)]",
                }
                for ep in recent_episodes
            ]
            st.dataframe(pd.DataFrame(ep_data), use_container_width=True)
        else:
            st.info("No episodes in SQLite store yet.")

    # -------------------------------------------------------------------------
    # Tier 3: Semantic Memory Knowledge Graph (NetworkX)
    # -------------------------------------------------------------------------
    with memory_tabs[2]:
        st.markdown("#### Tier 3: Semantic Knowledge Graph & Ebbinghaus Decay")
        st.caption("Extracted concepts, relationships, and retention probability score R = e^(-t/S).")

        graph = engine.semantic_memory.graph
        c1, c2 = st.columns(2)
        c1.metric("Graph Nodes", len(graph.nodes))
        c2.metric("Graph Edges", len(graph.edges))

        triples = engine.semantic_memory.get_triples()
        if triples:
            st.markdown("**Extracted Graph Triples:**")
            df_triples = pd.DataFrame(triples, columns=["Subject", "Predicate", "Object"])
            st.dataframe(df_triples, use_container_width=True)
        else:
            st.info("No graph triples extracted yet. Click 'Run Consolidation Daemon Pass' in the sidebar to consolidate pending episodes!")

        node_data = []
        for n, data in graph.nodes(data=True):
            last_acc = data.get("last_accessed", now)
            strength = data.get("strength", 1.0)
            retention = calculate_retention(last_accessed=last_acc, current_time=now, strength=strength)
            node_data.append({
                "Node Concept": n,
                "Strength (S)": strength,
                "Retention (R)": retention,
            })

        if node_data:
            st.markdown("**Ebbinghaus Retention Math Bar Chart:**")
            df_nodes = pd.DataFrame(node_data)
            st.bar_chart(df_nodes.set_index("Node Concept")["Retention (R)"])

st.markdown("---")
st.caption("Cognitive-Memory-Engine | Local-First Cognitive Architecture powered by Ollama & Qwen2.5")
