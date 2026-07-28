import os
import json
import time
import streamlit as st
import pandas as pd
import networkx as nx

from app.core.config import settings
from app.memory.semantic import SemanticMemory
from app.memory.decay import calculate_retention

st.set_page_config(
    page_title="Cognitive Memory Engine Visualizer",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Cognitive Memory Engine: Interactive State Visualizer")
st.markdown(
    "Visualizing human-inspired **Working Memory**, **Episodic SQLite Logs**, "
    "and **NetworkX Knowledge Graph** with **Ebbinghaus Decay**."
)

# Sidebar settings
st.sidebar.header("⚙️ Configuration")
st.sidebar.write(f"**Ollama Base URL:** `{settings.OLLAMA_BASE_URL}`")
st.sidebar.write(f"**LLM Model:** `{settings.OLLAMA_MODEL}`")
st.sidebar.write(f"**Working Max Tokens:** `{settings.WORKING_MEMORY_MAX_TOKENS}`")
st.sidebar.write(f"**Episodic DB:** `{settings.EPISODIC_DB_PATH}`")
st.sidebar.write(f"**Semantic Graph:** `{settings.SEMANTIC_GRAPH_PATH}`")

graph_file = settings.SEMANTIC_GRAPH_PATH

if not os.path.exists(graph_file):
    st.info(f"No semantic graph file found at `{graph_file}` yet. Run chat queries or `test_step4.py` to populate memory graph!")
else:
    semantic_mem = SemanticMemory(graph_path=graph_file)
    semantic_mem.load()
    graph = semantic_mem.graph

    now = time.time()

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Graph Nodes", len(graph.nodes))
    col2.metric("Graph Edges", len(graph.edges))
    col3.metric("Model Tag", settings.OLLAMA_MODEL)

    st.subheader("🕸️ Semantic Knowledge Graph Triples")
    triples = semantic_mem.get_triples()
    if triples:
        df_triples = pd.DataFrame(triples, columns=["Subject", "Predicate (Relation)", "Object"])
        st.dataframe(df_triples, use_container_width=True)
    else:
        st.write("No triples in graph.")

    st.subheader("📉 Ebbinghaus Memory Retention Decay (R = e^(-t/S))")
    node_data = []
    for n, data in graph.nodes(data=True):
        last_acc = data.get("last_accessed", now)
        strength = data.get("strength", 1.0)
        retention = calculate_retention(last_accessed=last_acc, current_time=now, strength=strength)
        delta_mins = round((now - last_acc) / 60.0, 1)
        node_data.append({
            "Node": n,
            "Strength (S)": strength,
            "Last Accessed (mins ago)": delta_mins,
            "Retention (R)": retention,
        })

    if node_data:
        df_nodes = pd.DataFrame(node_data)
        st.dataframe(df_nodes, use_container_width=True)
        st.bar_chart(df_nodes.set_index("Node")["Retention (R)"])

st.markdown("---")
st.caption("Cognitive-Memory-Engine | Local-First Cognitive Architecture")
