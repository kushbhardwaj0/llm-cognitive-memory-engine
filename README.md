# Cognitive-Memory-Engine 🧠

> **A Local-First, Human-Inspired Cognitive Memory Backend for LLM Agents Outperforming Naive Vector RAG on Long-Term Retention and Token Efficiency.**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Gateway-009688.svg)
![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5%3A7B-orange.svg)
![License](https://img.shields.io/badge/License-Source--Available-red.svg)

---

## 🏆 Key Benchmark Results

Empirical benchmarks comparing **Cognitive-Memory-Engine** against standard **Naive Vector RAG** (chunking + vector embeddings + top-K retrieval) on long-horizon agent context tasks:

| Metric Pillar | Naive Vector RAG | Cognitive Memory Engine | Performance Gain / Lead |
| :--- | :---: | :---: | :---: |
| **Long-Term Memory Retention Accuracy** | `33.3%` | **`83.3%`** | **`+50.0% Accuracy Lead`** 🚀 |
| **State Evolution (PostgreSQL → ScyllaDB)** | `0.0% (Failed)` | **`100.0% (Passed)`** | **`+100.0% State Accuracy`** 🎯 |
| **30-Turn Noise Haystack Recall** | `0.0%` | **`50.0%`** | **`+50.0% Noise Immunity`** 🛡️ |
| **Average Prompt Token Savings** | *Baseline* | **`~30.1% Savings`** | **`~30% Token Compression`** ⚡ |

### Why Cognitive Memory Outperforms Vector RAG:
1. **State Overwriting vs. Chunk Confusion**: Standard RAG stores outdated text blocks side-by-side with new ones (e.g., retrieving both *"I use PostgreSQL"* and *"I migrated to ScyllaDB"*), confusing the LLM. Cognitive Engine updates graph node state `(User) --[USES_DB]--> (ScyllaDB)`, resolving state conflicts cleanly.
2. **Spreading Activation Multi-Hop Retrieval**: Instead of retrieving disconnected keyword chunks, energy propagates across 2-3 hops over the NetworkX Knowledge Graph, stitching together complex multi-turn relationships.
3. **Ebbinghaus Memory Decay ($R = e^{-t/S}$)**: Non-recurring noise naturally decays over time while core user entities are reinforced.

---

## 🏛️ System Architecture & Memory Tiers

Standard RAG treats conversation history as static text blocks. **Cognitive-Memory-Engine** replaces static retrieval with a human-inspired 3-tiered memory pipeline:

```text
                       +-----------------------------+
                       |     User / Agent Prompt     |
                       +-----------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
|                     Cognitive Memory Engine Pipeline                       |
|                                                                           |
|  1. Working Memory (RAM) ---------> Sliding token window (max 2048)       |
|  2. Spreading Activation ---------> Multi-hop traversal over Knowledge    |
|                                     Graph to retrieve activated subgraphs |
|  3. Context Augmentation ---------> [System] + [Graph Triples] + [Buffer] |
|  4. Local LLM (Qwen2.5:7b) --------> Non-blocking Async Metal inference   |
+---------------------------------------------------------------------------+
                                      |
                                      | (Evicted Buffer Turns)
                                      v
+---------------------------------------------------------------------------+
|             Tier 2: Episodic Memory (aiosqlite Database)                 |
|             - Immutable, sequential timeline event log                    |
|             - Managed with `consolidated` state tracking (0 = pending)    |
+---------------------------------------------------------------------------+
                                      |
                                      | (Async Worker Task)
                                      v
+---------------------------------------------------------------------------+
|             Tier 3: Semantic Memory (NetworkX Knowledge Graph)            |
|             - Async Consolidation Daemon extracts (Subject, Pred, Object) |
|             - Ebbinghaus Forgetting Curves: R = e^(-t/S)                  |
|             - Automatic weak node pruning & strength reinforcement       |
+---------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```text
llm-cognitive-memory-engine/
├── app/                        # Core Engine Package
│   ├── algorithms/             # Spreading Activation & Triple Extraction
│   ├── api/                    # FastAPI REST routes (/chat, /memory/state)
│   ├── core/                   # Engine config, orchestrator & background daemon
│   ├── llm/                    # Abstract LLM client & Ollama HTTP wrapper
│   ├── memory/                 # Working, Episodic (SQLite), Semantic (NetworkX) stores & Decay math
│   └── main.py                 # FastAPI Gateway entrypoint
├── benchmarks/                 # Benchmarking Suite
│   ├── retention_accuracy/     # Memory Retention Stress-Test Suite (+50% lead)
│   └── token_efficiency/       # Token Compression Benchmark Suite (~30% reduction)
├── tests/                      # Automated Verification & Unit Test Suite
│   ├── test_step1.py           # Ollama client & JSON schema verification
│   ├── test_step2.py           # Working & Episodic Memory sliding window tests
│   ├── test_step3.py           # Semantic Graph, Ebbinghaus decay & Spreading Activation tests
│   ├── test_step4.py           # End-to-end engine & background daemon tests
│   └── test_step5.py           # FastAPI gateway REST route tests
├── visualizer/
│   └── app.py                  # Streamlit interactive Knowledge Graph dashboard
├── .env.example                # Environment configuration template
├── .gitignore                  # Git ignore rules
├── LICENSE                     # Custom Source-Available License (Non-Research)
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Environment & Setup
Ensure local [Ollama](https://ollama.com) is installed and serving `qwen2.5:7b`:
```bash
ollama pull qwen2.5:7b
```

Create and activate the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Verification Tests
```bash
PYTHONPATH=. python tests/test_step1.py
PYTHONPATH=. python tests/test_step2.py
PYTHONPATH=. python tests/test_step3.py
PYTHONPATH=. python tests/test_step4.py
PYTHONPATH=. python tests/test_step5.py
```

### 3. Launch FastAPI Gateway
```bash
uvicorn app.main:app --reload --port 8000
```
* **Swagger Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Chat Endpoint:** `POST /api/v1/chat`
* **Memory State Inspector:** `GET /api/v1/memory/state`
* **Trigger Consolidation:** `POST /api/v1/memory/consolidate`

### 4. Run Benchmarks
```bash
# Run Memory Retention & Accuracy Stress-Test Benchmark (+50% lead)
PYTHONPATH=. python benchmarks/retention_accuracy/run_retention_benchmarks.py

# Run Token Efficiency Benchmark (~30% savings)
PYTHONPATH=. python benchmarks/token_efficiency/run_token_benchmarks.py
```

### 5. Launch Streamlit Visualizer
```bash
streamlit run visualizer/app.py
```

---

## 📜 License

This project is protected under a **Source-Available Custom License**. All research, commercialization, and derivative development rights are strictly reserved by the author. Use for academic or institutional research without prior written consent is prohibited. See [LICENSE](LICENSE) for full details.
