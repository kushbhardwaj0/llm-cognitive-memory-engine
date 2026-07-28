# Cognitive-Memory-Engine 🧠

A local-first, human-inspired cognitive memory backend for Large Language Model (LLM) agents, built from scratch using Python, FastAPI, SQLite (`aiosqlite`), NetworkX, and Ollama (`qwen2.5:7b`).

---

## 🏛️ System Architecture

Standard Retrieval-Augmented Generation (RAG) relies on static vector database lookups, which suffer from high token overhead, lack of state evolution, and an inability to synthesize high-level concepts from raw conversational history.

This engine replaces static vector retrieval with a 3-tiered cognitive memory model:

1. **Working Memory (`app/memory/working.py`)**: In-memory, high-speed sliding token window buffer. When capacity is exceeded, oldest turns are evicted.
2. **Episodic Memory (`app/memory/episodic.py`)**: Persistent, sequential timeline event log stored in SQLite asynchronously via `aiosqlite`.
3. **Semantic Memory (`app/memory/semantic.py`)**: NetworkX directed Knowledge Graph (`nx.DiGraph`) representing extracted abstract concepts `(Subject, Predicate, Object)` with **Ebbinghaus Forgetting Curves** ($R = e^{-t/S}$) and **Spreading Activation Retrieval** (`app/algorithms/activation.py`).
4. **Background Consolidation Daemon (`app/core/daemon.py`)**: Asynchronous worker daemon (`asyncio.create_task`) that periodically pulls unconsolidated raw episodes from SQLite, prompts `qwen2.5:7b` to extract structured JSON triples, updates the Knowledge Graph, applies memory decay, and marks episodes as consolidated.

---

## 📁 Repository Structure

```text
llm-cognitive-memory-engine/
├── app/                        # Core Engine Package
│   ├── algorithms/             # Spreading Activation & Concept Extraction
│   ├── api/                    # FastAPI REST routes (/chat, /memory/state)
│   ├── core/                   # Engine config, orchestrator & background daemon
│   ├── llm/                    # Abstract LLM client & Ollama HTTP wrapper
│   ├── memory/                 # Working, Episodic (SQLite), Semantic (NetworkX) stores & Decay math
│   └── main.py                 # FastAPI Gateway entrypoint
├── benchmarks/                 # Benchmarking Suite
│   ├── retention_accuracy/     # Memory Retention Stress-Test (+50% lead over RAG)
│   └── token_efficiency/       # Token Compression Benchmark (~30% reduction)
├── tests/                      # Automated Verification & Unit Test Suite
│   ├── test_step1.py           # Ollama client & JSON schema verification
│   ├── test_step2.py           # Working & Episodic Memory sliding window tests
│   ├── test_step3.py           # Semantic Graph, Ebbinghaus decay & Spreading Activation tests
│   ├── test_step4.py           # End-to-end engine & background daemon tests
│   └── test_step5.py           # FastAPI gateway REST route tests
├── visualizer/
│   └── app.py                  # Streamlit interactive Knowledge Graph dashboard
├── .env.example                # Environment configuration template
├── .gitignore                  # Git ignore rules for venv, databases, and caches
├── LICENSE                     # Custom Source-Available License (Non-Research)
├── README.md                   # Repository documentation
└── requirements.txt            # Dependencies
```

---

## 🚀 Quick Start Guide

### 1. Environment & Setup
Make sure local [Ollama](https://ollama.com) is installed and serving `qwen2.5:7b`:
```bash
ollama pull qwen2.5:7b
```

Activate the virtual environment:
```bash
source venv/bin/activate
```

### 2. Run Verification Tests
```bash
# Run step-by-step verification suites
PYTHONPATH=. python tests/test_step1.py
PYTHONPATH=. python tests/test_step2.py
PYTHONPATH=. python tests/test_step3.py
PYTHONPATH=. python tests/test_step4.py
PYTHONPATH=. python tests/test_step5.py
```

### 3. Launch FastAPI Server
```bash
uvicorn app.main:app --reload --port 8000
```
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Chat Endpoint: `POST /api/v1/chat`
- Memory State Endpoint: `GET /api/v1/memory/state`
- Consolidate Endpoint: `POST /api/v1/memory/consolidate`

### 4. Run Benchmarking Suites
```bash
# Run Token Efficiency Benchmark
PYTHONPATH=. python benchmarks/token_efficiency/run_token_benchmarks.py

# Run Memory Retention & Accuracy Stress-Test Benchmark
PYTHONPATH=. python benchmarks/retention_accuracy/run_retention_benchmarks.py
```

### 5. Launch Streamlit Visualizer
```bash
streamlit run visualizer/app.py
```

---

## 📊 Benchmark Results Summary

### 1. Memory Retention & Accuracy Stress-Test (`benchmarks/retention_accuracy/retention_results.json`)
* **Cognitive Memory Engine Accuracy**: **83.3%**
* **Naive Vector RAG Accuracy**: **33.3%**
* **Accuracy Improvement**: **+50.0% Lead** over Naive RAG on long-horizon memory tasks.
  * **State Evolution (PostgreSQL -> ScyllaDB Migration)**: Cognitive Engine correctly retrieved updated active state (**100% vs 0% for RAG**).
  * **Noise Haystack (30-Turn Noise Dilution)**: Cognitive Engine extracted target needles from heavy noise (**50% vs 0% for RAG**).

### 2. Token Efficiency (`benchmarks/token_efficiency/token_results.json`)
* **Token Reduction**: **11.7% - 30.1% prompt token savings** compared to raw chunk injection.

---

## 📜 License

This project is licensed under a **Source-Available Custom License**. All research, commercialization, and derivative development rights are strictly reserved by the author. Usage for academic or institutional research without written permission is prohibited. See [LICENSE](LICENSE) for details.
