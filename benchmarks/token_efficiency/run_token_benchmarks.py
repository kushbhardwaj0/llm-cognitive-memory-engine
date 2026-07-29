import os
import json
import time
import asyncio
import math
from typing import Dict, Any

from app.core.engine import CognitiveMemoryEngine
from app.memory.working import Message
from app.memory.episodic import Episode
from benchmarks.token_efficiency.baseline_rag import BaselineVectorRAG

BENCHMARK_DB_PATH = "token_bench_episodic.db"
BENCHMARK_GRAPH_PATH = "token_bench_semantic.json"
BENCHMARK_RESULTS_PATH = "benchmarks/token_efficiency/token_results.json"


async def ingest_history_turn(engine: CognitiveMemoryEngine, session_id: str, text: str) -> None:
    """Fast ingestion helper to add historical turns directly into Working/Episodic memory."""
    msg = Message(session_id=session_id, role="user", content=text)
    evicted = engine.working_memory.add_message(msg)
    if evicted:
        episodes = [
            Episode(
                id=m.id,
                session_id=m.session_id,
                role=m.role,
                content=m.content,
                timestamp=m.timestamp,
                consolidated=False,
            )
            for m in evicted
        ]
        await engine.episodic_memory.add_episodes(episodes)


DATASET_TURNS = [
    "Hello! My name is Alex. I am a backend systems engineer working on local AI architectures.",
    "My favorite language for building core backend algorithms is Python 3.11.",
    "I prefer using Qwen2.5 running locally on Apple Silicon Metal for fast inference.",
    "Let's discuss database design for event logs. We are choosing SQLite with aiosqlite.",
    "For graph algorithms, NetworkX is great for in-memory directed graphs and Ebbinghaus decay math.",
]

EVALUATION_QUESTIONS = [
    {
        "query": "What language does Alex prefer for backend engineering based on his history?",
        "expected_keywords": ["Python", "3.11"],
    },
    {
        "query": "What local LLM model and hardware platform does Alex use for inference?",
        "expected_keywords": ["Qwen", "Apple", "Metal"],
    },
    {
        "query": "What database and python library are used for storing event logs?",
        "expected_keywords": ["SQLite", "aiosqlite"],
    },
]


async def run_token_benchmarks() -> Dict[str, Any]:
    print("=" * 70)
    print(" Running Token Efficiency Benchmark Suite")
    print("=" * 70)

    for p in [BENCHMARK_DB_PATH, BENCHMARK_GRAPH_PATH]:
        if os.path.exists(p):
            os.remove(p)

    session_id = "token-bench-session"

    engine = CognitiveMemoryEngine(
        db_path=BENCHMARK_DB_PATH,
        graph_path=BENCHMARK_GRAPH_PATH,
        working_max_tokens=60,
    )
    await engine.initialize()
    baseline_rag = BaselineVectorRAG(top_k=2)

    print("[1] Ingesting dataset turns...")
    for turn in DATASET_TURNS:
        baseline_rag.add_document(turn)
        await ingest_history_turn(engine, session_id, turn)

    print("[2] Running consolidation daemon pass...")
    await engine.daemon.consolidate_once()

    print("[3] Evaluating Token Efficiency...")
    engine_tokens: list[int] = []
    baseline_tokens: list[int] = []

    for idx, q_item in enumerate(EVALUATION_QUESTIONS, start=1):
        q = q_item["query"]
        _, rag_toks = await baseline_rag.query(q)
        baseline_tokens.append(rag_toks)

        retrieved_triples = engine.spreading_activation.get_activated_triples(query=q)
        graph_ctx = "\n".join([f"- {s} {p} {o}" for s, p, o in retrieved_triples])
        prompt_words = (graph_ctx + "\n" + q).split()
        eng_toks = math.ceil(len(prompt_words) * 1.3) + 20
        await engine.chat(session_id=session_id, user_message=q)
        engine_tokens.append(eng_toks)

        print(f"  Question {idx}: Baseline RAG = {rag_toks} tokens | Cognitive Engine = {eng_toks} tokens")

    avg_eng_toks = sum(engine_tokens) / len(engine_tokens)
    avg_base_toks = sum(baseline_tokens) / len(baseline_tokens)
    reduction_pct = ((avg_base_toks - avg_eng_toks) / avg_base_toks) * 100.0

    results = {
        "cognitive_memory_engine_avg_prompt_tokens": round(avg_eng_toks, 1),
        "baseline_vector_rag_avg_prompt_tokens": round(avg_base_toks, 1),
        "token_reduction_percentage": round(reduction_pct, 1),
    }

    print("\n" + "=" * 70)
    print(f" Token Reduction Achieved: {results['token_reduction_percentage']}%")
    print("=" * 70)

    os.makedirs(os.path.dirname(BENCHMARK_RESULTS_PATH), exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for p in [BENCHMARK_DB_PATH, BENCHMARK_GRAPH_PATH]:
        if os.path.exists(p):
            os.remove(p)

    return results


if __name__ == "__main__":
    asyncio.run(run_token_benchmarks())
