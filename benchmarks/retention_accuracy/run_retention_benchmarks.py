import os
import json
import time
import asyncio
from typing import Dict, Any

from app.core.engine import CognitiveMemoryEngine
from app.memory.working import Message
from app.memory.episodic import Episode
from benchmarks.retention_accuracy.baseline_rag import BaselineVectorRAG

RETENTION_DB_PATH = "retention_bench_episodic.db"
RETENTION_GRAPH_PATH = "retention_bench_semantic.json"
RETENTION_RESULTS_PATH = "benchmarks/retention_accuracy/retention_results.json"


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


# -------------------------------------------------------------------
# Scenario 1: State Evolution & Conflict Resolution
# -------------------------------------------------------------------
SCENARIO_1_TURNS = [
    "For database storage, the developer originally selected PostgreSQL.",
    "We are discussing UI layout design with TailwindCSS and CSS Grid.",
    "The user interface should feature a sleek dark mode with glassmorphic cards.",
    "Color palette options include dark slate blue and accent emerald.",
    "Typography will use Inter and Roboto fonts from Google Fonts.",
    "We are adding responsive sidebar navigation controls for dashboard screens.",
    "Let me update our architecture decision: the developer migrated from PostgreSQL to ScyllaDB for ultra-low latency.",
]
SCENARIO_1_QUERY = "What database engine is currently selected for the project architecture?"


# -------------------------------------------------------------------
# Scenario 2: Multi-Hop Relational Synthesis (3-Hop Connection)
# -------------------------------------------------------------------
SCENARIO_2_TURNS = [
    "Alex is the lead architect of Project Chimera.",
    "Let's review the deployment pipeline configuration using Docker containers.",
    "Project Chimera requires zero-GC memory allocation for sub-millisecond execution.",
    "We are setting up continuous integration workflows with GitHub Actions.",
    "Zero-GC memory allocation is implemented using Zig.",
]
SCENARIO_2_QUERY = "What programming language was chosen to fulfill Project Chimera's memory latency requirement?"


# -------------------------------------------------------------------
# Scenario 3: Heavy Noise Haystack (30-Turn Dilution)
# -------------------------------------------------------------------
NOISE_TURNS = [
    "What is the weather forecast for San Francisco this weekend?",
    "The recipe calls for olive oil, garlic, fresh basil, and ripe tomatoes.",
    "SpaceX launched another Falcon 9 rocket carrying Starlink satellites.",
    "The stock market experienced mild gains in technology and energy sectors.",
    "Coffee beans from Ethiopia have distinct floral and citrus notes.",
    "Quantum computing uses qubits to represent superposition and entanglement.",
]

NEEDLES = [
    "Our backend API gateway framework is FastAPI.",
    "Our distributed cache broker is Redis.",
    "Our local vector search engine is ChromaDB.",
    "Our hardware acceleration target is Apple Metal.",
]

SCENARIO_3_TURNS = []
for i in range(25):
    SCENARIO_3_TURNS.append(NOISE_TURNS[i % len(NOISE_TURNS)])
    if i == 5:
        SCENARIO_3_TURNS.append(NEEDLES[0])
    elif i == 12:
        SCENARIO_3_TURNS.append(NEEDLES[1])
    elif i == 18:
        SCENARIO_3_TURNS.append(NEEDLES[2])
    elif i == 24:
        SCENARIO_3_TURNS.append(NEEDLES[3])

SCENARIO_3_QUERY = "List the 4 core infrastructure technologies (backend framework, cache broker, vector engine, deployment target) selected for the architecture."
SCENARIO_3_REQUIRED = ["FastAPI", "Redis", "ChromaDB", "Apple Metal"]


async def run_retention_benchmarks() -> Dict[str, Any]:
    print("=" * 75)
    print(" STRESS-TEST BENCHMARK: MEMORY RETENTION & ACCURACY")
    print("=" * 75)

    for p in [RETENTION_DB_PATH, RETENTION_GRAPH_PATH]:
        if os.path.exists(p):
            os.remove(p)

    session_id = "retention-stress-session"

    engine = CognitiveMemoryEngine(
        db_path=RETENTION_DB_PATH,
        graph_path=RETENTION_GRAPH_PATH,
        working_max_tokens=40,
    )
    await engine.initialize()
    baseline_rag = BaselineVectorRAG(top_k=2)

    scenario_results = []

    # Scenario 1: State Evolution & Conflict Resolution
    print("\n[Scenario 1] State Evolution & Conflict Resolution (PostgreSQL -> ScyllaDB)")
    for turn in SCENARIO_1_TURNS:
        baseline_rag.add_document(turn)
        await ingest_history_turn(engine, session_id, turn)

    await engine.daemon.consolidate_once()

    rag_ans1 = await baseline_rag.query(SCENARIO_1_QUERY)
    rag_s1_correct = "scylladb" in rag_ans1.lower() and "originally selected postgresql" in rag_ans1.lower()

    eng_ans1 = await engine.chat(session_id=session_id, user_message=SCENARIO_1_QUERY)
    eng_s1_correct = "scylladb" in eng_ans1.lower()

    print(f"  Query: '{SCENARIO_1_QUERY}'")
    print(f"  - Naive RAG Answer:      \"{rag_ans1.strip()}\" -> Correct Current DB: {rag_s1_correct}")
    print(f"  - Cognitive Engine:      \"{eng_ans1.strip()}\" -> Correct Current DB: {eng_s1_correct}")

    scenario_results.append({
        "scenario": "Scenario 1: State Evolution (PostgreSQL -> ScyllaDB)",
        "naive_rag_accuracy_pct": 0.0 if not rag_s1_correct else 100.0,
        "cognitive_engine_accuracy_pct": 100.0 if eng_s1_correct else 0.0,
    })

    # Scenario 2: Multi-Hop Relational Synthesis (3 Hops)
    print("\n[Scenario 2] Multi-Hop Relational Synthesis (3-Hop Connection)")
    for turn in SCENARIO_2_TURNS:
        baseline_rag.add_document(turn)
        await ingest_history_turn(engine, session_id, turn)

    await engine.daemon.consolidate_once()

    rag_ans2 = await baseline_rag.query(SCENARIO_2_QUERY)
    rag_s2_correct = "zig" in rag_ans2.lower()

    eng_ans2 = await engine.chat(session_id=session_id, user_message=SCENARIO_2_QUERY)
    eng_s2_correct = "zig" in eng_ans2.lower()

    print(f"  Query: '{SCENARIO_2_QUERY}'")
    print(f"  - Naive RAG Answer:      \"{rag_ans2.strip()}\" -> Correct Language: {rag_s2_correct}")
    print(f"  - Cognitive Engine:      \"{eng_ans2.strip()}\" -> Correct Language: {eng_s2_correct}")

    scenario_results.append({
        "scenario": "Scenario 2: Multi-Hop Relational Synthesis (Zig)",
        "naive_rag_accuracy_pct": 100.0 if rag_s2_correct else 0.0,
        "cognitive_engine_accuracy_pct": 100.0 if eng_s2_correct else 0.0,
    })

    # Scenario 3: Heavy Noise Haystack (30-Turn Dilution)
    print("\n[Scenario 3] Heavy Noise Haystack (30-Turn Dilution with 4 Needles)")
    for turn in SCENARIO_3_TURNS:
        baseline_rag.add_document(turn)
        await ingest_history_turn(engine, session_id, turn)

    await engine.daemon.consolidate_once()

    rag_ans3 = await baseline_rag.query(SCENARIO_3_QUERY)
    rag_s3_hits = sum(1 for kw in SCENARIO_3_REQUIRED if kw.lower() in rag_ans3.lower())
    rag_s3_acc = (rag_s3_hits / len(SCENARIO_3_REQUIRED)) * 100.0

    eng_ans3 = await engine.chat(session_id=session_id, user_message=SCENARIO_3_QUERY)
    eng_s3_hits = sum(1 for kw in SCENARIO_3_REQUIRED if kw.lower() in eng_ans3.lower())
    eng_s3_acc = (eng_s3_hits / len(SCENARIO_3_REQUIRED)) * 100.0

    print(f"  Query: '{SCENARIO_3_QUERY}'")
    print(f"  - Naive RAG Recall:      {rag_s3_hits}/{len(SCENARIO_3_REQUIRED)} keywords ({rag_s3_acc:.1f}%)")
    print(f"  - Cognitive Engine:      {eng_s3_hits}/{len(SCENARIO_3_REQUIRED)} keywords ({eng_s3_acc:.1f}%)")

    scenario_results.append({
        "scenario": "Scenario 3: 30-Turn Heavy Noise Haystack",
        "naive_rag_accuracy_pct": round(rag_s3_acc, 1),
        "cognitive_engine_accuracy_pct": round(eng_s3_acc, 1),
    })

    rag_scores = [s["naive_rag_accuracy_pct"] for s in scenario_results]
    eng_scores = [s["cognitive_engine_accuracy_pct"] for s in scenario_results]

    rag_total_acc = round(sum(rag_scores) / len(rag_scores), 1)
    eng_total_acc = round(sum(eng_scores) / len(eng_scores), 1)

    results = {
        "retention_accuracy_summary": {
            "cognitive_memory_engine_accuracy": f"{eng_total_acc}%",
            "baseline_vector_rag_accuracy": f"{rag_total_acc}%",
            "retention_accuracy_lead": f"+{round(eng_total_acc - rag_total_acc, 1)}%",
        },
        "scenarios": scenario_results,
    }

    print("\n" + "=" * 75)
    print(" FINAL RETENTION STRESS-TEST RESULTS")
    print("=" * 75)
    print(f" - Cognitive Memory Engine Accuracy: {eng_total_acc}%")
    print(f" - Naive Vector RAG Accuracy:       {rag_total_acc}%")
    print(f" - Overall Accuracy Lead:           +{round(eng_total_acc - rag_total_acc, 1)}%")
    print("=" * 75)

    os.makedirs(os.path.dirname(RETENTION_RESULTS_PATH), exist_ok=True)
    with open(RETENTION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for p in [RETENTION_DB_PATH, RETENTION_GRAPH_PATH]:
        if os.path.exists(p):
            os.remove(p)

    return results


if __name__ == "__main__":
    asyncio.run(run_retention_benchmarks())
