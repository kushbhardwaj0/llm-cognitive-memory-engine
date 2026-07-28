import os
import asyncio
from app.core.engine import CognitiveMemoryEngine

TEST_DB_PATH = "test_engine_episodic.db"
TEST_GRAPH_PATH = "test_engine_graph.json"


async def main():
    print("=" * 60)
    print(" 🚀 Step 4 Verification: Unified Cognitive Engine & Daemon Pipeline")
    print("=" * 60)

    # Cleanup leftover artifacts
    for path in [TEST_DB_PATH, TEST_GRAPH_PATH]:
        if os.path.exists(path):
            os.remove(path)

    session_id = "session-kush-999"

    # Initialize Engine with small working memory capacity (60 tokens) to trigger eviction quickly
    engine = CognitiveMemoryEngine(
        db_path=TEST_DB_PATH,
        graph_path=TEST_GRAPH_PATH,
        working_max_tokens=60,
    )
    await engine.initialize()

    # -------------------------------------------------------------
    # Step 1: User introduces facts in Turn 1
    # -------------------------------------------------------------
    print("[1] Turn 1: User introduces user facts...")
    turn1_msg = "Hello! My name is Kush. I am building Cognitive-Memory-Engine in Python."
    res1 = await engine.chat(session_id=session_id, user_message=turn1_msg)
    print(f"    - User:      \"{turn1_msg}\"")
    print(f"    - Assistant: \"{res1}\"")
    print("-" * 60)

    # -------------------------------------------------------------
    # Step 2: Push more turns to force Working Memory eviction into SQLite
    # -------------------------------------------------------------
    print("[2] Turns 2 & 3: Overflowing Working Memory to force eviction into SQLite...")
    turn2_msg = "Can you explain how async IO works in Python high throughput web servers?"
    res2 = await engine.chat(session_id=session_id, user_message=turn2_msg)

    turn3_msg = "What are the key benefits of using NetworkX for in-memory graph representation?"
    res3 = await engine.chat(session_id=session_id, user_message=turn3_msg)

    episodes_count = await engine.episodic_memory.count_episodes(session_id=session_id)
    print(f"    - Total Evicted Episodes in SQLite: {episodes_count}")
    assert episodes_count > 0, "Expected working memory overflow turns to be persisted to Episodic SQLite store."
    print("✅ SUCCESS: Overflow messages successfully evicted to Episodic Memory.")
    print("-" * 60)

    # -------------------------------------------------------------
    # Step 3: Trigger Background Consolidation Daemon
    # -------------------------------------------------------------
    print("[3] Running Background Consolidation Daemon pass...")
    consolidated_count = await engine.daemon.consolidate_once()
    print(f"    - Consolidated Episodes Count: {consolidated_count}")
    
    triples = engine.semantic_memory.get_triples()
    print(f"    - Semantic Knowledge Graph Triples ({len(triples)}):")
    for sub, pred, obj in triples:
        print(f"      * ({sub}) --[{pred}]--> ({obj})")

    assert len(triples) > 0, "Expected background daemon to populate Semantic Knowledge Graph with triples."
    print("✅ SUCCESS: Background daemon extracted facts into Semantic Knowledge Graph.")
    print("-" * 60)

    # -------------------------------------------------------------
    # Step 4: Test Long-Term Memory Recall via Spreading Activation
    # -------------------------------------------------------------
    print("[4] Turn 4: Test Long-Term Recall (Asking query after Working Memory eviction)...")
    recall_query = "What project am I building and what language am I using for it?"
    res4 = await engine.chat(session_id=session_id, user_message=recall_query)
    
    print(f"    - Query:     \"{recall_query}\"")
    print(f"    - Assistant: \"{res4}\"")

    assert "Cognitive" in res4 or "Memory" in res4 or "Python" in res4, \
        "Expected Assistant to correctly recall long-term memory facts retrieved via Spreading Activation."

    print("=" * 60)
    print(" 🎉 ALL STEP 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

    # Clean up test artifacts
    for path in [TEST_DB_PATH, TEST_GRAPH_PATH]:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    asyncio.run(main())
