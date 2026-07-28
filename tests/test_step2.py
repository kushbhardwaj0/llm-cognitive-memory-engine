import os
import asyncio
from app.memory.working import WorkingMemory, Message
from app.memory.episodic import EpisodicMemory, Episode

TEST_DB_PATH = "test_episodic.db"


async def main():
    print("=" * 60)
    print(" 🚀 Step 2 Verification: Working & Episodic Memory Stores")
    print("=" * 60)

    # Cleanup test DB if left from previous run
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    session_id = "test-session-101"

    # -------------------------------------------------------------
    # Part 1: Working Memory Sliding Window Test
    # -------------------------------------------------------------
    print("[1] Testing Working Memory (Small max_tokens = 50 limit)...")
    working_mem = WorkingMemory(max_tokens=50)

    msg1 = Message(session_id=session_id, role="user", content="Hello! I am Kush and I am building an AI memory engine.")
    msg2 = Message(session_id=session_id, role="assistant", content="Awesome project! Tell me more about the architecture.")
    msg3 = Message(session_id=session_id, role="user", content="It has working memory, episodic memory in SQLite, and a NetworkX semantic graph.")

    evicted1 = working_mem.add_message(msg1)
    evicted2 = working_mem.add_message(msg2)
    evicted3 = working_mem.add_message(msg3)

    total_evicted = evicted1 + evicted2 + evicted3

    print(f"    - Buffer Active Messages Count: {len(working_mem.get_messages())}")
    print(f"    - Buffer Active Token Count:    {working_mem.total_tokens}")
    print(f"    - Evicted Messages Count:       {len(total_evicted)}")

    assert len(total_evicted) > 0, "Expected at least 1 message to be evicted due to token capacity limit!"
    print("✅ SUCCESS: Working Memory sliding window correctly evicted oldest messages.")
    print("-" * 60)

    # -------------------------------------------------------------
    # Part 2: Episodic Memory Persistent Storage Test
    # -------------------------------------------------------------
    print("[2] Initializing Episodic Memory SQLite Store...")
    episodic_mem = EpisodicMemory(db_path=TEST_DB_PATH)
    await episodic_mem.initialize()

    print("[3] Persisting evicted messages to Episodic Memory...")
    episodes_to_add = [
        Episode(
            id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            consolidated=False,
        )
        for msg in total_evicted
    ]
    await episodic_mem.add_episodes(episodes_to_add)

    count = await episodic_mem.count_episodes(session_id=session_id)
    print(f"    - Total Episodes in DB: {count}")
    assert count == len(total_evicted), f"Expected {len(total_evicted)} episodes in DB, found {count}"
    print("✅ SUCCESS: Episodes saved to SQLite store.")
    print("-" * 60)

    # -------------------------------------------------------------
    # Part 3: Querying Unconsolidated Episodes & Marking Consolidated
    # -------------------------------------------------------------
    print("[4] Querying unconsolidated episodes (consolidated = 0)...")
    unconsolidated = await episodic_mem.get_unconsolidated_episodes(session_id=session_id)
    print(f"    - Unconsolidated Episodes Count: {len(unconsolidated)}")
    assert len(unconsolidated) == count, "All newly added episodes should be unconsolidated."

    print("[5] Marking episodes as consolidated (consolidated = 1)...")
    ids_to_mark = [ep.id for ep in unconsolidated]
    await episodic_mem.mark_consolidated(ids_to_mark)

    remaining_unconsolidated = await episodic_mem.get_unconsolidated_episodes(session_id=session_id)
    print(f"    - Unconsolidated Episodes Remaining: {len(remaining_unconsolidated)}")
    assert len(remaining_unconsolidated) == 0, "Expected 0 unconsolidated episodes after consolidation update."
    print("✅ SUCCESS: Consolidation flag status update verified.")
    print("-" * 60)

    # Cleanup DB artifact
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    print("=" * 60)
    print(" 🎉 ALL STEP 2 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
