import os
import asyncio
from fastapi.testclient import TestClient

from app.main import app
from benchmarks.token_efficiency.run_token_benchmarks import run_token_benchmarks
from benchmarks.retention_accuracy.run_retention_benchmarks import run_retention_benchmarks


def test_fastapi_endpoints():
    print("=" * 60)
    print(" 🚀 Step 5 Verification: FastAPI Gateway & Benchmarks")
    print("=" * 60)

    print("[1] Testing FastAPI Gateway routes via TestClient...")
    with TestClient(app) as client:
        # Test Root Endpoint
        res_root = client.get("/")
        print(f"    - GET /: Status {res_root.status_code} | {res_root.json()}")
        assert res_root.status_code == 200

        # Test Memory State Endpoint
        res_state = client.get("/api/v1/memory/state")
        print(f"    - GET /api/v1/memory/state: Status {res_state.status_code} | {res_state.json()}")
        assert res_state.status_code == 200

        # Test Chat Endpoint
        chat_payload = {
            "session_id": "test-session-fastapi",
            "message": "Hello! I am testing the FastAPI gateway endpoint."
        }
        res_chat = client.post("/api/v1/chat", json=chat_payload)
        print(f"    - POST /api/v1/chat: Status {res_chat.status_code}")
        print(f"      Response: \"{res_chat.json()['response']}\"")
        assert res_chat.status_code == 200

        # Test Consolidate Endpoint
        res_cons = client.post("/api/v1/memory/consolidate")
        print(f"    - POST /api/v1/memory/consolidate: Status {res_cons.status_code} | {res_cons.json()}")
        assert res_cons.status_code == 200

    print("✅ SUCCESS: All FastAPI Gateway endpoints verified successfully!")
    print("-" * 60)


async def main():
    test_fastapi_endpoints()

    print("\n[2] Running Token Efficiency Benchmark Suite...")
    token_results = await run_token_benchmarks()
    assert os.path.exists("benchmarks/token_efficiency/token_results.json")

    print("\n[3] Running Memory Retention & Accuracy Stress-Test Benchmark Suite...")
    retention_results = await run_retention_benchmarks()
    assert os.path.exists("benchmarks/retention_accuracy/retention_results.json")

    print("=" * 60)
    print(" 🎉 ALL STEP 5 VERIFICATION TESTS & BENCHMARKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
