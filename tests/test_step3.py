import os
import time
import asyncio
from app.memory.semantic import SemanticMemory
from app.memory.decay import calculate_retention, prune_decayed_nodes
from app.algorithms.extraction import ConceptExtractor, ExtractedTriple
from app.algorithms.activation import SpreadingActivation

TEST_GRAPH_PATH = "test_semantic_graph.json"


async def main():
    print("=" * 60)
    print(" Step 3 Verification: Semantic Memory, Decay & Spreading Activation")
    print("=" * 60)

    if os.path.exists(TEST_GRAPH_PATH):
        os.remove(TEST_GRAPH_PATH)

    semantic_mem = SemanticMemory(graph_path=TEST_GRAPH_PATH)

    # Part 1: LLM Knowledge Extraction Test
    print("[1] Testing LLM Concept Triple Extraction (qwen2.5:7b)...")
    extractor = ConceptExtractor()
    sample_text = (
        "Alex is a Lead Engineer who builds Cognitive Memory Engine. "
        "Cognitive Memory Engine uses NetworkX for Knowledge Graph representation."
    )
    
    triples: list[ExtractedTriple] = await extractor.extract_triples(sample_text)
    print(f"    - Extracted {len(triples)} triples from raw text:")
    for t in triples:
        print(f"      * ({t.subject}) --[{t.predicate}]--> ({t.object})")

    assert len(triples) > 0, "Expected at least 1 extracted triple from LLM"
    print("[SUCCESS] LLM extracted JSON knowledge triples.")
    print("-" * 60)

    # Part 2: Building Semantic Knowledge Graph
    print("[2] Populate Semantic Memory Graph...")
    for t in triples:
        semantic_mem.add_triple(subject=t.subject, predicate=t.predicate, obj=t.object)

    semantic_mem.add_triple("Alex", "PREFERS", "Qwen2.5")
    semantic_mem.add_triple("Qwen2.5", "RUNS_ON", "Apple Metal")
    semantic_mem.add_triple("Old Project", "USED", "Obsolete Framework")

    print(f"    - Total Nodes in Graph: {len(semantic_mem.graph.nodes)}")
    print(f"    - Total Edges in Graph: {len(semantic_mem.graph.edges)}")

    semantic_mem.save()
    assert os.path.exists(TEST_GRAPH_PATH), "Expected test graph JSON file to be written to disk"
    print("[SUCCESS] Semantic Memory graph saved to JSON disk storage.")
    print("-" * 60)

    # Part 3: Ebbinghaus Forgetting Curve Math & Decay Test
    print("[3] Testing Ebbinghaus Retention Math & Pruning...")
    now = time.time()
    
    old_time = now - (100 * 24 * 3600)
    semantic_mem.graph.nodes["Old Project"]["last_accessed"] = old_time
    semantic_mem.graph.nodes["Old Project"]["strength"] = 0.5

    retention_fresh = calculate_retention(last_accessed=now, current_time=now, strength=1.0)
    retention_old = calculate_retention(last_accessed=old_time, current_time=now, strength=0.5)

    print(f"    - Retention score for fresh node (t=0):         {retention_fresh}")
    print(f"    - Retention score for 100-day old node (t=100d): {retention_old}")
    assert retention_fresh == 1.0, f"Expected retention 1.0, got {retention_fresh}"
    assert retention_old < 0.15, f"Expected retention < 0.15, got {retention_old}"

    pruned_nodes = semantic_mem.apply_decay(threshold=0.15, current_time=now)
    print(f"    - Pruned nodes below 0.15 threshold: {pruned_nodes}")
    assert "Old Project" in pruned_nodes, "Expected 'Old Project' node to be pruned due to decay."
    print("[SUCCESS] Decay function correctly identified and pruned weak memory node.")
    print("-" * 60)

    # Part 4: Spreading Activation Retrieval Test
    print("[4] Testing Spreading Activation Graph Traversal...")
    sp_activation = SpreadingActivation(semantic_memory=semantic_mem, decay_factor=0.7, max_hops=2)

    query = "Tell me about Alex and what model he prefers."
    seed_nodes = sp_activation.find_seed_nodes(query)
    print(f"    - Query: '{query}'")
    print(f"    - Seed Nodes Identified: {seed_nodes}")

    activated_scores = sp_activation.activate(seed_nodes)
    print("    - Node Activation Energies:")
    for node, energy in activated_scores.items():
        print(f"      * {node}: Energy = {energy}")

    subgraph_triples = sp_activation.get_activated_triples(query=query)
    print(f"    - Retrieved Context Subgraph Triples ({len(subgraph_triples)}):")
    for sub, pred, obj in subgraph_triples:
        print(f"      - {sub} {pred} {obj}")

    assert "Alex" in activated_scores, "Seed node Alex should be activated"
    assert len(subgraph_triples) > 0, "Sub-graph triples should be retrieved"
    print("[SUCCESS] Spreading activation successfully retrieved relevant subgraph context!")
    print("-" * 60)

    if os.path.exists(TEST_GRAPH_PATH):
        os.remove(TEST_GRAPH_PATH)

    print("=" * 60)
    print(" ALL STEP 3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
