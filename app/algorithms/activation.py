import time
import logging
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx

from app.memory.semantic import SemanticMemory
from app.memory.decay import calculate_retention

logger = logging.getLogger(__name__)


class SpreadingActivation:
    """
    Spreading Activation algorithm for graph-based contextual memory retrieval.
    Propagates activation energy from seed nodes across graph edges over N hops.
    """

    def __init__(
        self,
        semantic_memory: SemanticMemory,
        decay_factor: float = 0.7,
        max_hops: int = 2,
        activation_threshold: float = 0.2,
    ):
        self.semantic_memory = semantic_memory
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.activation_threshold = activation_threshold

    def find_seed_nodes(self, query: str) -> List[str]:
        """Find graph nodes matching words or substrings in the query string."""
        if not query or not self.semantic_memory.graph.nodes:
            return []

        query_lower = query.lower()
        seed_nodes: List[str] = []

        import re
        for node in self.semantic_memory.graph.nodes:
            node_lower = node.lower()
            if len(node_lower) <= 2:
                if re.search(r'\b' + re.escape(node_lower) + r'\b', query_lower):
                    seed_nodes.append(node)
            else:
                if node_lower in query_lower:
                    seed_nodes.append(node)

        return seed_nodes

    def activate(
        self, seed_nodes: List[str], current_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Fire activation energy from seed nodes and propagate through graph edges over N hops.
        
        Returns:
            Dictionary mapping activated node names to their activation energy scores.
        """
        graph = self.semantic_memory.graph
        now = current_time or time.time()

        if not graph.nodes or not seed_nodes:
            return {}

        # Energy dictionary mapping node -> energy
        activations: Dict[str, float] = {}

        # Step 1: Fire initial energy into valid seed nodes
        valid_seeds = [s for s in seed_nodes if graph.has_node(s)]
        for seed in valid_seeds:
            activations[seed] = 1.0

        if not activations:
            return {}

        # Step 2: Propagate energy over N hops
        for hop in range(self.max_hops):
            current_level = dict(activations)
            for node, energy in current_level.items():
                if energy <= 0.05:
                    continue

                # Propagate to both outgoing and incoming neighbors
                neighbors = set(graph.successors(node)).union(set(graph.predecessors(node)))
                for neighbor in neighbors:
                    # Calculate retention score of target neighbor
                    data = graph.nodes[neighbor]
                    last_accessed = data.get("last_accessed", now)
                    strength = data.get("strength", 1.0)
                    retention = calculate_retention(last_accessed, now, strength)

                    # Edge weight modifier
                    if graph.has_edge(node, neighbor):
                        edge_data = graph.get_edge_data(node, neighbor)
                    else:
                        edge_data = graph.get_edge_data(neighbor, node)

                    weight = edge_data.get("weight", 1.0) if edge_data else 1.0

                    # Energy formula: E_next = E_curr * decay * weight * retention
                    transferred = energy * self.decay_factor * weight * retention
                    activations[neighbor] = max(activations.get(neighbor, 0.0), transferred)

        # Filter out nodes below activation threshold
        filtered_activations = {
            node: round(score, 4)
            for node, score in activations.items()
            if score >= self.activation_threshold
        }

        # Reinforce strength of retrieved activated nodes (touch node)
        for node in filtered_activations:
            self.semantic_memory.touch_node(node, current_time=now)

        return filtered_activations

    def get_activated_triples(
        self, query: str, seed_nodes: Optional[List[str]] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Extract (subject, predicate, object) subgraphs for nodes activated above threshold.
        """
        if seed_nodes is None:
            seed_nodes = self.find_seed_nodes(query)

        activations = self.activate(seed_nodes)
        if not activations:
            return []

        activated_nodes: Set[str] = set(activations.keys())
        triples: List[Tuple[str, str, str]] = []

        graph = self.semantic_memory.graph
        for u, v, data in graph.edges(data=True):
            if u in activated_nodes or v in activated_nodes:
                relation = data.get("relation", "RELATED_TO")
                triples.append((u, relation, v))

        return triples
