import os
import json
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx

from app.core.config import settings
from app.memory.decay import calculate_retention, reinforce_strength, prune_decayed_nodes

logger = logging.getLogger(__name__)


class SemanticMemory:
    """NetworkX-backed Knowledge Graph representing abstract facts and concepts."""

    def __init__(self, graph_path: Optional[str] = None):
        self.graph_path = graph_path or settings.SEMANTIC_GRAPH_PATH
        self.graph = nx.DiGraph()

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        weight: float = 1.0,
        current_time: Optional[float] = None,
    ) -> None:
        """Add or update a (subject, predicate, object) fact in the knowledge graph."""
        now = current_time or time.time()
        sub_id = subject.strip()
        obj_id = obj.strip()
        pred_id = predicate.strip().upper()

        # Update or initialize Subject Node
        if self.graph.has_node(sub_id):
            node_data = self.graph.nodes[sub_id]
            node_data["access_count"] += 1
            node_data["last_accessed"] = now
            node_data["strength"] = reinforce_strength(node_data["strength"])
        else:
            self.graph.add_node(
                sub_id,
                label=sub_id,
                access_count=1,
                last_accessed=now,
                strength=1.0,
            )

        # Update or initialize Object Node
        if self.graph.has_node(obj_id):
            node_data = self.graph.nodes[obj_id]
            node_data["access_count"] += 1
            node_data["last_accessed"] = now
            node_data["strength"] = reinforce_strength(node_data["strength"])
        else:
            self.graph.add_node(
                obj_id,
                label=obj_id,
                access_count=1,
                last_accessed=now,
                strength=1.0,
            )

        # Add or update Directed Edge
        self.graph.add_edge(sub_id, obj_id, relation=pred_id, weight=weight)

    def touch_node(self, node_id: str, current_time: Optional[float] = None) -> None:
        """Reinforce node strength and update last_accessed timestamp upon retrieval."""
        if self.graph.has_node(node_id):
            now = current_time or time.time()
            data = self.graph.nodes[node_id]
            data["access_count"] += 1
            data["last_accessed"] = now
            data["strength"] = reinforce_strength(data["strength"])

    def get_triples(self) -> List[Tuple[str, str, str]]:
        """Return all (subject, predicate, object) triples in the graph."""
        triples = []
        for u, v, data in self.graph.edges(data=True):
            relation = data.get("relation", "RELATED_TO")
            triples.append((u, relation, v))
        return triples

    def apply_decay(
        self, threshold: float = 0.15, current_time: Optional[float] = None, time_scale_hours: float = 24.0
    ) -> List[str]:
        """Prune nodes whose Ebbinghaus retention drops below threshold."""
        return prune_decayed_nodes(
            self.graph,
            threshold=threshold,
            current_time=current_time,
            time_scale_hours=time_scale_hours,
        )

    def save(self, filepath: Optional[str] = None) -> None:
        """Serialize graph to JSON file using NetworkX node-link format."""
        target_path = filepath or self.graph_path
        data = nx.node_link_data(self.graph)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved Semantic Graph with {len(self.graph.nodes)} nodes to '{target_path}'")

    def load(self, filepath: Optional[str] = None) -> None:
        """Deserialize graph from JSON file."""
        target_path = filepath or self.graph_path
        if not os.path.exists(target_path):
            logger.warning(f"No graph file found at '{target_path}'. Starting with empty graph.")
            return

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        logger.info(f"Loaded Semantic Graph with {len(self.graph.nodes)} nodes from '{target_path}'")

    def to_text_summary(self) -> str:
        """Format active knowledge graph into clean text lines for LLM prompts."""
        triples = self.get_triples()
        if not triples:
            return "No semantic memory graph records available."

        lines = [f"- {sub} {pred} {obj}" for sub, pred, obj in triples]
        return "\n".join(lines)
