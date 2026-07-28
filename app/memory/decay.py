import math
import time
from typing import List
import networkx as nx


def calculate_retention(
    last_accessed: float,
    current_time: float,
    strength: float = 1.0,
    time_scale_hours: float = 24.0,
) -> float:
    """
    Calculate Ebbinghaus retention R = e^(-t / S).
    
    Args:
        last_accessed: Unix timestamp when node was last accessed.
        current_time: Current unix timestamp.
        strength: Memory strength multiplier S (higher S = slower decay).
        time_scale_hours: Time unit scaling factor (default: 24 hours per unit).
        
    Returns:
        Retention score R between 0.0 and 1.0.
    """
    delta_seconds = max(0.0, current_time - last_accessed)
    delta_time_units = delta_seconds / (time_scale_hours * 3600.0)
    
    # Avoid division by zero
    s = max(0.1, strength)
    retention = math.exp(-delta_time_units / s)
    return round(retention, 4)


def reinforce_strength(current_strength: float, boost: float = 0.5, max_strength: float = 10.0) -> float:
    """Increase node memory strength when accessed or re-consolidated."""
    return min(max_strength, current_strength + boost)


def prune_decayed_nodes(
    graph: nx.DiGraph,
    threshold: float = 0.15,
    current_time: float = None,
    time_scale_hours: float = 24.0,
) -> List[str]:
    """
    Evaluate retention score for all graph nodes and prune those below retention threshold.
    
    Returns:
        List of pruned node names.
    """
    if current_time is None:
        current_time = time.time()

    pruned: List[str] = []
    for node, data in list(graph.nodes(data=True)):
        last_accessed = data.get("last_accessed", current_time)
        strength = data.get("strength", 1.0)
        
        retention = calculate_retention(
            last_accessed=last_accessed,
            current_time=current_time,
            strength=strength,
            time_scale_hours=time_scale_hours,
        )
        
        # Store computed retention in node data for inspection
        graph.nodes[node]["retention"] = retention

        if retention < threshold:
            pruned.append(node)

    for node in pruned:
        graph.remove_node(node)

    return pruned
