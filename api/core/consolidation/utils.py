from typing import List
import numpy as np
from api.core.memory.models import Memory

def prepare_cluster_context(memories: List[Memory]) -> str:
    context_parts = []
    for memory in memories:
        context_parts.append(f"Memory {memory.id}: {memory.content}")
    return "\n\n".join(context_parts)

def calculate_cluster_centroid(vectors: np.ndarray) -> np.ndarray:
    return np.mean(vectors, axis=0)