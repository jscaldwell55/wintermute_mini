# api/core/consolidation/utils.py
# No changes needed for the MVE simplification

from typing import List
import numpy as np
from api.core.memory.models import Memory

def prepare_cluster_context(memories: List[Memory]) -> str:
    # Simplified for MVE: Just concatenate the content
    return "\n".join([mem.content for mem in memories])

def calculate_cluster_centroid(vectors: np.ndarray) -> np.ndarray:
    # Simplified for MVE: Simple average
    return np.mean(vectors, axis=0)