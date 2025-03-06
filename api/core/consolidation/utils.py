# api/core/consolidation/utils.py (CORRECTED)
import logging
from typing import List
import numpy as np
from api.core.memory.models import Memory

logger = logging.getLogger(__name__)

def prepare_cluster_context(memories: List[Memory], max_length: int = 300) -> str:
    """Prepares context from a list of memories for LLM prompting."""
    context = ""
    for memory in memories:
        if len(context) + len(memory.content) + 1 <= max_length:
            context += memory.content + "\n"
    return context.strip()

def calculate_cluster_centroid(memories: List[Memory]) -> np.ndarray:
    """Calculates the centroid of the semantic vectors of a list of memories."""
    if not memories:
        raise ValueError("Cannot calculate centroid of empty memory list.")

    # Extract vectors and ensure they are NumPy arrays
    vectors = [np.array(mem.semantic_vector) for mem in memories]

    # Stack the vectors into a 2D array
    vector_array = np.stack(vectors)

    # Calculate the mean along axis 0 (the vector dimension)
    centroid = np.mean(vector_array, axis=0)
    return centroid