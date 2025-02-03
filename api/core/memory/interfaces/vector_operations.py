# api/core/memory/interfaces/vector_operations.py
from abc import ABC, abstractmethod
from typing import List

class VectorOperations(ABC):

    @abstractmethod
    async def create_semantic_vector(self, text: str) -> List[float]:
        """Creates a semantic vector embedding for given text."""
        pass

    @abstractmethod
    async def create_batch_vectors(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Creates semantic vector embeddings for a list of texts in batches."""
        pass

    @abstractmethod  # Add this if you want to enforce it in the interface
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates the cosine similarity between two vectors."""
        pass

    @abstractmethod  # Add this if you want to enforce it in the interface
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalizes a vector to unit length."""
        pass