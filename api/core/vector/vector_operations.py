from typing import List
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from api.utils.config import get_settings
import logging

logger = logging.getLogger(__name__)

class VectorOperations:
    """Handles vector operations using OpenAI's embeddings."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def create_semantic_vector(self, text: str) -> List[float]:
        """
        Creates a semantic vector embedding for given text using OpenAI's API.

        Args:
            text: The text to create a vector for.

        Returns:
            A list of floats representing the semantic vector.
        """
        try:
            # Clean text
            text = text.replace("\n", " ")

            # Generate embedding
            response = await self.client.embeddings.create(
                input=[text],
                model=self.settings.vector_model_id
            )
            
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalizes a vector to unit length.

        Args:
            vector: The vector to normalize

        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (np.array(vector) / norm).tolist()