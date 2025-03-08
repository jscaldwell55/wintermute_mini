# core/vector/vector_operations.py
from typing import List
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log
from api.utils.config import get_settings
import logging
logging.basicConfig(level=logging.INFO)
from api.core.memory.interfaces.vector_operations import VectorOperations

logger = logging.getLogger(__name__)

class VectorOperationsImpl(VectorOperations):
    """Handles vector operations using OpenAI's embeddings."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3), before_sleep=before_sleep_log(logger, logging.WARNING))
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
                model=self.settings.embedding_model  # Changed from embedding_model_id
            )
        
            # Normalize the embedding before returning
            embedding = response.data[0].embedding
            return self.normalize_vector(embedding)

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def create_batch_vectors(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Creates semantic vector embeddings for a list of texts in batches.

        Args:
            texts: List of texts to create vectors for.
            batch_size: Batch size for processing.

        Returns:
            A list of lists, where each inner list contains floats representing a semantic vector.
        """
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.settings.embedding_model  # Changed from embedding_model_id
                )
                # Normalize each vector before adding to results
                batch_vectors = [self.normalize_vector(data.embedding) for data in response.data]
                vectors.extend(batch_vectors)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                raise
        return vectors
    
    async def create_episodic_memory_vector(self, content: str) -> List[float]:
        """
        Creates a vector specifically optimized for episodic memories by focusing on the user query.
        
        Args:
            content: The full conversation content (typically in User/Assistant format)
            
        Returns:
            A list of floats representing the semantic vector optimized for episodic retrieval
        """
        import re
        
        # Extract just the user's query from the conversation
        query_match = re.match(r"User: (.*?)(\nAssistant:|$)", content)
        
        if query_match:
            user_query = query_match.group(1)
            logger.info(f"Extracted user query for vectorization: {user_query[:50]}...")
            
            # Create vector from user query (gives more weight to the query portion)
            return await self.create_semantic_vector(user_query)
        
        # Fallback to using full content if pattern doesn't match
        logger.warning(f"Could not extract user query from content: {content[:50]}... - using full content")
        return await self.create_semantic_vector(content)


    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors with improved numerical stability.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        try:
            # Convert to numpy arrays for better numerical stability
            v1 = np.array(vec1, dtype=np.float64)
            v2 = np.array(vec2, dtype=np.float64)

            # Normalize vectors (even if they're already normalized, this ensures numerical stability)
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            # Handle zero vectors
            if v1_norm == 0 or v2_norm == 0:
                return 0.0

            v1_normalized = v1 / v1_norm
            v2_normalized = v2 / v2_norm

            # Calculate similarity
            # Calculate similarity and handle numerical precision
            raw_similarity = np.dot(v1_normalized, v2_normalized)
            if raw_similarity > 1.0 and raw_similarity < 1.0 + 1e-10:  # Handle small numerical errors
                similarity = 1.0
            else:
                similarity = np.clip(raw_similarity, -1.0, 1.0)

            return float(similarity)  # Convert from np.float64 to Python float

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            raise

    def normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalizes a vector to unit length with improved numerical stability.

        Args:
            vector: The vector to normalize

        Returns:
            Normalized vector as a list of floats
        """
        try:
            # Convert to numpy array for better numerical stability
            v = np.array(vector, dtype=np.float64)
            norm = np.linalg.norm(v)

            # Handle zero vector
            if norm == 0:
                return [0.0] * len(vector)

            # Normalize and convert back to list
            normalized = (v / norm).tolist()
            
            # Ensure we're returning Python floats, not numpy types
            return [float(x) for x in normalized]

        except Exception as e:
            logger.error(f"Error normalizing vector: {e}")
            raise