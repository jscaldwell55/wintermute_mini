import asyncio
import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime, timedelta, timezone
import numpy as np
import hdbscan
from typing import List
from sklearn.preprocessing import normalize  # Import normalize
import uuid

from api.core.consolidation.config import ConsolidationConfig
from api.core.consolidation.utils import prepare_cluster_context, calculate_cluster_centroid
from api.core.memory.models import Memory, MemoryType
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.core.memory.exceptions import MemoryOperationError
from api.utils.config import get_settings
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache()
def get_consolidation_config() -> ConsolidationConfig:
    settings = get_settings()
    return ConsolidationConfig(
        min_cluster_size=settings.min_cluster_size,
        max_episodic_age_days=7,
        max_memories_per_consolidation=1000
    )

class MemoryConsolidator:
    def __init__(
        self,
        config: ConsolidationConfig,
        pinecone_service: PineconeService,
        llm_service: LLMService
    ):
        self.config = config
        self.pinecone_service = pinecone_service
        self.llm_service = llm_service

    async def consolidate_memories(self) -> None:
        """
        Consolidation process:
        1. Fetch recent episodic memories (last 7 days only).
        2. Cluster using HDBSCAN.
        3. Generate learned memories from clusters.
        """
        try:
            logger.info("Starting memory consolidation process")
            
            # Calculate the cutoff date for 7 days ago
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.max_episodic_age_days)
            cutoff_date_str = cutoff_date.isoformat() + "Z"  # Add Z for UTC timezone
            
            logger.info(f"Using cutoff date for episodic memories: {cutoff_date_str}")

            # 1. Fetch Episodic Memories with 7-day time filter
            query_results = await self.pinecone_service.query_memories(
                query_vector=[0.0] * self.pinecone_service.embedding_dimension,
                top_k=self.config.max_memories_per_consolidation,
                filter={
                    "memory_type": "EPISODIC",
                    "created_at": {"$gte": cutoff_date_str}  # Only consider memories created in last 7 days
                },
                include_metadata=True
            )

            logger.info(f"Pinecone query results: {len(query_results)} memories within last 7 days")
            episodic_memories = []

            for i, mem in enumerate(query_results):
                memory_data = mem[0]
                logger.info(f"Processing memory {i}: {memory_data['id']}")

                logger.info(f"Raw metadata: {memory_data.get('metadata', {})}")

                created_at = memory_data.get('metadata', {}).get('created_at')
                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                elif not created_at:
                    created_at = datetime.utcnow().isoformat() + "Z"
                logger.info(f"Processed created_at: {created_at}")

                try:
                    memory = Memory(
                      id=memory_data['id'],
                      content=memory_data['metadata']['content'],
                      memory_type=MemoryType(memory_data['metadata']['memory_type']),
                      semantic_vector=memory_data.get('vector', [0.0] * self.pinecone_service.embedding_dimension),
                      metadata=memory_data.get('metadata', {}),
                      created_at=created_at,
                      window_id=memory_data.get('metadata', {}).get('window_id')
                    )
                    episodic_memories.append(memory)
                except Exception as e:
                    logger.error(f"Error creating Memory object from Pinecone result: {e}")
                    logger.error(f"Problematic memory data: {memory_data}")
                    continue

            if not episodic_memories:
                logger.info("No episodic memories found for consolidation")
                return

            # 2. Prepare Vectors
            vectors = np.array([mem.semantic_vector for mem in episodic_memories])
            logger.info(f"Shape of vectors array: {vectors.shape}")
            if vectors.size == 0:
                logger.info("No vectors to cluster.")
                return
            if len(vectors.shape) == 1:
                logger.info("Only one vector, reshaping.")
                vectors = vectors.reshape(1, -1)

            # --- Normalize the vectors ---
            vectors = normalize(vectors)


            # 3. Perform Clustering (HDBSCAN)
            logger.info(f"Running HDBSCAN with min_cluster_size: {self.config.min_cluster_size}")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                metric='euclidean',  # Use Euclidean distance
            )
            clusters = clusterer.fit_predict(vectors)
            logger.info(f"HDBSCAN clusters: {clusters}")

            logger.info(f"Unique cluster labels (before removing -1): {np.unique(clusters)}")

            # 4. Create Learned Memories (instead of Semantic)
            for cluster_idx in np.unique(clusters):
                if cluster_idx == -1:
                    logger.info(f"Skipping noise cluster (-1)")
                    continue

                cluster_memories = [
                    episodic_memories[i] for i, label in enumerate(clusters)
                    if label == cluster_idx
                ]
                logger.info(f"Cluster {cluster_idx}: Found {len(cluster_memories)} memories.")

                await self._create_learned_memory(cluster_memories)

        except Exception as e:
            logger.error(f"Memory consolidation failed: {str(e)}")
            raise MemoryOperationError("consolidation", str(e))

    async def _create_learned_memory(self, cluster_memories: List[Memory]) -> None:
        """Creates a learned memory from a cluster of episodic memories."""
        if not cluster_memories:
            return

        try:
            logger.info(f"Creating learned memory from {len(cluster_memories)} episodic memories")
            combined_content = "\n".join([mem.content for mem in cluster_memories])

            logger.info(f"Combined content for LLM: {combined_content[:200]}...")
            consolidated_content = await self.llm_service.generate_summary(combined_content)

            if not consolidated_content:
                logger.warning("LLM returned empty content for learned memory. Skipping.")
                return

            logger.info(f"LLM generated content: {consolidated_content[:200]}...")

            centroid_vector = calculate_cluster_centroid(cluster_memories)
            # --- Normalize the centroid ---
            centroid_vector = normalize(centroid_vector.reshape(1, -1))[0] # Reshape and normalize

            # Create learned memory ID with prefix
            learned_memory_id = f"mem_{str(uuid.uuid4())}"
            
            # Use the required structure for learned memories
            metadata = {
                "content": consolidated_content,
                "memory_type": "LEARNED",
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "source": "learned memories"
                # Additional metadata for debugging that won't affect functionality
                # "source_episodic_ids": [mem.id for mem in cluster_memories],
                # "creation_method": "consolidation_hdbscan",
                # "cluster_size": len(cluster_memories)
            }

            await self.pinecone_service.create_memory(
                memory_id=learned_memory_id,
                vector=centroid_vector.tolist(),  # tolist() after normalization
                metadata=metadata
            )
            logger.info(f"Learned memory '{learned_memory_id}' created successfully.")

        except Exception as e:
            logger.error(f"Failed to create learned memory: {e}")
            # Don't raise, keep processing other clusters