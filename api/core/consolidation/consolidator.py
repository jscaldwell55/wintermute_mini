# api/core/consolidation/consolidator.py
import asyncio
import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime, timedelta, timezone
import numpy as np
import hdbscan
from typing import List
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
        # No longer need to pass consolidation_prompt or context_length
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
        1.  Fetch episodic memories.
        2.  Cluster using HDBSCAN.
        3.  Generate semantic memories from clusters.
        """
        try:
            logger.info("Starting memory consolidation process")

            # 1. Fetch Episodic Memories (NO Time-Based filter)
            query_results = await self.pinecone_service.query_memories(
                query_vector=[0.0] * self.pinecone_service.embedding_dimension,
                top_k=1000,
                filter={"memory_type": "EPISODIC"},
                include_metadata=True
            )

            logger.info(f"Pinecone query results: {len(query_results)} memories")
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

            # 3. Perform Clustering (HDBSCAN)
            logger.info(f"Running HDBSCAN with min_cluster_size: {self.config.min_cluster_size}")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                metric='cosine',
            )
            clusters = clusterer.fit_predict(vectors)
            logger.info(f"HDBSCAN clusters: {clusters}")

            logger.info(f"Unique cluster labels (before removing -1): {np.unique(clusters)}")

            # 4. Create Semantic Memories (Simplified)
            for cluster_idx in np.unique(clusters):
                if cluster_idx == -1:
                    logger.info(f"Skipping noise cluster (-1)")
                    continue

                cluster_memories = [
                    episodic_memories[i] for i, label in enumerate(clusters)
                    if label == cluster_idx
                ]
                logger.info(f"Cluster {cluster_idx}: Found {len(cluster_memories)} memories.")

                await self._create_semantic_memory(cluster_memories)

        except Exception as e:
            logger.error(f"Memory consolidation failed: {str(e)}")
            raise MemoryOperationError("consolidation", str(e))

    async def _create_semantic_memory(self, cluster_memories: List[Memory]) -> None:
        """Creates a semantic memory from a cluster of episodic memories."""
        if not cluster_memories:
            return

        try:
            logger.info(f"Creating semantic memory from {len(cluster_memories)} episodic memories")
            # Combine the content of the memories.  NO prompt formatting here.
            combined_content = "\n".join([mem.content for mem in cluster_memories])

            logger.info(f"Combined content for LLM: {combined_content[:200]}...")
            consolidated_content = await self.llm_service.generate_summary(combined_content) # Use generate_summary

            if not consolidated_content:
                logger.warning("LLM returned empty content for semantic memory. Skipping.")
                return

            logger.info(f"LLM generated content: {consolidated_content[:200]}...")

            centroid_vector = calculate_cluster_centroid(cluster_memories)

            semantic_memory_id = str(uuid.uuid4())
            metadata = {
                "content": consolidated_content,
                "memory_type": "SEMANTIC",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "source_episodic_ids": [mem.id for mem in cluster_memories],
                "creation_method": "consolidation_hdbscan", # Added creation method
                "cluster_size": len(cluster_memories) # Added cluster size
            }

            await self.pinecone_service.create_memory(
                memory_id=semantic_memory_id,
                vector=centroid_vector.tolist(),
                metadata=metadata
            )
            logger.info(f"Semantic memory '{semantic_memory_id}' created successfully.")

        except Exception as e:
            logger.error(f"Failed to create semantic memory: {e}")
            #Don't raise, keep processing other clusters.