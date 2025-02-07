# api/core/consolidation/consolidator.py
import asyncio
import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime, timedelta, timezone  # Import timezone
import numpy as np
#from sklearn.cluster import DBSCAN # Removed DBSCAN
import hdbscan  # Import HDBSCAN
from typing import List

#from sklearn.metrics.pairwise import cosine_distances  # Import cosine_distances  -- No longer needed

from api.core.consolidation.models import ConsolidationConfig
from api.core.consolidation.utils import prepare_cluster_context, calculate_cluster_centroid
from api.core.memory.models import Memory, MemoryType  # <--- IMPORT MemoryType
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.core.memory.exceptions import MemoryOperationError

logger = logging.getLogger(__name__)

class MemoryConsolidator: # No longer inherits from anything.
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
        1.  Fetch recent episodic memories.
        2.  Cluster using HDBSCAN.
        3.  Generate semantic memories from clusters.
        4.  Archive old episodic memories.
        """
        try:
            logger.info("Starting memory consolidation process")

            # 1. Fetch Recent Episodic Memories (Time-Based)
            cutoff_time = datetime.utcnow() - timedelta(days=self.config.max_age_days)
            cutoff_timestamp = int(cutoff_time.timestamp()) #int timestamp
            logger.info(f"Consolidation cutoff timestamp: {cutoff_timestamp}")

            query_results = await self.pinecone_service.query_memories(
                query_vector=[0.0] * self.pinecone_service.embedding_dimension,  # Use correct dimension
                top_k=1000, # Fetch many, the filter limits.  Adjust as needed.
                filter={
                    "memory_type": "EPISODIC",
                    "created_at": {"$gte": cutoff_timestamp}  # Correct time-based filter
                },
                include_metadata = True
            )

            logger.info(f"Pinecone query results: {len(query_results)} memories")
            episodic_memories = []
            for i, mem in enumerate(query_results):
                memory_data = mem[0]
                logger.info(f"Processing memory {i}: {memory_data['id']}")

                # Log the raw metadata *before* any processing:
                logger.info(f"Raw metadata: {memory_data.get('metadata', {})}")

                created_at = memory_data.get('metadata', {}).get('created_at')
                # No change needed here, your existing code is good.
                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                elif not created_at:
                    created_at = datetime.utcnow().isoformat()
                # Log the created_at value *after* processing:
                logger.info(f"Processed created_at: {created_at}")

                try:
                  memory = Memory(
                      id=memory_data['id'],
                      content=memory_data['metadata']['content'],
                      memory_type=MemoryType(memory_data['metadata']['memory_type']),
                      semantic_vector=memory_data.get('vector', [0.0] * self.pinecone_service.embedding_dimension),  # Use service's dimension
                      metadata=memory_data.get('metadata', {}),
                      created_at=created_at,  # Use the parsed datetime object
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
                # No eps parameter needed for HDBSCAN
                #allow_single_cluster=False #Removed to match intended dbscan settings
                )
            clusters = clusterer.fit_predict(vectors)
            logger.info(f"HDBSCAN clusters: {clusters}")

            logger.info(f"Unique cluster labels (before removing -1): {np.unique(clusters)}")

            # 4. Create Semantic Memories (Simplified)
            for cluster_idx in np.unique(clusters):
                if cluster_idx == -1:  # Ignore noise points
                    logger.info(f"Skipping noise cluster (-1)")
                    continue

                cluster_memories = [
                    episodic_memories[i] for i, label in enumerate(clusters)
                    if label == cluster_idx
                ]
                logger.info(f"Cluster {cluster_idx}: Found {len(cluster_memories)} memories.")

                await self._create_semantic_memory(cluster_memories)

            # 5. Archive Old Memories
            # await self._archive_old_memories(episodic_memories) # temporarily removed for testing.

        except Exception as e:
            logger.error(f"Memory consolidation failed: {str(e)}")
            raise MemoryOperationError("consolidation", str(e))


    async def _create_semantic_memory(self, cluster_memories: List[Memory]) -> None:
        """Generate a semantic memory from a cluster of episodic memories (Simplified)."""
        try:
            # --- Prepare context: simple concatenation for MVE ---
            context = "\n".join([mem.content for mem in cluster_memories])

            # --- Generate semantic summary using LLM ---
            summary = await self.llm_service.generate_summary(context)

            # --- Calculate centroid: simple average for MVE ---
            centroid = np.mean([mem.semantic_vector for mem in cluster_memories], axis=0).tolist()

            # --- Create memory data ---
            memory_data = {
                "content": summary,
                "memory_type": MemoryType.SEMANTIC.value,  # Use .value here
                "metadata": {
                    "source_memories": [mem.id for mem in cluster_memories],
                    "creation_method": "consolidation_hdbscan",  # Indicate the method, changed to hdbscan
                    "cluster_size": len(cluster_memories),
                    "created_at": int(datetime.utcnow().timestamp()), #<-- Storing as integer timestamp
                    # No window_id for semantic memories (for MVE simplicity)
                }
            }

            # --- Store using query-compatible format ---
            await self.pinecone_service.create_memory(
                memory_id=f"sem_{datetime.utcnow().timestamp()}",  # Use a prefix and timestamp
                vector=centroid,
                metadata=memory_data
            )

        except Exception as e:
            logger.error(f"Failed to create semantic memory: {e}")
            raise MemoryOperationError("semantic_memory_creation", str(e))