import asyncio
import logging
from datetime import datetime
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List

from api.core.consolidation.models import ConsolidationConfig
from api.core.consolidation.utils import prepare_cluster_context, calculate_cluster_centroid
from api.core.memory.models import Memory
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.utils.responses import MemoryOperationError

logger = logging.getLogger(__name__)

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
        Main consolidation process that runs periodically to:
        1. Cluster similar episodic memories
        2. Generate semantic memories from clusters
        3. Archive old episodic memories
        """
        try:
            # Fetch recent episodic memories
            episodic_memories = await self.pinecone_service.fetch_recent_memories(
                memory_type="EPISODIC",
                max_age_days=self.config.max_age_days
            )
            
            if not episodic_memories:
                logger.info("No episodic memories found for consolidation")
                return
            
            # Convert memories to vector format for clustering
            vectors = np.array([mem.vector for mem in episodic_memories])
            
            # Ensure proper shape for DBSCAN
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            elif len(vectors.shape) > 2:
                raise MemoryOperationError(
                    "consolidation",
                    f"Invalid vector shape: {vectors.shape}"
                )
            
            # Perform DBSCAN clustering only if we have enough memories
            if vectors.shape[0] >= self.config.min_cluster_size:
                clusters = self._cluster_memories(vectors)
                
                # Process each significant cluster
                for cluster_idx in np.unique(clusters):
                    if cluster_idx == -1:  # Skip noise points
                        continue
                        
                    cluster_memories = [
                        episodic_memories[i] for i, label in enumerate(clusters)
                        if label == cluster_idx
                    ]
                    
                    if len(cluster_memories) >= self.config.min_cluster_size:
                        await self._create_semantic_memory(cluster_memories)
            else:
                logger.info(f"Not enough memories for clustering. Found {vectors.shape[0]}, need {self.config.min_cluster_size}")
            
            # Archive old memories regardless of clustering
            await self._archive_old_memories(episodic_memories)
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {str(e)}")
            raise MemoryOperationError("consolidation", str(e))

    def _cluster_memories(self, vectors: np.ndarray) -> np.ndarray:
        """
        Cluster memory vectors using DBSCAN algorithm.
        """
        if vectors.shape[0] < self.config.min_cluster_size:
            return np.array([-1] * vectors.shape[0])  # All points marked as noise
            
        clustering = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_cluster_size,
            metric='cosine'
        ).fit(vectors)
        return clustering.labels_

    async def _create_semantic_memory(self, cluster_memories: List[Memory]) -> None:
        """
        Generate a semantic memory from a cluster of similar episodic memories.
        """
        # Prepare context for LLM
        context = prepare_cluster_context(cluster_memories)
        
        # Generate semantic summary using LLM
        summary = await self.llm_service.generate_summary(context)
        
        # Create and store semantic memory
        semantic_memory = Memory(
            content=summary,
            memory_type="SEMANTIC",
            metadata={
                "source_memories": [mem.id for mem in cluster_memories],
                "creation_method": "consolidation",
                "cluster_size": len(cluster_memories)
            }
        )
        
        await self.pinecone_service.store_memory(semantic_memory)

    async def _archive_old_memories(self, memories: List[Memory]) -> None:
        """
        Archive old episodic memories that have been consolidated.
        """
        current_time = datetime.utcnow()
        for memory in memories:
            age_days = (current_time - memory.created_at).days
            if age_days > self.config.max_age_days:
                await self.pinecone_service.archive_memory(memory.id)

async def run_consolidation(consolidator: MemoryConsolidator):
    """Background task for memory consolidation"""
    while True:
        try:
            await consolidator.consolidate_memories()
        except Exception as e:
            logger.error(f"Consolidation error: {str(e)}")
        finally:
            await asyncio.sleep(consolidator.config.consolidation_interval_hours * 3600)