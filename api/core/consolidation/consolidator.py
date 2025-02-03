# consolidator.py
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
from api.core.memory.exceptions import MemoryOperationError

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
            # Fetch recent episodic memories using query instead of direct fetch
            query_results = await self.pinecone_service.query_memories(
                query_vector=[0.1] * 1536,  # Default query vector
                top_k=1000,
                filter={"memory_type": "EPISODIC"}
            )
        
            # Create memory objects with proper datetime handling
            episodic_memories = []
            for mem in query_results:
                memory_data = mem[0]  # Get the memory data from the tuple
                # Ensure created_at is a string in ISO format
                created_at = memory_data.get('metadata', {}).get('created_at')
                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                elif not created_at:
                    created_at = datetime.utcnow().isoformat()

                memory = Memory(
                    id=memory_data['id'],
                    content=memory_data['metadata']['content'],
                    memory_type=memory_data['memory_type'],
                    semantic_vector=memory_data.get('vector', [0.1] * 1536),
                    metadata=memory_data.get('metadata', {}),
                    created_at=created_at,  # Now we're sure this is a string
                    window_id=memory_data.get('metadata', {}).get('window_id')
                )
                episodic_memories.append(memory)
        
            if not episodic_memories:
                logger.info("No episodic memories found for consolidation")
                return

            # Adjust clustering parameters based on current memory set
            if isinstance(self, AdaptiveConsolidator):
                await self.adjust_clustering_params(episodic_memories)
        
            # Convert memories to vector format for clustering
            vectors = np.array([mem.semantic_vector for mem in episodic_memories])
        
            # Reshape vectors if needed
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            elif len(vectors.shape) > 2:
                raise MemoryOperationError(
                    "consolidation",
                    f"Invalid vector shape: {vectors.shape}"
                )
        
            # Perform clustering
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


            # Perform clustering
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
        
            # Archive old memories
            await self._archive_old_memories(episodic_memories)
        
        except Exception as e:
            logger.error(f"Memory consolidation failed: {str(e)}")
            raise MemoryOperationError("consolidation", str(e))

    async def _create_semantic_memory(self, cluster_memories: List[Memory]) -> None:
        """Generate a semantic memory from a cluster of similar episodic memories."""
        try:
            # Prepare context for LLM
            context = prepare_cluster_context(cluster_memories)
        
            # Generate semantic summary
            summary = await self.llm_service.generate_summary(context)
        
            # Calculate centroid vector for the cluster
            centroid = calculate_cluster_centroid([mem.semantic_vector for mem in cluster_memories])
        
            # Convert numpy array to list
            if hasattr(centroid, 'tolist'):
                centroid = centroid.tolist()
        
            # Create memory data
            memory_data = {
                "content": summary,
                "memory_type": "SEMANTIC",
                "metadata": {
                    "source_memories": [mem.id for mem in cluster_memories],
                    "creation_method": "consolidation",
                    "cluster_size": len(cluster_memories),
                    "created_at": datetime.utcnow().isoformat(),
                }
            }
        
            # Store using query-compatible format
            await self.pinecone_service.create_memory(
                memory_id=f"sem_{datetime.utcnow().timestamp()}",
                vector=centroid,
                metadata=memory_data
            )

        except Exception as e:
            logger.error(f"Failed to create semantic memory: {e}")
            raise MemoryOperationError("semantic_memory_creation", str(e))

    def _cluster_memories(self, vectors: np.ndarray) -> np.ndarray:
        """Cluster memory vectors using DBSCAN algorithm."""
        if vectors.shape[0] < self.config.min_cluster_size:
            return np.array([-1] * vectors.shape[0])
            
        clustering = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_cluster_size,
            metric='cosine'
        ).fit(vectors)
        return clustering.labels_

    async def _archive_old_memories(self, memories: List[Memory]) -> None:
        """Archive old episodic memories that have been consolidated."""
        current_time = datetime.utcnow()
        for memory in memories:
            try:
                age_days = (current_time - memory.created_at).days
                if age_days > self.config.max_age_days:
                    await self.pinecone_service.update_memory(
                        memory_id=memory.id,
                        vector=memory.semantic_vector,
                        metadata={
                            **memory.metadata,
                            "archived": True,
                            "archived_at": current_time.isoformat()
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to archive memory {memory.id}: {e}")
                continue

class AdaptiveConsolidator(MemoryConsolidator):  # Inherit from MemoryConsolidator

    def __init__(self, config, pinecone_service, llm_service):
        self.config = config
        self.pinecone_service = pinecone_service
        self.llm_service = llm_service

        
    def _calculate_nearest_neighbor_distances(self, vectors: np.ndarray) -> List[float]:
        """Calculate the average distance of each point to its nearest neighbor."""
        if vectors.shape[0] <= 1:
            return [0.0]  # No distances to calculate with 0 or 1 points

        distances = []
        for i in range(vectors.shape[0]):
            # Calculate Euclidean distances from point i to all other points
            dists = np.linalg.norm(vectors - vectors[i], axis=1)
            
            # Exclude the point itself (distance 0) and get the minimum distance
            nearest_neighbor_dist = np.min(dists[dists > 0])
            distances.append(nearest_neighbor_dist)
        
        return distances

    async def adjust_clustering_params(self, memories: List[Memory]) -> None:
        """
        Adjust clustering parameters based on memory density.
        """
        if not memories:
            logger.info("No memories to adjust clustering parameters.")
            return

        vectors = np.array([mem.semantic_vector for mem in memories])

        # Ensure that vectors are reshaped correctly for the distance calculations
        if len(vectors.shape) != 2:
            logger.warning(f"Unexpected vectors shape: {vectors.shape}. Expected 2D array.")
            return

        # Adjust eps based on memory density
        distances = self._calculate_nearest_neighbor_distances(vectors)
        
        # Calculate epsilon as the median of the nearest neighbor distances
        self.config.eps = np.median(distances) if distances else 0.5
        
        # Adjust cluster size based on total memories
        self.config.min_cluster_size = max(
            3,
            int(len(memories) * 0.05)  # 5% of total memories
        )
        logger.info(f"Clustering parameters adjusted - eps: {self.config.eps}, min_cluster_size: {self.config.min_cluster_size}")

async def run_consolidation(consolidator: MemoryConsolidator):
    """Background task for memory consolidation"""
    while True:
        try:
            await consolidator.consolidate_memories()
        except Exception as e:
            logger.error(f"Consolidation error: {str(e)}")
        finally:
            await asyncio.sleep(consolidator.config.consolidation_interval_hours * 3600)