# enhanced_memory_consolidator.py

import asyncio
import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime, timedelta, timezone
import numpy as np
import hdbscan
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import normalize
import uuid

from api.core.consolidation.config import ConsolidationConfig
from api.core.consolidation.utils import prepare_cluster_context, calculate_cluster_centroid
from api.core.memory.models import Memory, MemoryType
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.core.memory.exceptions import MemoryOperationError
from api.utils.config import get_settings
from functools import lru_cache

# Import new graph-based components
from api.core.memory.graph.memory_graph import MemoryGraph
from api.core.memory.graph.relationship_detector import MemoryRelationshipDetector

logger = logging.getLogger(__name__)

@lru_cache()
def get_consolidation_config() -> ConsolidationConfig:
    settings = get_settings()
    return ConsolidationConfig(
        min_cluster_size=settings.min_cluster_size,
        max_episodic_age_days=7,
        max_memories_per_consolidation=1000
    )

class EnhancedMemoryConsolidator:
    """
    Enhanced memory consolidator with graph-based relationships.
    """
    def __init__(
        self,
        config: ConsolidationConfig,
        pinecone_service: PineconeService,
        llm_service: LLMService,
        memory_graph: MemoryGraph,
        relationship_detector: MemoryRelationshipDetector
    ):
        self.config = config
        self.pinecone_service = pinecone_service
        self.llm_service = llm_service
        self.memory_graph = memory_graph
        self.relationship_detector = relationship_detector
        self.logger = logging.getLogger(__name__)

    async def consolidate_memories(self) -> None:
        """
        Enhanced consolidation process:
        1. Fetch recent episodic memories (last 7 days only)
        2. Add all memories to the graph
        3. Detect relationships between memories
        4. Cluster using HDBSCAN
        5. Generate learned memories from clusters
        6. Add learned memories to the graph with relationships to source memories
        """
        try:
            self.logger.info("Starting enhanced memory consolidation process")
            
            # Calculate the cutoff date for 7 days ago
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.max_episodic_age_days)
            cutoff_timestamp = int(cutoff_date.timestamp())  # Convert to Unix timestamp
                        
            self.logger.info(f"Using cutoff date for episodic memories: {cutoff_date.isoformat()}Z (timestamp: {cutoff_timestamp})")

            # 1. Fetch Episodic Memories with 7-day time filter
            query_results = await self.pinecone_service.query_memories(
                query_vector=[0.0] * self.pinecone_service.embedding_dimension,
                top_k=self.config.max_memories_per_consolidation,
                filter={
                    "memory_type": "EPISODIC",
                    "created_at": {"$gte": cutoff_timestamp}  # Use integer timestamp instead of string
                },
                include_metadata=True
            )

            self.logger.info(f"Pinecone query results: {len(query_results)} memories within last 7 days")
            episodic_memories = []

            for i, mem in enumerate(query_results):
                memory_data = mem[0]
                self.logger.info(f"Processing memory {i}: {memory_data['id']}")

                self.logger.info(f"Raw metadata: {memory_data.get('metadata', {})}")

                created_at = memory_data.get('metadata', {}).get('created_at')
                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                elif not created_at:
                    created_at = datetime.utcnow().isoformat() + "Z"
                self.logger.info(f"Processed created_at: {created_at}")

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
                    self.logger.error(f"Error creating Memory object from Pinecone result: {e}")
                    self.logger.error(f"Problematic memory data: {memory_data}")
                    continue

            if not episodic_memories:
                self.logger.info("No episodic memories found for consolidation")
                return

            # 2. Add all memories to the graph
            self.logger.info("Adding memories to the graph...")
            for memory in episodic_memories:
                self.memory_graph.add_memory_node(memory)
            
            # 3. Detect relationships between memories in the graph
            # This is a potentially expensive operation, so we'll use batching
            await self._detect_memory_relationships(episodic_memories)

            # 4. Prepare Vectors for HDBSCAN clustering
            vectors = np.array([mem.semantic_vector for mem in episodic_memories])
            self.logger.info(f"Shape of vectors array: {vectors.shape}")
            if vectors.size == 0:
                self.logger.info("No vectors to cluster.")
                return
            if len(vectors.shape) == 1:
                self.logger.info("Only one vector, reshaping.")
                vectors = vectors.reshape(1, -1)

            # Normalize the vectors
            vectors = normalize(vectors)

            # 5. Perform Clustering (HDBSCAN)
            self.logger.info(f"Running HDBSCAN with min_cluster_size: {self.config.min_cluster_size}")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                metric='euclidean',
            )
            clusters = clusterer.fit_predict(vectors)
            self.logger.info(f"HDBSCAN clusters: {clusters}")

            self.logger.info(f"Unique cluster labels (before removing -1): {np.unique(clusters)}")

            # 6. Create Learned Memories
            for cluster_idx in np.unique(clusters):
                if cluster_idx == -1:
                    self.logger.info(f"Skipping noise cluster (-1)")
                    continue

                cluster_memories = [
                    episodic_memories[i] for i, label in enumerate(clusters)
                    if label == cluster_idx
                ]
                self.logger.info(f"Cluster {cluster_idx}: Found {len(cluster_memories)} memories.")

                # Create learned memory
                learned_memory = await self._create_learned_memory(cluster_memories)
                
                # If learned memory was created, add it to the graph
                if learned_memory:
                    self._add_learned_memory_to_graph(learned_memory, cluster_memories)

            # Log graph statistics after consolidation
            stats = self.memory_graph.get_graph_stats()
            self.logger.info(f"Memory graph statistics after consolidation: {stats}")

        except Exception as e:
            self.logger.error(f"Memory consolidation failed: {str(e)}")
            raise MemoryOperationError("consolidation", str(e))

    async def _detect_memory_relationships(self, memories: List[Memory]) -> None:
        """
        Detect relationships between memories and add them to the graph.
        Uses batching to avoid processing too many memory pairs at once.
        """
        self.logger.info(f"Detecting relationships among {len(memories)} memories")
        
        # Use a small batch size to avoid too many concurrent LLM calls
        batch_size = 5
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i+batch_size]
            
            # Process each memory in the batch
            for memory in batch:
                # Get candidate memories (exclude the current memory)
                candidates = [m for m in memories if m.id != memory.id]
                
                # Detect relationships
                relationships = await self.relationship_detector.analyze_memory_relationships(
                    memory, candidates
                )
                
                # Add relationships to the graph
                for rel_type, related_memories in relationships.items():
                    for related_memory, strength in related_memories:
                        self.memory_graph.add_relationship(
                            source_id=memory.id,
                            target_id=related_memory.id,
                            rel_type=rel_type,
                            weight=strength
                        )
            
            # Log progress
            self.logger.info(f"Processed relationships for memories {i} to {min(i+batch_size, len(memories))}")

    async def _create_learned_memory(self, cluster_memories: List[Memory]) -> Optional[Memory]:
        """
        Creates a learned memory from a cluster of episodic memories.
        Returns the created Memory object or None if creation failed.
        """
        if not cluster_memories:
            return None

        try:
            self.logger.info(f"Creating learned memory from {len(cluster_memories)} episodic memories")
            combined_content = "\n".join([mem.content for mem in cluster_memories])

            self.logger.info(f"Combined content for LLM: {combined_content[:200]}...")
            consolidated_content = await self.llm_service.generate_summary(combined_content)

            if not consolidated_content:
                self.logger.warning("LLM returned empty content for learned memory. Skipping.")
                return None

            self.logger.info(f"LLM generated content: {consolidated_content[:200]}...")

            centroid_vector = calculate_cluster_centroid(cluster_memories)
            # Normalize the centroid
            centroid_vector = normalize(centroid_vector.reshape(1, -1))[0]

            # Create learned memory ID with prefix
            learned_memory_id = f"mem_{str(uuid.uuid4())}"
            
            # Get current time and create both timestamp formats
            current_time = datetime.now(timezone.utc)
            created_at_timestamp = int(current_time.timestamp())
            created_at_iso = current_time.isoformat() + "Z"
            
            # Use integer timestamp in metadata for Pinecone
            metadata = {
                "content": consolidated_content,
                "memory_type": "LEARNED",
                "created_at": created_at_timestamp,  # Integer timestamp for Pinecone filtering
                "source": "learned memories",
                # Additional metadata for graph analysis
                "source_episodic_ids": [mem.id for mem in cluster_memories],
                "creation_method": "consolidation_hdbscan",
                "cluster_size": len(cluster_memories)
            }

            # Store in Pinecone
            await self.pinecone_service.create_memory(
                memory_id=learned_memory_id,
                vector=centroid_vector.tolist(),
                metadata=metadata
            )
            self.logger.info(f"Learned memory '{learned_memory_id}' created successfully.")

            # Create and return a Memory object with ISO string format for the timestamp
            memory = Memory(
                id=learned_memory_id,
                content=consolidated_content,
                memory_type=MemoryType.LEARNED,
                created_at=created_at_iso,  # ISO format string for Memory object
                metadata=metadata,
                semantic_vector=centroid_vector.tolist()
            )
            
            return memory

        except Exception as e:
            self.logger.error(f"Failed to create learned memory: {e}")
            return None

    def _add_learned_memory_to_graph(self, learned_memory: Memory, source_memories: List[Memory]) -> None:
        """
        Add a learned memory to the graph with relationships to its source memories.
        """
        try:
            # Add the learned memory node
            self.memory_graph.add_memory_node(learned_memory)
            
            # Create relationships between the learned memory and its source memories
            for source_memory in source_memories:
                # Bidirectional relationships:
                # 1. Source -> Learned (elaboration: source contributed to learned)
                self.memory_graph.add_relationship(
                    source_id=source_memory.id,
                    target_id=learned_memory.id,
                    rel_type=self.relationship_detector.REL_ELABORATION,
                    weight=0.8,
                    metadata={"contribution": "source"}
                )
                
                # 2. Learned -> Source (hierarchical: learned summarizes source)
                self.memory_graph.add_relationship(
                    source_id=learned_memory.id,
                    target_id=source_memory.id,
                    rel_type=self.relationship_detector.REL_HIERARCHICAL,
                    weight=0.9,
                    metadata={"relationship": "summarizes"}
                )
            
            self.logger.info(f"Added learned memory {learned_memory.id} to graph with {len(source_memories)} source relationships")
            
        except Exception as e:
            self.logger.error(f"Failed to add learned memory to graph: {e}")
            # Continue execution even if graph addition fails