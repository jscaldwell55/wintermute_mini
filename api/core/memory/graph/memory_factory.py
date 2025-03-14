# memory_factory.py

import logging
from typing import Optional

from api.core.memory.graph.memory_graph import MemoryGraph
from api.core.memory.graph.relationship_detector import MemoryRelationshipDetector
from api.core.memory.graph.graph_memory_retriever import GraphMemoryRetriever
from api.core.consolidation.enhanced_memory_consolidator import EnhancedMemoryConsolidator
from api.core.consolidation.config import ConsolidationConfig
from api.utils.pinecone_service import PineconeService
from api.core.vector.vector_operations import VectorOperationsImpl

from api.utils.llm_service import LLMService

logger = logging.getLogger(__name__)

class GraphMemoryFactory:
    """
    Factory class for creating and initializing graph-based memory components.
    """
    
    @staticmethod
    async def create_graph_memory_system(
        pinecone_service: PineconeService,
        vector_operations: VectorOperationsImpl,
        llm_service: LLMService,
        config: Optional[ConsolidationConfig] = None
    ) -> dict:
        """
        Create and initialize all graph-based memory components.
        
        Args:
            pinecone_service: Pinecone service for vector operations
            vector_operations: Vector operations for embeddings
            llm_service: LLM service for content generation
            config: Optional consolidation configuration
            
        Returns:
            Dictionary containing all initialized components
        """
        logger.info("Creating graph-based memory system components")
        
        # Create memory graph
        memory_graph = MemoryGraph()
        logger.info("Memory graph initialized")
        
        # Create relationship detector
        relationship_detector = MemoryRelationshipDetector(llm_service)
        logger.info("Relationship detector initialized")
        
        # Create graph memory retriever
        graph_retriever = GraphMemoryRetriever(
            memory_graph=memory_graph,
            pinecone_service=pinecone_service,
            vector_operations=vector_operations
        )
        logger.info("Graph memory retriever initialized")
        
        # Create enhanced memory consolidator
        if not config:
            # Use default config if not provided
            from api.utils.config import get_consolidation_config
            config = get_consolidation_config()
            
        enhanced_consolidator = EnhancedMemoryConsolidator(
            config=config,
            pinecone_service=pinecone_service,
            llm_service=llm_service,
            memory_graph=memory_graph,
            relationship_detector=relationship_detector
        )
        logger.info("Enhanced memory consolidator initialized")
        
        # Return all components
        return {
            "memory_graph": memory_graph,
            "relationship_detector": relationship_detector,
            "graph_retriever": graph_retriever,
            "enhanced_consolidator": enhanced_consolidator
        }
    
    @staticmethod
    async def initialize_graph_from_existing_memories(
        memory_graph: MemoryGraph,
        relationship_detector: MemoryRelationshipDetector,
        pinecone_service: PineconeService,
        batch_size: int = 100,
        max_memories: int = 500
    ) -> bool:
        """
        Initialize the memory graph from existing memories in Pinecone.
        This is useful for populating the graph with existing memories on startup.
        
        Args:
            memory_graph: The memory graph to populate
            relationship_detector: For detecting relationships
            pinecone_service: For retrieving memories
            batch_size: Number of memories to process at once
            max_memories: Maximum number of memories to retrieve
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Initializing memory graph from existing memories (max {max_memories})")
            
            # Query for existing memories (all types)
            query_results = await pinecone_service.query_memories(
                query_vector=[0.0] * pinecone_service.embedding_dimension,
                top_k=max_memories,
                filter={},  # No filter to get all memory types
                include_metadata=True
            )
            
            logger.info(f"Retrieved {len(query_results)} memories for graph initialization")
            
            # Process in batches
            memories = []
            for memory_data, _ in query_results:
                try:
                    # Convert to Memory objects
                    from api.core.memory.models import Memory, MemoryType
                    
                    memory = Memory(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                        created_at=memory_data["metadata"]["created_at"],
                        metadata=memory_data["metadata"],
                        semantic_vector=memory_data.get("vector")
                    )
                    memories.append(memory)
                except Exception as e:
                    logger.error(f"Error processing memory for graph initialization: {e}")
                    continue
            
            # Add nodes to graph
            for memory in memories:
                memory_graph.add_memory_node(memory)
            
            logger.info(f"Added {len(memories)} memory nodes to graph")
            logger.info("Adding temporal relationships between memories")
            memory_graph.add_temporal_relationships()
            
            # Process relationships in smaller batches to avoid overwhelming the LLM service
            max_relationship_batch = 20
            relationship_count = 0
            
            for i in range(0, len(memories), max_relationship_batch):
                batch = memories[i:i+max_relationship_batch]
                for memory in batch:
                    # Only analyze relationships with a sample of other memories
                    # to avoid quadratic complexity
                    import random
                    candidate_sample = random.sample(
                        [m for m in memories if m.id != memory.id],
                        min(100, len(memories) - 1)  # Up to 100 candidates
                    )
                    
                    # Detect and add relationships
                    relationships_by_type = await relationship_detector.analyze_memory_relationships(
                        memory, candidate_sample
                    )
                    
                    for rel_type, rel_list in relationships_by_type.items():
                        for related_memory, strength in rel_list:
                            memory_graph.add_relationship(
                                source_id=memory.id,
                                target_id=related_memory.id,
                                rel_type=rel_type,
                                weight=strength
                            )
                            relationship_count += 1
            
            logger.info(f"Added {relationship_count} relationships to graph")
            
            # Log graph statistics
            stats = memory_graph.get_graph_stats()
            logger.info(f"Memory graph initialization complete. Stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize graph from existing memories: {e}")
            return False