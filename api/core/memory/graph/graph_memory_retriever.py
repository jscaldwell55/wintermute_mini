# graph_memory_retriever.py

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from datetime import datetime

from api.core.memory.models import Memory, MemoryType, QueryRequest, QueryResponse, MemoryResponse
from api.core.memory.graph.memory_graph import MemoryGraph
from api.utils.pinecone_service import PineconeService
from api.core.vector.vector_operations import VectorOperationsImpl

logger = logging.getLogger(__name__)

class GraphMemoryRetriever:
    """
    Enhanced memory retrieval using graph-based associative memory.
    Combines vector similarity search with graph traversal to find
    relevant memories through direct and indirect relationships.
    """
    
    def __init__(
        self,
        memory_graph: MemoryGraph,
        pinecone_service: PineconeService,
        vector_operations: VectorOperationsImpl
    ):
        self.memory_graph = memory_graph
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_graph_depth = 2  # Maximum hops in graph traversal
        self.max_memories_per_hop = 3  # Max memories to retrieve per hop
        self.association_score_decay = 0.7  # Score decay per hop
        self.min_association_score = 0.3  # Minimum score to include an associated memory
        
        # Weighting for different retrieval methods
        self.vector_weight = 0.7
        self.graph_weight = 0.3

    async def retrieve_memories(self, request: QueryRequest) -> QueryResponse:
        """
        Retrieve memories using both vector similarity and graph associations.
        
        Args:
            request: Query request containing the prompt and other parameters
            
        Returns:
            QueryResponse with combined vector and graph-based matches
        """
        self.logger.info(f"Retrieving memories for query: {request.prompt[:50]}...")
        
        # 1. First, retrieve memories using vector similarity (existing approach)
        vector_results = await self._retrieve_vector_memories(request)
        self.logger.info(f"Vector retrieval returned {len(vector_results.matches)} matches")
        
        # If graph is empty, just return vector results
        if self.memory_graph.graph.number_of_nodes() == 0:
            self.logger.info("Memory graph is empty. Using vector-only retrieval.")
            return vector_results
        
        # 2. Use vector-retrieved memories as entry points for graph traversal
        top_vector_memories = vector_results.matches[:min(5, len(vector_results.matches))]
        graph_matches, graph_scores = await self._retrieve_graph_associations(
            top_vector_memories, request
        )
        self.logger.info(f"Graph association retrieval returned {len(graph_matches)} additional matches")
        
        # 3. Combine and deduplicate results
        combined_matches, combined_scores = self._combine_results(
            vector_results.matches, 
            vector_results.similarity_scores,
            graph_matches,
            graph_scores
        )
        
        # 4. Sort by combined score and limit to requested top_k
        sorted_indices = np.argsort(combined_scores)[::-1]  # Descending order
        top_k_indices = sorted_indices[:request.top_k]
        
        top_matches = [combined_matches[i] for i in top_k_indices]
        top_scores = [combined_scores[i] for i in top_k_indices]
        
        self.logger.info(f"Final combined retrieval returned {len(top_matches)} matches")
        
        return QueryResponse(
            matches=top_matches,
            similarity_scores=top_scores
        )
    
    async def _retrieve_vector_memories(self, request: QueryRequest) -> QueryResponse:
        """
        Retrieve memories using vector similarity (using existing system).
        This is a placeholder that would call your existing memory retrieval system.
        """
        # In production, this would call your existing memory system's query_memories method
        # For now, we're implementing a simplified version
        
        try:
            # Create query vector
            query_vector = await self.vector_operations.create_semantic_vector(request.prompt)
            
            # Prepare filter
            filter_dict = {}
            if request.memory_type:
                filter_dict["memory_type"] = request.memory_type.value
            if request.window_id:
                filter_dict["window_id"] = request.window_id
                
            # Query Pinecone
            results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=request.top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Convert to MemoryResponse objects
            matches = []
            scores = []
            
            for memory_data, score in results:
                try:
                    memory_response = MemoryResponse(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                        created_at=memory_data["metadata"]["created_at"].isoformat() + "Z" if isinstance(memory_data["metadata"]["created_at"], datetime) else memory_data["metadata"]["created_at"],
                        metadata=memory_data["metadata"],
                        window_id=memory_data["metadata"].get("window_id")
                    )
                    matches.append(memory_response)
                    scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error processing memory result: {e}")
                    continue
            
            return QueryResponse(
                matches=matches,
                similarity_scores=scores
            )
            
        except Exception as e:
            self.logger.error(f"Error in vector memory retrieval: {e}")
            return QueryResponse(matches=[], similarity_scores=[])
    
    async def _retrieve_graph_associations(
        self, 
        entry_point_memories: List[MemoryResponse],
        request: QueryRequest
    ) -> Tuple[List[MemoryResponse], List[float]]:
        """
        Retrieve associated memories through graph traversal.
        
        Args:
            entry_point_memories: Initial memories to start graph traversal from
            request: Original query request
            
        Returns:
            Tuple of (associated_memories, association_scores)
        """
        if not entry_point_memories:
            return [], []
            
        associated_memories = []
        association_scores = []
        visited_ids = set()  # Track visited memory IDs
        
        # Add entry point memory IDs to visited set
        for memory in entry_point_memories:
            visited_ids.add(memory.id)
        
        # For each entry point memory, find associated memories in the graph
        for i, memory in enumerate(entry_point_memories):
            try:
                # Get entry point vector score (for weighting later)
                entry_point_score = 1.0
                if i < len(entry_point_memories):
                    entry_point_score = 0.9  # Slightly discount if not top result
                
                # Get connected memories from graph
                connected_memories = self.memory_graph.get_connected_memories(
                    memory.id, 
                    max_depth=self.max_graph_depth
                )
                
                for connected_id, relevance, depth in connected_memories:
                    # Skip if already visited
                    if connected_id in visited_ids:
                        continue
                        
                    visited_ids.add(connected_id)
                    
                    # Apply score decay based on hop distance
                    decayed_score = relevance * (self.association_score_decay ** (depth - 1))
                    
                    # Weight by entry point's relevance
                    final_score = decayed_score * entry_point_score
                    
                    # Skip if score too low
                    if final_score < self.min_association_score:
                        continue
                    
                    # Get memory details from Pinecone
                    memory_data = await self.pinecone_service.get_memory_by_id(connected_id)
                    if not memory_data:
                        continue
                    
                    try:
                        # Create MemoryResponse
                        memory_response = MemoryResponse(
                            id=memory_data["id"],
                            content=memory_data["content"],
                            memory_type=memory_data["memory_type"],
                            created_at=memory_data["created_at"],
                            metadata=memory_data.get("metadata", {}),
                            window_id=memory_data.get("window_id")
                        )
                        
                        associated_memories.append(memory_response)
                        association_scores.append(float(final_score))
                        
                        self.logger.info(f"Added associated memory {memory_data['id']} with score {final_score:.3f} (depth {depth})")
                        
                    except Exception as e:
                        self.logger.error(f"Error creating MemoryResponse from graph result: {e}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error processing graph traversal for memory {memory.id}: {e}")
                continue
        
        return associated_memories, association_scores
    
    def _combine_results(
        self,
        vector_matches: List[MemoryResponse],
        vector_scores: List[float],
        graph_matches: List[MemoryResponse],
        graph_scores: List[float]
    ) -> Tuple[List[MemoryResponse], List[float]]:
        """
        Combine and deduplicate vector and graph results.
        
        Args:
            vector_matches: Memories retrieved via vector similarity
            vector_scores: Scores for vector-retrieved memories
            graph_matches: Memories retrieved via graph traversal
            graph_scores: Scores for graph-retrieved memories
            
        Returns:
            Tuple of (combined_matches, combined_scores)
        """
        # Create dictionaries for faster lookup
        vector_dict = {match.id: (match, score) for match, score in zip(vector_matches, vector_scores)}
        graph_dict = {match.id: (match, score) for match, score in zip(graph_matches, graph_scores)}
        
        # Combine dictionaries
        combined_dict = {}
        
        # Add vector results with vector weight
        for memory_id, (memory, score) in vector_dict.items():
            combined_dict[memory_id] = (memory, score * self.vector_weight)
        
        # Add or update with graph results
        for memory_id, (memory, score) in graph_dict.items():
            if memory_id in combined_dict:
                # If memory exists in both, use weighted average
                existing_memory, existing_score = combined_dict[memory_id]
                combined_score = existing_score + (score * self.graph_weight)
                combined_dict[memory_id] = (existing_memory, combined_score)
            else:
                # If only in graph results, add with graph weight
                combined_dict[memory_id] = (memory, score * self.graph_weight)
        
        # Convert back to lists
        combined_matches = []
        combined_scores = []
        
        for memory_id, (memory, score) in combined_dict.items():
            combined_matches.append(memory)
            combined_scores.append(score)
        
        return combined_matches, combined_scores
    
    async def find_paths_between_memories(
        self,
        source_memory_id: str,
        target_memory_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find paths connecting two memories in the graph.
        Useful for explaining why a particular memory was retrieved.
        
        Args:
            source_memory_id: Starting memory ID
            target_memory_id: Target memory ID
            
        Returns:
            List of path descriptions with node and edge details
        """
        # Get path from graph
        path_node_ids = self.memory_graph.find_path_between_memories(
            source_memory_id, target_memory_id
        )
        
        if not path_node_ids or len(path_node_ids) < 2:
            return []
            
        paths = []
        
        # Convert path to detailed description
        for i in range(len(path_node_ids) - 1):
            source_id = path_node_ids[i]
            target_id = path_node_ids[i+1]
            
            # Get edge data
            if self.memory_graph.graph.has_edge(source_id, target_id):
                edge_data = self.memory_graph.graph.edges[source_id, target_id]
                
                # Get node data
                source_data = self.memory_graph.graph.nodes[source_id]
                target_data = self.memory_graph.graph.nodes[target_id]
                
                # Create path segment description
                path_segment = {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_type": source_data.get("memory_type", "UNKNOWN"),
                    "target_type": target_data.get("memory_type", "UNKNOWN"),
                    "relationship_type": edge_data.get("type", "UNKNOWN"),
                    "relationship_strength": edge_data.get("weight", 0.0),
                    "source_content": source_data.get("content", "")[:100] + "...",  # Truncate for readability
                    "target_content": target_data.get("content", "")[:100] + "..."
                }
                
                paths.append(path_segment)
        
        return paths