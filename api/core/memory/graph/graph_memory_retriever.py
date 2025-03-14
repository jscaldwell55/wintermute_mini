# graph_memory_retriever.py

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import math
from datetime import datetime, timezone, timedelta


from api.core.memory.models import Memory, MemoryType, QueryRequest, QueryResponse, MemoryResponse
from api.core.memory.graph.memory_graph import MemoryGraph
from api.utils.pinecone_service import PineconeService
from api.core.vector.vector_operations import VectorOperationsImpl
from api.utils.utils import normalize_timestamp
from api.utils.config import get_settings

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
        self.settings = get_settings()
        
        # Configuration
        self.max_graph_depth = 2  # Maximum hops in graph traversal
        self.max_memories_per_hop = 3  # Max memories to retrieve per hop
        self.association_score_decay = 0.7  # Score decay per hop
        self.min_association_score = 0.3  # Minimum score to include an associated memory
        
        # Weighting for different retrieval methods
        self.vector_weight = 0.7
        self.graph_weight = 0.3

    def _calculate_bell_curve_recency(self, age_hours):
        """
        Calculate recency score using a bell curve pattern:
        - Very recent memories (<1h) are heavily de-prioritized (0.2-0.4)
        - Memories <24h old have gradually increasing priority (0.4-0.8)
        - Peak priority is around 2-3 days old (0.8-1.0)
        - Gradual decay for older memories (approaching 0.2 at 7 days)
        
        Args:
            age_hours: Age of memory in hours
            
        Returns:
            Recency score between 0 and 1
        """
        # Get settings parameters (or use defaults if not defined)
        peak_hours = getattr(self.settings, 'episodic_peak_hours', 60)
        very_recent_threshold = getattr(self.settings, 'episodic_very_recent_threshold', 1.0)
        recent_threshold = getattr(self.settings, 'episodic_recent_threshold', 24.0)
        steepness = getattr(self.settings, 'episodic_bell_curve_steepness', 2.5)
        
        # For very recent memories (<1h): start with a low score
        if age_hours < very_recent_threshold:
            # Map from 0.2 (at 0 hours) to 0.4 (at 1 hour)
            return 0.2 + (0.2 * age_hours / very_recent_threshold)
        
        # For recent memories (1h-24h): gradual increase
        elif age_hours < recent_threshold:
            # Map from 0.4 (at 1 hour) to 0.8 (at 24 hours)
            relative_position = (age_hours - very_recent_threshold) / (recent_threshold - very_recent_threshold)
            return 0.4 + (0.4 * relative_position)
        
        # Bell curve peak and decay
        else:
            # Calculate distance from peak (in hours)
            distance_from_peak = abs(age_hours - peak_hours)
            
            # Convert to a bell curve shape (Gaussian-inspired)
            # Distance of 0 from peak = 1.0 score
            # Maximum reasonable age for scoring is settings.episodic_max_age_days * 24
            max_distance = getattr(self.settings, 'episodic_max_age_days', 7) * 24 - peak_hours
            
            # Normalized distance from peak (0-1)
            normalized_distance = min(1.0, distance_from_peak / max_distance)
            
            # Apply bell curve formula (variant of Gaussian)
            bell_value = math.exp(-(normalized_distance ** 2) * steepness)
            
            # Scale between 0.8 (peak) and 0.2 (oldest)
            return 0.8 * bell_value + 0.2

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
        if combined_matches and combined_scores:
            sorted_indices = np.argsort(combined_scores)[::-1]  # Descending order
            top_k_indices = sorted_indices[:request.top_k]
            
            top_matches = [combined_matches[i] for i in top_k_indices]
            top_scores = [combined_scores[i] for i in top_k_indices]
        else:
            top_matches = []
            top_scores = []
        
        self.logger.info(f"Final combined retrieval returned {len(top_matches)} matches")
        
        return QueryResponse(
            matches=top_matches,
            similarity_scores=top_scores
        )
    
    def _ensure_string_timestamp(self, timestamp_value) -> str:
        """Helper method to ensure timestamp is in proper string format"""
        if timestamp_value is None:
            # Default to current time if missing
            return datetime.now(timezone.utc).isoformat() + "Z"
        
        if isinstance(timestamp_value, datetime):
            return timestamp_value.isoformat() + "Z"
        
        if isinstance(timestamp_value, (int, float)):
            # If it's a Unix timestamp
            dt = datetime.fromtimestamp(timestamp_value, timezone.utc)
            return dt.isoformat() + "Z"
        
        # Already a string, normalize it
        return normalize_timestamp(timestamp_value)
    
    async def _retrieve_vector_memories(self, request: QueryRequest) -> QueryResponse:
        """
        Retrieve memories using vector similarity (using existing system).
        This is a placeholder that would call your existing memory retrieval system.
        """
        try:
            # Create query vector
            query_vector = await self.vector_operations.create_semantic_vector(request.prompt)
            
            # Prepare filter
            filter_dict = {}
            if request.memory_type:
                filter_dict["memory_type"] = request.memory_type.value
                
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
                    # Handle the created_at field properly
                    created_at_str = self._ensure_string_timestamp(memory_data["metadata"].get("created_at"))
                    
                    memory_response = MemoryResponse(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                        created_at=created_at_str,
                        metadata=memory_data["metadata"],
                        window_id=memory_data["metadata"].get("window_id")
                    )
                    matches.append(memory_response)
                    scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error processing memory result: {e}")
                    continue

            # Apply bell curve scoring to memories
            current_time = datetime.now(timezone.utc)
            
            for i, memory in enumerate(matches):
                try:
                    # Only apply bell curve to episodic memories
                    if memory.memory_type == MemoryType.EPISODIC:
                        # Get created_at timestamp
                        created_at = datetime.fromisoformat(memory.created_at.rstrip('Z'))
                        
                        # Calculate age in hours
                        age_hours = (current_time - created_at).total_seconds() / (60*60)
                        
                        # Use bell curve recency scoring
                        recency_score = self._calculate_bell_curve_recency(age_hours)
                        
                        # Adjust score with recency
                        recency_weight = getattr(self.settings, 'episodic_recency_weight', 0.35)
                        relevance_weight = 1 - recency_weight
                        
                        # Combine score components
                        combined_score = (relevance_weight * scores[i]) + (recency_weight * recency_score)
                        
                        # Apply memory type weight
                        memory_weight = getattr(self.settings, 'episodic_memory_weight', 0.40)
                        adjusted_score = combined_score * memory_weight
                        
                        self.logger.info(f"Applied bell curve: Memory {memory.id} (EPISODIC): " 
                                        f"Age={age_hours:.1f}h, Recency={recency_score:.3f}, "
                                        f"Original={scores[i]:.3f}, Adjusted={adjusted_score:.3f}")
                        
                        # Update score
                        scores[i] = adjusted_score
                except Exception as e:
                    self.logger.error(f"Error applying bell curve to memory {memory.id}: {e}")
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
                if i > 0:  # Only discount if not the top result
                    entry_point_score = 0.9  # Slightly discount if not top result
                
                # Get connected memories from graph
                connected_memories = self.memory_graph.get_connected_memories(
                    memory.id, 
                    max_depth=self.max_graph_depth,
                    max_results=self.max_memories_per_hop
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
                        # Ensure created_at is a properly formatted string
                        created_at_str = self._ensure_string_timestamp(memory_data.get("created_at"))
                        
                        # Create MemoryResponse
                        memory_response = MemoryResponse(
                            id=memory_data["id"],
                            content=memory_data["content"],
                            memory_type=memory_data["memory_type"],
                            created_at=created_at_str,
                            metadata=memory_data.get("metadata", {}),
                            window_id=memory_data.get("window_id")
                        )
                        
                        # Apply bell curve scoring if this is an episodic memory
                        if memory_data["memory_type"] == MemoryType.EPISODIC.value:
                            current_time = datetime.now(timezone.utc)
                            created_at = datetime.fromisoformat(created_at_str.rstrip('Z'))
                            
                            # Calculate age in hours
                            age_hours = (current_time - created_at).total_seconds() / (60*60)
                            
                            # Use bell curve recency scoring
                            recency_score = self._calculate_bell_curve_recency(age_hours)
                            
                            # Adjust score with recency
                            recency_weight = getattr(self.settings, 'episodic_recency_weight', 0.35)
                            relevance_weight = 1 - recency_weight
                            
                            # Combine score components
                            combined_score = (relevance_weight * final_score) + (recency_weight * recency_score)
                            
                            # Apply memory type weight
                            memory_weight = getattr(self.settings, 'episodic_memory_weight', 0.40)
                            final_score = combined_score * memory_weight
                            
                            self.logger.info(f"Applied bell curve to associated memory {memory_data['id']}: " 
                                            f"Age={age_hours:.1f}h, Recency={recency_score:.3f}, "
                                            f"Adjusted Score={final_score:.3f}")
                        
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