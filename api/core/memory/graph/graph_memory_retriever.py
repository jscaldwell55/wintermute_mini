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
            return datetime.now(timezone.utc).isoformat() 
        
        if isinstance(timestamp_value, datetime):
            return timestamp_value.isoformat() 
        
        if isinstance(timestamp_value, (int, float)):
            # If it's a Unix timestamp
            dt = datetime.fromtimestamp(timestamp_value, timezone.utc)
            return dt.isoformat() 
        
        # Already a string, normalize it
        if isinstance(timestamp_value, str):
            normalized_value = normalize_timestamp(timestamp_value)
            if isinstance(normalized_value, str):
                return normalized_value
            else:
                # If normalize_timestamp returned a datetime
                return normalized_value.isoformat()
        
        return str(timestamp_value)
    
    async def _retrieve_vector_memories(self, request: QueryRequest) -> QueryResponse:
        is_time_query = any(x in request.prompt.lower() for x in ["this morning", "today", "yesterday", "last week"])
    
        if is_time_query:
            # Add additional fields to retrieve and log
            logger.info(f"Time-specific query detected: '{request.prompt}'")
            # Use more generous top_k for time queries
            original_top_k = request.top_k
            request.top_k = request.top_k * 2 
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
            try:
                results = await self.pinecone_service.query_memories(
                    query_vector=query_vector,
                    top_k=request.top_k,
                    filter=filter_dict,
                    include_metadata=True
                )
                
                # Add check for empty results
                if not results:
                    self.logger.warning("No results returned from Pinecone query")
                    return QueryResponse(matches=[], similarity_scores=[])
                
                # Convert to MemoryResponse objects
                matches = []
                scores = []
                
                # Skip processing if no results returned
                if len(results) == 0:
                    self.logger.info("Pinecone returned empty results list")
                    return QueryResponse(matches=[], similarity_scores=[])
                    
            except Exception as e:
                self.logger.error(f"Error querying Pinecone: {e}")
                return QueryResponse(matches=[], similarity_scores=[])
                        
            # Convert to MemoryResponse objects
            matches = []
            scores = []
            
            for memory_data, score in results:
                try:
                    # Handle the created_at field properly
                    created_at_str = self._ensure_string_timestamp(memory_data["metadata"].get("created_at"))
                    
                    memory_response = MemoryResponse(
                        id=memory_data["id"],
                        # Try to access content directly, then in metadata
                        content=memory_data.get("content", memory_data.get("metadata", {}).get("content", "")),
                        # Similar pattern for memory_type
                        memory_type=memory_data.get("memory_type", 
                                        memory_data.get("metadata", {}).get("memory_type", MemoryType.EPISODIC)),
                        created_at=created_at_str,
                        metadata=memory_data.get("metadata", {}),
                        window_id=memory_data.get("window_id", memory_data.get("metadata", {}).get("window_id"))
                    )
                    matches.append(memory_response)
                    scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error processing memory result: {e}")
                    self.logger.info(f"Querying with filter: {filter_dict}")
                    self.logger.info(f"Raw query results: {results[:3]}")
                    self.logger.info(f"Graph stats: {self.memory_graph.get_graph_stats()}")
                    continue

            # Apply bell curve scoring to memories
            current_time = datetime.now(timezone.utc)
            
            for i, memory in enumerate(matches):
                try:
                    # Only apply bell curve to episodic memories
                    if memory.memory_type == MemoryType.EPISODIC:
                        # Calculate age in hours
                        if isinstance(memory.created_at, str):
                            created_at = datetime.fromisoformat(memory.created_at.rstrip('Z'))
                            if created_at.tzinfo is None:
                                created_at = created_at.replace(tzinfo=timezone.utc)
                        else:
                            created_at = memory.created_at
                            if created_at.tzinfo is None:
                                created_at = created_at.replace(tzinfo=timezone.utc)
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
        Retrieve associated memories through graph traversal using Dijkstra's algorithm,
        prioritizing temporal relationships for time-related queries.
        """
        if not entry_point_memories:
            return [], []

        missing_nodes = [memory.id for memory in entry_point_memories
                        if memory.id not in self.memory_graph.graph.nodes]
        if missing_nodes:
            await self.repair_missing_nodes(missing_nodes)

        associated_memories = []
        association_scores = []
        visited_ids = set()  # Track visited memory IDs

        # Add entry point memory IDs to visited set
        for memory in entry_point_memories:
            visited_ids.add(memory.id)

        # **Temporal Query Detection Logic (Phase 1 Enhancement):**
        is_temporal_query = any(x in request.prompt.lower() for x in ["when", "yesterday", "last week", "this morning", "today", "past"]) # Add more keywords as needed
        self.logger.info(f"Is temporal query: {is_temporal_query} for query: {request.prompt[:50]}...")


        # For each entry point memory, find associated memories using Dijkstra's
        for i, memory in enumerate(entry_point_memories):
            try:
                # Get entry point vector score (for weighting later)
                entry_point_score = 1.0
                if i > 0:  # Only discount if not the top result
                    entry_point_score = 0.9  # Slightly discount if not top result

                # Use Dijkstra's pathfinding to get best paths, pass the temporal query flag
                best_paths = self.memory_graph.find_best_paths(
                    memory.id,
                    max_depth=self.max_graph_depth,
                    is_temporal_query=is_temporal_query # Pass the flag here
                )

                for connected_id, path_cost, path_node_ids in best_paths[:self.max_memories_per_hop]: # Limit results
                    # Skip if already visited (keep this)
                    if connected_id in visited_ids:
                        continue
                    visited_ids.add(connected_id)

                    # Calculate score - now based on path_cost (lower cost = better)
                    # You'll need to experiment with how to convert path_cost to a relevance score
                    # A simple approach:  Invert and normalize path_cost
                    if path_cost == 0: # Handle direct connection case
                        path_relevance = 1.0
                    else:
                        path_relevance = 1.0 / (1 + path_cost) # Normalize and invert (adjust normalization as needed)

                    final_score = path_relevance * entry_point_score  # Weight with entry point score

                    if final_score < self.min_association_score:
                        continue

                    # Get memory details from Pinecone
                    memory_data = await self.pinecone_service.get_memory_by_id(connected_id)
                    if not memory_data:
                        continue

                    try:
                        # Debug the memory data structure
                        self.logger.debug(f"Memory data structure: {memory_data.keys()}")

                        # Initialize variables with defaults
                        content = ""
                        memory_type_str = "EPISODIC"  # Default memory type
                        window_id = None

                        # Try to extract from metadata first
                        if "metadata" in memory_data and isinstance(memory_data["metadata"], dict):
                            metadata = memory_data["metadata"]

                            # Get content
                            if "content" in metadata:
                                content = metadata["content"]

                            # Get memory_type
                            if "memory_type" in metadata:
                                memory_type_str = metadata["memory_type"]

                            # Get window_id
                            if "window_id" in metadata:
                                window_id = metadata["window_id"]

                        # Fall back to direct access if not found in metadata
                        if not content and "content" in memory_data:
                            content = memory_data["content"]

                        if memory_type_str == "EPISODIC" and "memory_type" in memory_data:
                            memory_type_str = memory_data["memory_type"]

                        if not window_id and "window_id" in memory_data:
                            window_id = memory_data["window_id"]

                        # Ensure created_at is a properly formatted string
                        created_at_str = self._ensure_string_timestamp(
                            memory_data.get("created_at") or memory_data.get("metadata", {}).get("created_at")
                        )

                        # Make sure memory_type_str is a valid enum value
                        if memory_type_str not in [t.value for t in MemoryType]:
                            self.logger.warning(f"Invalid memory_type: {memory_type_str}, defaulting to EPISODIC")
                            memory_type_str = "EPISODIC"

                        # Create MemoryResponse with extracted fields
                        memory_response = MemoryResponse(
                            id=memory_data["id"],
                            content=content,
                            memory_type=MemoryType(memory_type_str),
                            created_at=created_at_str,
                            metadata=memory_data.get("metadata", {}),
                            window_id=window_id
                        )

                        # Apply bell curve scoring if this is an episodic memory
                        memory_type_str = memory_data.get("metadata", {}).get("memory_type", "UNKNOWN")
                        if memory_type_str == "EPISODIC":
                            current_time = datetime.now(timezone.utc)
                            if isinstance(created_at_str, str):
                                created_at = datetime.fromisoformat(created_at_str.rstrip('Z'))
                                if created_at.tzinfo is None:
                                    created_at = created_at.replace(tzinfo=timezone.utc)
                            else:
                                created_at = created_at_str
                                if created_at.tzinfo is None:
                                    created_at = created_at.replace(tzinfo=timezone.utc)

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
                        self.logger.info(f"Added associated memory {memory_data['id']} with score {final_score:.3f} (path cost {path_cost:.3f})")


                    except Exception as e:
                        self.logger.error(f"Error creating MemoryResponse from graph result: {e}")
                        continue

            except Exception as e:
                self.logger.error(f"Error processing graph traversal (Dijkstra's) for memory {memory.id}: {e}")
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
    
    async def repair_missing_nodes(self, memory_ids):
        """Add missing nodes to the graph from Pinecone."""
        from api.core.memory.models import Memory, MemoryType  # Import here to avoid circular imports
        from datetime import datetime, timezone
        
        for memory_id in memory_ids:
            # Check if node exists in the graph.nodes collection
            if memory_id not in self.memory_graph.graph.nodes:
                # Fetch from Pinecone
                memory_data = await self.pinecone_service.get_memory_by_id(memory_id)
                if memory_data:
                    try:
                        # Get created_at value, with fallbacks
                        created_at = None
                        metadata = memory_data.get("metadata", {})
                        
                        # Try different timestamp fields in order of preference
                        if "created_at_iso" in metadata:
                            created_at = metadata["created_at_iso"]
                        elif "created_at" in metadata:
                            created_at = metadata["created_at"]
                        elif "created_at_unix" in metadata:
                            # Convert Unix timestamp to ISO string
                            unix_ts = metadata["created_at_unix"]
                            created_at = datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat() + "Z"
                        else:
                            # Use current time as fallback
                            created_at = datetime.now(timezone.utc).isoformat() + "Z"
                        
                        # Create Memory object with proper created_at
                        memory = Memory(
                            id=memory_data["id"],
                            content=metadata.get("content", ""),
                            memory_type=MemoryType(metadata.get("memory_type", "EPISODIC")),
                            created_at=created_at,  # Now this should be valid
                            metadata=metadata,
                            window_id=metadata.get("window_id"),
                            semantic_vector=memory_data.get("vector")
                        )
                        
                        # Add to graph
                        self.memory_graph.add_memory_node(memory)
                        logger.info(f"Repaired missing node: {memory_id}")
                        
                    except Exception as e:
                        logger.error(f"Error converting memory data to Memory object: {e}")
                        # As a fallback, add the node directly to graph
                        self.memory_graph.graph.add_node(memory_id, **memory_data.get("metadata", {}))
                        logger.info(f"Added node {memory_id} directly to graph as fallback")