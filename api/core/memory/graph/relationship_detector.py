# relationship_detector.py

import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from functools import lru_cache

from api.core.memory.models import Memory, MemoryType
from api.utils.llm_service import LLMService
from api.utils.rate_limiter import openai_limiter

logger = logging.getLogger(__name__)

class MemoryRelationshipDetector:
    """
    Detects relationships between memories using vector similarity, 
    temporal analysis, and content analysis.
    
    Optimized implementation with:
    - Increased batch sizes
    - Parallel processing
    - Caching
    - Vectorized operations
    - Batched LLM calls
    - Memory type prioritization
    """
    
    # Define relationship types
    REL_SEMANTIC = "semantic_similarity"
    REL_TEMPORAL = "temporal_sequence"
    REL_THEMATIC = "thematic_association"
    REL_CAUSAL = "causal_relationship"
    REL_HIERARCHICAL = "hierarchical_relationship"  # parent/child concepts
    REL_ELABORATION = "elaboration"  # one memory expands on another
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.semantic_similarity_threshold = 0.65
        self.temporal_proximity_threshold = timedelta(minutes=30)
        self.max_relationships_per_memory = 10
        
        # Cache for similarity scores
        self.similarity_cache = {}  # (memory_id1, memory_id2) -> similarity_score
        
        # Maximum number of LLM calls per batch
        self.max_llm_batch_size = 5
        
        # Batch size for vector operations
        self.vector_batch_size = 100  # Increased from 10
    
    async def detect_semantic_relationships(self, 
                                     memory: Memory, 
                                     candidate_memories: List[Memory]) -> List[Tuple[Memory, float]]:
        """
        Detect semantic relationships based on vector similarity.
        Optimized with vectorized operations and caching.
        
        Args:
            memory: Target memory to find relationships for
            candidate_memories: List of memories to compare against
            
        Returns:
            List of (related_memory, similarity_score) tuples
        """
        # Skip if no vector for target memory
        if not memory.semantic_vector:
            self.logger.warning(f"No semantic vector for memory {memory.id}")
            return []
        
        relationships = []
        processed_ids = set()  # Track processed memory IDs
        
        # Get the memory's vector and reshape for sklearn
        memory_vector = np.array(memory.semantic_vector).reshape(1, -1)
        
        # Process candidates in larger batches for efficiency
        for i in range(0, len(candidate_memories), self.vector_batch_size):
            batch = candidate_memories[i:i+self.vector_batch_size]
            
            # Filter valid candidates first (vectorized approach)
            valid_candidates = []
            candidate_vectors = []
            
            for candidate in batch:
                # Skip if already processed, no vector, or same as target
                if (candidate.id in processed_ids or 
                    not candidate.semantic_vector or 
                    candidate.id == memory.id):
                    continue
                
                # Check cache first
                cache_key = (memory.id, candidate.id)
                if cache_key in self.similarity_cache:
                    similarity = self.similarity_cache[cache_key]
                    if similarity >= self.semantic_similarity_threshold:
                        relationships.append((candidate, similarity))
                    processed_ids.add(candidate.id)
                else:
                    valid_candidates.append(candidate)
                    candidate_vectors.append(candidate.semantic_vector)
            
            # Skip if no valid candidates to process
            if not candidate_vectors:
                continue
                
            # Compute similarities in one vectorized operation
            candidate_matrix = np.array(candidate_vectors)
            similarities = cosine_similarity(memory_vector, candidate_matrix)[0]
            
            # Process results and update cache
            for idx, similarity in enumerate(similarities):
                candidate = valid_candidates[idx]
                cache_key = (memory.id, candidate.id)
                self.similarity_cache[cache_key] = float(similarity)
                
                if similarity >= self.semantic_similarity_threshold:
                    relationships.append((valid_candidates[idx], float(similarity)))
                
                processed_ids.add(candidate.id)
        
        # Sort by similarity (descending)
        relationships.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max relationships
        return relationships[:self.max_relationships_per_memory]
    
    def detect_temporal_relationships(self, 
                                     memory: Memory, 
                                     candidate_memories: List[Memory]) -> List[Tuple[Memory, float]]:
        """
        Detect temporal relationships based on creation time proximity.
        Optimized for faster processing.
        
        Args:
            memory: Target memory
            candidate_memories: List of memories to compare against
            
        Returns:
            List of (related_memory, strength) tuples
        """
        # Ensure memory has a timestamp
        if not memory.created_at:
            self.logger.warning(f"No timestamp for memory {memory.id}")
            return []
            
        relationships = []
        processed_ids = set()
        
        # Pre-filter candidates that might be within the threshold
        # This helps reduce the number of detailed calculations
        if isinstance(memory.created_at, datetime):
            min_time = memory.created_at - self.temporal_proximity_threshold
            max_time = memory.created_at + self.temporal_proximity_threshold
            
            # Only process candidates that might be within range
            potential_candidates = [
                c for c in candidate_memories 
                if (c.id not in processed_ids and
                    c.id != memory.id and
                    c.created_at and
                    min_time <= c.created_at <= max_time)
            ]
        else:
            # Fallback if timestamps aren't datetime objects
            potential_candidates = [
                c for c in candidate_memories
                if c.id not in processed_ids and c.id != memory.id and c.created_at
            ]
        
        # Process filtered candidates
        for candidate in potential_candidates:
            processed_ids.add(candidate.id)
            
            # Calculate time difference
            time_diff = abs(memory.created_at - candidate.created_at)
            
            # Check if within threshold
            if time_diff <= self.temporal_proximity_threshold:
                # Calculate strength: 1.0 (same time) to 0.5 (at threshold)
                proximity_ratio = 1.0 - (time_diff / self.temporal_proximity_threshold)
                strength = 0.5 + (proximity_ratio * 0.5)  # Scale from 0.5 to 1.0
                
                # Check chronological order (for sequential relationship)
                if memory.created_at > candidate.created_at:
                    # Candidate happened before memory
                    relationships.append((candidate, float(strength)))
        
        # Sort by strength (descending)
        relationships.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max relationships
        return relationships[:self.max_relationships_per_memory]
    
    async def detect_thematic_relationships(self,
                                 memory: Memory,
                                 candidate_memories: List[Memory],
                                 top_k: int = 5) -> List[Tuple[Memory, float]]:
        """
        Detect thematic relationships using batched LLM analysis.
        Optimized with batched LLM calls instead of individual calls.
        
        Args:
            memory: Target memory
            candidate_memories: List of memories to compare against
            top_k: Number of candidates to analyze
            
        Returns:
            List of (related_memory, strength) tuples
        """
        # No candidates to analyze
        if not candidate_memories or len(candidate_memories) == 0:
            return []
        
        # Take top-k candidates for analysis
        # If we already have semantic relationships, use those
        if isinstance(candidate_memories[0], tuple):
            # Format is already (memory, score)
            top_candidates = candidate_memories[:top_k]
            candidates = [mem for mem, _ in top_candidates]
        else:
            # First, filter candidates with basic semantic similarity
            semantic_results = await self.detect_semantic_relationships(
                memory, candidate_memories
            )
            top_candidates = semantic_results[:top_k]
            candidates = [mem for mem, _ in top_candidates]
        
        if not candidates:
            return []
            
        relationships = []
        
        # Process in batches to reduce LLM API calls
        batch_size = min(self.max_llm_batch_size, len(candidates))
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            
            try:
                # Wait for rate limiter
                delay = await self._get_rate_limit_delay()
                if delay > 0:
                    await asyncio.sleep(delay)
                await openai_limiter.consume()
                
                # Create a single batch prompt for multiple candidates
                batch_prompt = f"""Analyze the thematic connections between the following passages:

MAIN PASSAGE: {memory.content}

"""
                for j, candidate in enumerate(batch):
                    batch_prompt += f"PASSAGE {j+1}: {candidate.content}\n\n"
                
                batch_prompt += f"""For each passage (1-{len(batch)}), rate its thematic connection to the MAIN PASSAGE on a scale of 0-10.
Consider shared concepts, themes, ideas, or logical progression.
Respond with ONLY the numbers separated by commas, one rating per passage. Example: 7,4,9"""
                
                # Get LLM response
                response = await self.llm_service.generate_gpt_response_async(
                    batch_prompt, temperature=0.3
                )
                
                # Parse the ratings
                try:
                    ratings = [float(r.strip()) for r in response.strip().split(',')]
                    
                    # Match ratings with candidates if we have the right number
                    if len(ratings) == len(batch):
                        for j, rating in enumerate(ratings):
                            # Normalize rating to 0-1 scale
                            strength = min(1.0, max(0.0, rating / 10.0))
                            
                            # If meaningful connection found
                            if strength >= 0.6:  # Threshold for thematic connection
                                relationships.append((batch[j], strength))
                    else:
                        self.logger.warning(
                            f"Rating count mismatch: got {len(ratings)}, expected {len(batch)}"
                        )
                except ValueError as e:
                    self.logger.warning(f"Failed to parse LLM ratings: {response} - {e}")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing thematic connections: {e}")
        
        # Sort by strength (descending)
        relationships.sort(key=lambda x: x[1], reverse=True)
        return relationships
    
    async def _get_rate_limit_delay(self) -> float:
        """Helper method to efficiently check rate limiter delay."""
        # This would need to be implemented in your rate limiter
        # For now, return 0 as a placeholder
        return 0
    
    def prioritize_candidate_memories(self, candidates: List[Memory]) -> List[Memory]:
        """Prioritize memories by type to optimize processing."""
        # Sort candidates by type priority:
        # 1. EPISODIC (highest priority)
        # 2. LEARNED
        # 3. SEMANTIC (lowest priority)
        return sorted(
            candidates,
            key=lambda m: (
                0 if m.memory_type == MemoryType.EPISODIC else
                1 if m.memory_type == MemoryType.LEARNED else 2
            )
        )
    
    async def analyze_memory_relationships(self,
                                    memory: Memory,
                                    candidate_memories: List[Memory]) -> Dict[str, List[Tuple[Memory, float]]]:
        """
        Comprehensive analysis of a memory's relationships.
        Optimized with parallel processing and prioritization.
        
        Args:
            memory: Target memory
            candidate_memories: List of memories to compare against
            
        Returns:
            Dictionary of relationship types to lists of (memory, strength) tuples
        """
        # Initialize results dictionary
        relationships = {
            self.REL_SEMANTIC: [],
            self.REL_TEMPORAL: [],
            self.REL_THEMATIC: []
        }
        
        # Skip processing if no candidates
        if not candidate_memories:
            return relationships
        
        # Prioritize candidates for more efficient processing
        prioritized_candidates = self.prioritize_candidate_memories(candidate_memories)
        
        # Use asyncio.gather to run semantic and temporal detection in parallel
        semantic_task = asyncio.create_task(
            self.detect_semantic_relationships(memory, prioritized_candidates)
        )
        
        # Temporal doesn't need to be async, but run it concurrently
        # with the semantic task using run_in_executor
        loop = asyncio.get_event_loop()
        temporal_task = loop.run_in_executor(
            None, self.detect_temporal_relationships, memory, prioritized_candidates
        )
        
        # Wait for both tasks to complete
        semantic_relationships, temporal_relationships = await asyncio.gather(
            semantic_task, temporal_task
        )
        
        # Store results
        relationships[self.REL_SEMANTIC] = semantic_relationships
        relationships[self.REL_TEMPORAL] = temporal_relationships
        
        # Now use semantic results for thematic detection
        if semantic_relationships:
            # We already have the memories and scores from semantic relationships
            relationships[self.REL_THEMATIC] = await self.detect_thematic_relationships(
                memory, semantic_relationships, top_k=3
            )
        
        return relationships
    
    async def batch_analyze_relationships(
        self, memories: List[Memory], sample_size: int = 100
    ) -> Dict[str, Dict[str, List[Tuple[Memory, float]]]]:
        """
        Analyze relationships for multiple memories in a batch.
        Useful for initial graph population.
        
        Args:
            memories: List of memories to analyze
            sample_size: Maximum number of candidate memories to consider
            
        Returns:
            Dictionary mapping memory IDs to their relationships
        """
        if not memories:
            return {}
        
        results = {}
        
        # Use a fixed set of candidate memories for all analyses
        # This reduces redundant comparisons
        candidate_set = memories[:sample_size]
        
        # Process in batches of 10 memories
        batch_size = 10
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i+batch_size]
            
            # Create tasks for parallel processing
            tasks = []
            for memory in batch:
                # Skip analyzing candidates for relationship with themselves
                filtered_candidates = [c for c in candidate_set if c.id != memory.id]
                task = self.analyze_memory_relationships(memory, filtered_candidates)
                tasks.append((memory.id, task))
            
            # Process batch in parallel
            for memory_id, task in tasks:
                try:
                    relationships = await task
                    results[memory_id] = relationships
                except Exception as e:
                    self.logger.error(f"Error analyzing relationships for memory {memory_id}: {e}")
        
        return results
    
    async def incremental_graph_population(
        self, new_memories: List[Memory], existing_graph_memories: Set[str], sample_size: int = 100
    ) -> Dict[str, Dict[str, List[Tuple[Memory, float]]]]:
        """
        Incrementally populate graph with new memories against a sample of existing memories.
        
        Args:
            new_memories: New memories to add to the graph
            existing_graph_memories: Set of memory IDs already in the graph
            sample_size: Size of the sample to use for relationship detection
            
        Returns:
            Dictionary of relationships for the new memories
        """
        # Filter out memories already in the graph
        truly_new_memories = [m for m in new_memories if m.id not in existing_graph_memories]
        
        if not truly_new_memories:
            return {}
            
        self.logger.info(f"Processing {len(truly_new_memories)} new memories for graph population")
        
        # Get a representative sample of existing memories for comparison
        sample_candidates = [m for m in new_memories if m.id in existing_graph_memories]
        
        # If we don't have enough in the sample, we could fetch more from storage
        # ...
        
        # Use the batch analysis method with our constructed sample
        return await self.batch_analyze_relationships(
            truly_new_memories, 
            sample_size=min(sample_size, len(sample_candidates))
        )