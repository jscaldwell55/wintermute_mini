# relationship_detector.py

import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import asyncio


from api.core.memory.models import Memory, MemoryType
from api.utils.llm_service import LLMService
from api.utils.rate_limiter import openai_limiter

class MemoryRelationshipDetector:
    # ... existing code ...
    
    async def detect_relationships(self, source_memory, target_memories):
        # Wait until we have capacity before making API call
        while not await openai_limiter.consume():
            await asyncio.sleep(0.1)
            
        # Make the API call through LLM service
        # ... existing code ...

logger = logging.getLogger(__name__)

class MemoryRelationshipDetector:
    """
    Detects relationships between memories using vector similarity, 
    temporal analysis, and content analysis.
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
    
    async def detect_semantic_relationships(self, 
                                     memory: Memory, 
                                     candidate_memories: List[Memory]) -> List[Tuple[Memory, float]]:
        """
        Detect semantic relationships based on vector similarity.
        
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
        
        # Get the memory's vector and reshape for sklearn
        memory_vector = np.array(memory.semantic_vector).reshape(1, -1)
        
        # Process candidates in batches for efficiency
        batch_size = 10
        await asyncio.sleep(1) 
        for i in range(0, len(candidate_memories), batch_size):
            batch = candidate_memories[i:i+batch_size]
            
            # Create matrix of candidate vectors
            valid_candidates = []
            candidate_vectors = []
            
            for candidate in batch:
                # Skip if no vector or same as target memory
                if not candidate.semantic_vector or candidate.id == memory.id:
                    continue
                    
                valid_candidates.append(candidate)
                candidate_vectors.append(candidate.semantic_vector)
            
            if not candidate_vectors:
                continue
                
            # Compute similarities
            candidate_matrix = np.array(candidate_vectors)
            similarities = cosine_similarity(memory_vector, candidate_matrix)[0]
            
            # Find candidates above threshold
            for idx, similarity in enumerate(similarities):
                if similarity >= self.semantic_similarity_threshold:
                    relationships.append((valid_candidates[idx], float(similarity)))
        
        # Sort by similarity (descending)
        relationships.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max relationships
        return relationships[:self.max_relationships_per_memory]
    
    def detect_temporal_relationships(self, 
                                     memory: Memory, 
                                     candidate_memories: List[Memory]) -> List[Tuple[Memory, float]]:
        """
        Detect temporal relationships based on creation time proximity.
        
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
        
        for candidate in candidate_memories:
            # Skip if no timestamp or same memory
            if not candidate.created_at or candidate.id == memory.id:
                continue
                
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
        Detect thematic relationships using LLM analysis.
        Uses a sampling approach since it requires LLM calls.
        
        Args:
            memory: Target memory
            candidate_memories: List of memories to compare against
            top_k: Number of candidates to analyze
            
        Returns:
            List of (related_memory, strength) tuples
        """
        # First, filter candidates with basic semantic similarity
        semantic_candidates = await self.detect_semantic_relationships(
            memory, candidate_memories
        )
        
        # Take top-k candidates for deeper analysis
        top_candidates = semantic_candidates[:top_k]
        
        if not top_candidates:
            return []
            
        relationships = []
        
        # Analyze thematic connections with LLM
        for candidate, similarity in top_candidates:
            try:
                # Wait until we have capacity before making API call
                await openai_limiter.consume()  # Add this line here
                
                # Prepare LLM prompt
                prompt = f"""Analyze these two passages and determine if they share thematic connections:
                
                PASSAGE 1: {memory.content}
                
                PASSAGE 2: {candidate.content}
                
                On a scale from 0 to 10, how strongly are these passages thematically connected?
                Consider shared concepts, themes, or logical progression.
                Respond with only a number (0-10)."""
                
                # Get LLM response
                response = await self.llm_service.generate_gpt_response_async(prompt)
                # Extract numerical rating
                try:
                    rating = float(response.strip())
                    # Normalize rating to 0-1 scale
                    strength = min(1.0, max(0.0, rating / 10.0))
                    
                    # If meaningful connection found
                    if strength >= 0.6:  # Threshold for thematic connection
                        relationships.append((candidate, strength))
                except ValueError:
                    self.logger.warning(f"Failed to parse LLM rating: {response}")
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error analyzing thematic connection: {e}")
                continue
        
        # Sort by strength (descending)
        relationships.sort(key=lambda x: x[1], reverse=True)
        return relationships
    
    async def analyze_memory_relationships(self,
                                    memory: Memory,
                                    candidate_memories: List[Memory]) -> Dict[str, List[Tuple[Memory, float]]]:
        """
        Comprehensive analysis of a memory's relationships.
        
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
        
        # Detect semantic relationships
        relationships[self.REL_SEMANTIC] = await self.detect_semantic_relationships(
            memory, candidate_memories
        )
        
        # Detect temporal relationships
        relationships[self.REL_TEMPORAL] = self.detect_temporal_relationships(
            memory, candidate_memories
        )
        
        # Detect thematic relationships (for a small sample)
        if len(candidate_memories) > 0:
            # Sample candidates based on semantic similarity first
            semantic_candidates = [mem for mem, _ in relationships[self.REL_SEMANTIC]]
            if semantic_candidates:
                relationships[self.REL_THEMATIC] = await self.detect_thematic_relationships(
                    memory, semantic_candidates, top_k=3
                )
        
        return relationships