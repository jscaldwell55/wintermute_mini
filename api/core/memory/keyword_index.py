import logging
from collections import defaultdict
import time
from typing import List, Dict, Set, Tuple, Optional
from api.core.memory.models import Memory, MemoryResponse

logger = logging.getLogger(__name__)

class KeywordIndex:
    """Inverted index for efficient keyword-based memory retrieval"""
    
    def __init__(self):
        self.index = defaultdict(list)  # word -> list of memory_ids
        self.memory_lookup = {}  # memory_id -> Memory object
        self.last_update = {}  # memory_id -> last update timestamp
        self.stopwords = {
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
            'about', 'like', 'of', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'and',
            'or', 'if', 'then', 'else', 'when', 'up', 'down', 'out', 'over'
        }
        logger.info("Initializing keyword index")
        
    async def build_index(self, memories: List[Memory]):
        """Build or update the keyword index with memories"""
        start_time = time.time()
        added_count = 0
        
        for memory in memories:
            if self.add_to_index(memory):
                added_count += 1
                
        logger.info(f"Built keyword index in {time.time() - start_time:.2f}s with {len(self.index)} unique terms and {added_count} new memories")
            
    def add_to_index(self, memory: Memory) -> bool:
        """
        Add a single memory to the index
        Returns True if memory was added, False if it was already indexed
        """
        if memory.id in self.memory_lookup:
            # Check if content has changed
            if self.memory_lookup[memory.id].content == memory.content:
                return False  # Already indexed with same content
            
            # Content changed, remove old index entries
            self._remove_from_index(memory.id)
            
        # Add to memory lookup
        self.memory_lookup[memory.id] = memory
        self.last_update[memory.id] = time.time()
        
        # Tokenize and index content
        words = self._tokenize(memory.content)
        for word in words:
            self.index[word].append(memory.id)
            
        return True
    
    def _remove_from_index(self, memory_id: str):
        """Remove a memory from the index"""
        if memory_id not in self.memory_lookup:
            return
            
        # Get the memory content
        memory = self.memory_lookup[memory_id]
        words = self._tokenize(memory.content)
        
        # Remove from index
        for word in words:
            if memory_id in self.index[word]:
                self.index[word].remove(memory_id)
                
        # Remove from memory lookup
        del self.memory_lookup[memory_id]
        if memory_id in self.last_update:
            del self.last_update[memory_id]
    
    async def search(self, keywords: List[str], limit: int = 20, memory_type: Optional[str] = None, window_id: Optional[str] = None) -> List[Tuple[Memory, float]]:
        """
        Search for keywords and return scored memories
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results to return
            memory_type: Optional filter by memory type
            window_id: Optional filter by window ID
            
        Returns:
            List of (Memory, score) tuples, sorted by descending score
        """
        start_time = time.time()
        memory_scores = defaultdict(float)
        
        # For each keyword, find matching memories and add scores
        for keyword in keywords:
            keyword_lower = keyword.lower()
            matching_memory_ids = self.index.get(keyword_lower, [])
            for memory_id in matching_memory_ids:
                # Apply filters
                memory = self.memory_lookup.get(memory_id)
                if not memory:
                    continue
                    
                # Apply memory_type filter if specified
                if memory_type and memory.memory_type.value != memory_type:
                    continue
                    
                # Apply window_id filter if specified
                if window_id and memory.window_id != window_id:
                    continue
                    
                # Calculate score based on keyword frequency and position
                base_score = 1.0
                
                # Boost exact matches in title/first sentence
                content_lower = memory.content.lower()
                first_100_chars = content_lower[:100]
                if keyword_lower in first_100_chars:
                    base_score *= 1.5
                    
                # Add score for this keyword
                memory_scores[memory_id] += base_score
        
        # Normalize scores by number of keywords for more balanced results
        if keywords:
            for memory_id in memory_scores:
                # Score between 0-1 based on how many keywords matched
                memory_scores[memory_id] = memory_scores[memory_id] / len(keywords)
        
        # Get top results
        top_memory_ids = sorted(memory_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:limit]
        
        # Convert to memory objects with scores
        results = [(self.memory_lookup[mid], score) for mid, score in top_memory_ids]
        
        logger.info(f"Keyword search completed in {time.time() - start_time:.4f}s, found {len(results)} matches for {keywords}")
        return results
        
    def _tokenize(self, text: str) -> Set[str]:
        """Extract unique tokens from text"""
        # Normalize and split
        tokens = text.lower().split()
        
        # Remove stopwords, punctuation, short words
        tokens = [self._clean_token(t) for t in tokens if t not in self.stopwords and len(t) > 2]
        tokens = [t for t in tokens if t]  # Remove empty tokens
        
        return set(tokens)  # Return unique tokens only
    
    def _clean_token(self, token: str) -> str:
        """Clean a token by removing punctuation"""
        # Remove common punctuation at start/end
        token = token.strip('.,;:!?"\'()[]{}')
        
        # Remove common non-alphanumeric characters
        if not token.isalnum() and len(token) > 2:
            # Keep hyphenated words but remove other symbols
            token = ''.join(c if c.isalnum() or c == '-' else ' ' for c in token)
            token = token.strip()
            
        return token