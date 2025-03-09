# api/utils/response_cache.py

import time
import logging
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio

logger = logging.getLogger(__name__)

class ResponseCache:
    """Cache for LLM responses to reduce latency for duplicate or similar queries."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, similarity_threshold: float = 0.92):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of entries in the cache
            ttl_seconds: Time to live for cache entries in seconds
            similarity_threshold: Threshold for considering cached responses (0.0-1.0)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.lock = asyncio.Lock()
        self.hit_count = 0
        self.miss_count = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # Run cleanup every 5 minutes
        
        logger.info(f"Response cache initialized with max_size={max_size}, ttl={ttl_seconds}s")
    
    def _hash_query(self, query: str) -> str:
        """Generate a hash for the query string."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    async def get(self, query: str, window_id: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """
        Get a cached response for the query.
        
        Args:
            query: The query string
            window_id: Optional window ID for context-specific caching
            
        Returns:
            Tuple of (cached_response, similarity_score) or None if no match
        """
        try:
            # Check if cleanup is needed
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval:
                await self._cleanup_expired()
            
            query_hash = self._hash_query(query)
            query_key = f"{query_hash}:{window_id}" if window_id else query_hash
            
            exact_match = None
            best_match = None
            best_score = 0.0
            
            async with self.lock:
                # First check for exact match
                if query_key in self.cache:
                    entry = self.cache[query_key]
                    if current_time < entry["expires_at"]:
                        # Update access time
                        entry["last_accessed"] = current_time
                        exact_match = (entry["response"], 1.0)
                
                # If no exact match, try semantic similarity (future enhancement)
                # For now, simple lowercase and whitespace normalization
                if not exact_match:
                    normalized_query = query.lower().strip()
                    
                    for key, entry in self.cache.items():
                        # Skip expired entries
                        if current_time >= entry["expires_at"]:
                            continue
                        
                        # Skip entries from different contexts if window_id is provided
                        if window_id and ":" in key and not key.endswith(f":{window_id}"):
                            continue
                        
                        # Simple string similarity check
                        orig_query = entry.get("original_query", "")
                        if not orig_query:
                            continue
                            
                        normalized_orig = orig_query.lower().strip()
                        
                        # Check if queries are similar enough
                        # This is a very simple check - replace with semantic similarity later
                        if (normalized_query in normalized_orig or 
                            normalized_orig in normalized_query or 
                            self._simple_similarity(normalized_query, normalized_orig) > self.similarity_threshold):
                            
                            similarity = self._simple_similarity(normalized_query, normalized_orig)
                            if similarity > best_score:
                                best_score = similarity
                                best_match = (entry["response"], similarity)
                                # Update access time
                                entry["last_accessed"] = current_time
            
            if exact_match:
                self.hit_count += 1
                logger.info(f"Cache hit (exact): {query[:50]}...")
                return exact_match
            elif best_match and best_score > self.similarity_threshold:
                self.hit_count += 1
                logger.info(f"Cache hit (similar, score={best_score:.3f}): {query[:50]}...")
                return best_match
            
            self.miss_count += 1
            logger.info(f"Cache miss: {query[:50]}...")
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    async def set(self, query: str, response: str, window_id: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a response in the cache.
        
        Args:
            query: The original query string
            response: The response to cache
            window_id: Optional window ID for context-specific caching
            metadata: Optional metadata to store with the cache entry
        """
        try:
            query_hash = self._hash_query(query)
            query_key = f"{query_hash}:{window_id}" if window_id else query_hash
            
            current_time = time.time()
            expires_at = current_time + self.ttl_seconds
            
            cache_entry = {
                "response": response,
                "original_query": query,
                "created_at": current_time,
                "last_accessed": current_time,
                "expires_at": expires_at,
                "window_id": window_id,
                "metadata": metadata or {}
            }
            
            async with self.lock:
                # Check if cache is full and needs cleanup
                if len(self.cache) >= self.max_size:
                    await self._evict_entries()
                
                self.cache[query_key] = cache_entry
                
            logger.info(f"Cached response for query: {query[:50]}... (key={query_key})")
        
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
    
    async def invalidate(self, query: str = None, window_id: Optional[str] = None) -> int:
        """
        Invalidate cache entries based on query or window_id.
        
        Args:
            query: Specific query to invalidate (None to use only window_id)
            window_id: Specific window_id to invalidate (None to use only query)
            
        Returns:
            Number of entries invalidated
        """
        try:
            count = 0
            
            if not query and not window_id:
                logger.warning("Both query and window_id are None, no entries will be invalidated")
                return 0
            
            query_hash = self._hash_query(query) if query else None
            
            async with self.lock:
                keys_to_remove = []
                
                for key in self.cache.keys():
                    if query_hash and key.startswith(query_hash):
                        keys_to_remove.append(key)
                    elif window_id and ":" in key and key.endswith(f":{window_id}"):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.cache[key]
                    count += 1
            
            logger.info(f"Invalidated {count} cache entries")
            return count
        
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0
    
    async def _cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        try:
            count = 0
            current_time = time.time()
            
            async with self.lock:
                keys_to_remove = [
                    key for key, entry in self.cache.items()
                    if current_time >= entry["expires_at"]
                ]
                
                for key in keys_to_remove:
                    del self.cache[key]
                    count += 1
            
            self.last_cleanup = current_time
            
            if count > 0:
                logger.info(f"Removed {count} expired cache entries")
            
            return count
        
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    async def _evict_entries(self) -> int:
        """
        Evict entries when cache is full.
        Uses LRU (Least Recently Used) strategy.
        
        Returns:
            Number of entries evicted
        """
        try:
            current_time = time.time()
            async with self.lock:
                # First remove expired entries
                expired_count = await self._cleanup_expired()
                
                # If still over limit, remove least recently used
                if len(self.cache) >= self.max_size:
                    # Sort by last accessed time (ascending)
                    sorted_entries = sorted(
                        self.cache.items(),
                        key=lambda x: x[1]["last_accessed"]
                    )
                    
                    # Remove oldest 10% or at least 1 entry
                    remove_count = max(1, int(len(self.cache) * 0.1))
                    
                    for i in range(remove_count):
                        if i < len(sorted_entries):
                            key, _ = sorted_entries[i]
                            del self.cache[key]
                    
                    logger.info(f"Evicted {remove_count} least recently used cache entries")
                    return expired_count + remove_count
            
            return expired_count
        
        except Exception as e:
            logger.error(f"Error evicting cache entries: {e}")
            return 0
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate a simple similarity score between two strings.
        This is a placeholder for more sophisticated semantic similarity.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Jaccard similarity of words
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_ratio": hit_ratio,
            "ttl_seconds": self.ttl_seconds
        }
    
    async def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        try:
            async with self.lock:
                count = len(self.cache)
                self.cache.clear()
            
            logger.info(f"Cleared {count} cache entries")
            return count
        
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0