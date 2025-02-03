# api/core/memory/cache.py
from typing import Dict, Any, Optional
from collections import OrderedDict
import asyncio
from api.core.memory.models import Memory

class MemoryCache:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()  # Add lock for thread safety

    async def get(self, key: str) -> Optional[Memory]:
        """Get a value from cache asynchronously."""
        async with self._lock:
            if key in self.cache:
                # Move the accessed item to the end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    async def put(self, memory: Memory) -> None:
        """Put a value in cache asynchronously."""
        async with self._lock:
            self.cache[memory.id] = memory
            self.cache.move_to_end(memory.id)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # Remove least recently used item

    async def delete(self, key: str) -> bool:
        """Delete a value from cache asynchronously."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear the entire cache asynchronously."""
        async with self._lock:
            self.cache.clear()