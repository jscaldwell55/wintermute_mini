# memory.py
from datetime import datetime
import uuid
from typing import List, Dict, Optional, Tuple, Any
import logging
import asyncio
from pydantic import BaseModel, Field
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError, before_sleep_log
from collections import OrderedDict

# Updated imports:
from api.utils.config import Settings
from api.core.memory.cache import MemoryCache
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.interfaces.vector_operations import VectorOperations
from api.core.memory.models import Memory, MemoryType, CreateMemoryRequest, MemoryResponse, QueryRequest, QueryResponse
from api.core.memory.exceptions import MemoryOperationError

logger = logging.getLogger(__name__)

# Constants for retry mechanism
RETRY_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 1

class MemorySystem:
    def __init__(
        self,
        pinecone_service: MemoryService,
        vector_operations: VectorOperations,
        settings: Optional[Settings] = None,
        cache_capacity: int = 100,
    ):
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        self.settings = settings or Settings()
        self.cache = MemoryCache(capacity=cache_capacity)
        self._initialized = False  # Add initialization flag
        if not self.cache:
            logger.error("Failed to initialize memory cache")
            raise MemoryOperationError("Cache initialization failed")

    async def initialize(self) -> bool:
        """Initialize the memory system and its components."""
        try:
            # Verify vector operations initialization
            if hasattr(self.vector_operations, 'initialize'):
                await self.vector_operations.initialize()

            # Verify Pinecone initialization
            if hasattr(self.pinecone_service, 'initialize'):
                await self.pinecone_service.initialize()

            # Initialize cache if enabled
            if self.cache:
                await self.cache.clear()  # Now properly awaited

            self._initialized = True
            logger.info("Memory system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            self._initialized = False
            return False

    async def ensure_initialized(self) -> bool:
        """Ensure the memory system is initialized."""
        if not self._initialized:
            return await self.initialize()
        return True

    async def batch_create_memories(
        self,
        requests: List[CreateMemoryRequest]
    ) -> List[str]:
        """Create multiple memories in batch."""
        try:
            memory_ids = []
            vectors_batch = []

            # Process in chunks of 100
            chunk_size = 100
            for i in range(0, len(requests), chunk_size):
                chunk = requests[i:i + chunk_size]

                # Generate vectors in parallel
                vector_tasks = [
                    self.vector_operations.create_semantic_vector(req.content)
                    for req in chunk
                ]
                vectors = await asyncio.gather(*vector_tasks)

                # Prepare batch upsert data
                for req, vector in zip(chunk, vectors):
                    memory_id = f"mem_{uuid.uuid4()}"
                    memory_ids.append(memory_id)

                    metadata = {
                        "content": req.content,
                        "created_at": datetime.utcnow().isoformat(),
                        "memory_type": req.memory_type.value,
                        **(req.metadata or {}),
                    }

                    if req.window_id:
                        metadata["window_id"] = req.window_id

                    vectors_batch.append((memory_id, vector, metadata))

                # Batch upsert to Pinecone
                await self.pinecone_service.batch_upsert_memories(vectors_batch)

            return memory_ids

        except Exception as e:
            logger.error(f"Batch memory creation failed: {e}")
            raise MemoryOperationError(f"Failed to create memories in batch: {str(e)}")
    
    async def create_memory_from_request(
        self,
        request: CreateMemoryRequest
    ) -> MemoryResponse:
        """Create a memory from a validated request."""
        try:
            logger.info(f"Creating memory from request: {request}")
        
            # First create the memory
            memory_id = await self.add_memory(
                content=request.content,
                memory_type=request.memory_type,
                metadata=request.metadata,
                window_id=request.window_id
            )
        
            if not memory_id:
                raise MemoryOperationError("Failed to generate memory ID")
            
            logger.info(f"Memory created with ID: {memory_id}")

            # Add a small delay to ensure consistency
            await asyncio.sleep(0.5)

            # Retrieve the created memory
            memory = await self.get_memory_by_id(memory_id)
            if not memory:
                raise MemoryOperationError(f"Failed to retrieve newly created memory: {memory_id}")

            logger.info(f"Successfully retrieved memory: {memory_id}")
        
            return MemoryResponse(
                id=memory.id,
                content=memory.content,
                memory_type=memory.memory_type,
                created_at=memory.created_at,
                metadata=memory.metadata,
                window_id=memory.window_id,
                semantic_vector=memory.semantic_vector
            )

        except Exception as e:
            logger.error(f"Error creating memory from request: {e}", exc_info=True)
            raise MemoryOperationError(f"Failed to create memory: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_fixed(RETRY_WAIT_SECONDS),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _add_memory_with_retry(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        window_id: Optional[str] = None
    ) -> str:
        """Add a memory to the system with retry mechanism."""
        logger.info(f"Attempting to add memory. Content: {content}, Memory Type: {memory_type}")

        # Generate memory ID with 'mem_' prefix here
        memory_id = "mem_" + str(uuid.uuid4())

        # Create ISO format timestamp
        created_at = datetime.utcnow().isoformat()

        try:
            # Generate semantic vector
            semantic_vector = await self.vector_operations.create_semantic_vector(content)
        
            # Convert to list if it's a numpy array
            if hasattr(semantic_vector, 'tolist'):
                semantic_vector = semantic_vector.tolist()

            # Prepare metadata
            full_metadata = {
                "content": content,
                "created_at": created_at,
                "memory_type": memory_type.value,
                **(metadata or {}),
            }

            if window_id:
                full_metadata["window_id"] = window_id

            memory = Memory(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                semantic_vector=semantic_vector,  # Now it's a list
                created_at=created_at,
                metadata=full_metadata,
                window_id=window_id
            )

            # Store in Pinecone
            await self.pinecone_service.create_memory(
                memory_id=memory.id,
                vector=semantic_vector,
                metadata=memory.metadata,
            )

            logger.info(f"Successfully added memory with ID: {memory_id}")
            return memory.id

        except Exception as e:
            logger.error(f"Error during memory creation process: {e}")
            raise MemoryOperationError(f"Failed to add memory during vector creation or upsertion: {e}")

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        window_id: Optional[str] = None
    ) -> str:
        """Add a memory to the system."""
        try:
            return await self._add_memory_with_retry(content, memory_type, metadata, window_id)
        except RetryError as e:
            logger.error(f"Error adding memory after multiple retries: {e.last_attempt.result()}")
            raise MemoryOperationError(f"Failed to add memory after retries: {e.last_attempt.result()}")

    async def get_memory_by_id(self, memory_id: str) -> Memory:
        """Retrieve a memory by its ID."""
        try:
            if not memory_id:
                raise MemoryOperationError("Memory ID cannot be None or empty")

            # Check if memory is in cache
            try:
                cached_memory = await self.cache.get(memory_id)
                if cached_memory:
                    logger.info(f"Cache hit for memory ID: {memory_id}")
                    return cached_memory
            except Exception as cache_error:
                logger.warning(f"Cache retrieval failed: {cache_error}")
                # Continue to Pinecone if cache fails

            # Fetch from Pinecone
            logger.info(f"Fetching memory from Pinecone: {memory_id}")
            memory_data = await self.pinecone_service.get_memory_by_id(memory_id)

            if not memory_data:
                logger.error(f"No memory data found for ID: {memory_id}")
                return None  # Return None instead of raising error for not found case

            # Create memory object
            memory = Memory(
                id=memory_id,
                content=memory_data["metadata"]["content"],
                memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                created_at=memory_data["metadata"]["created_at"],
                semantic_vector=memory_data["vector"],
                metadata=memory_data["metadata"],
                window_id=memory_data["metadata"].get("window_id")
            )

            # Try to cache the memory, but don't fail if caching fails
            try:
                await self.cache.put(memory)
            except Exception as cache_error:
                logger.warning(f"Failed to cache memory {memory_id}: {cache_error}")

            return memory

        except Exception as e:
            logger.error(f"Error retrieving memory: {e}", exc_info=True)
            raise MemoryOperationError(f"Failed to retrieve memory: {str(e)}")

    async def query_memories(
        self,
        request: QueryRequest
    ) -> QueryResponse:
        """Query memories based on a query request."""
        try:
            # Generate query vector
            query_vector = await self.vector_operations.create_semantic_vector(
                request.prompt
            )

            # Build filter
            filter_dict = {}
            if request.window_id:
                filter_dict["window_id"] = request.window_id

            # Query Pinecone
            results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=request.top_k,
                filter=filter_dict,
            )

            # Convert results to response model
            matches = []
            similarity_scores = []

            for memory_data, score in results:
                memory_response = MemoryResponse(
                    id=memory_data["id"],
                    content=memory_data["metadata"]["content"],
                    memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                    created_at=memory_data["metadata"]["created_at"],
                    metadata=memory_data["metadata"],
                    window_id=memory_data["metadata"].get("window_id"),
                    semantic_vector=memory_data["vector"]
                )
                matches.append(memory_response)
                similarity_scores.append(score)

            return QueryResponse(
                matches=matches,
                similarity_scores=similarity_scores,
            )

        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            raise MemoryOperationError(f"Failed to query memories: {str(e)}")

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            await self.pinecone_service.delete_memory(memory_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False

    async def add_interaction(
        self,
        user_input: str,
        response: str,
        window_id: Optional[str] = None
    ) -> str:
        """Add an interaction as an episodic memory."""
        content = f"User: {user_input}\nAssistant: {response}"
        memory_id = await self.add_memory(
            content=content,
            memory_type=MemoryType.EPISODIC,
            metadata={"interaction": True},
            window_id=window_id,
        )
        return memory_id
    
    async def health_check(self):
        """Checks the health of the memory system."""
        return {"status": "healthy", "cache_enabled": self.settings.enable_memory_cache}