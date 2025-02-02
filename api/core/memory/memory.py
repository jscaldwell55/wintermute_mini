from datetime import datetime
import uuid
from typing import List, Dict, Optional, Tuple, Any
import logging
from api.core.memory.models import (
    Memory,
    MemoryType,
    CreateMemoryRequest,
    MemoryResponse,
    QueryRequest,
    QueryResponse
)
from api.utils.config import get_settings

logger = logging.getLogger(__name__)

class MemoryOperationError(Exception):
    """Base exception for memory operations."""
    pass

class MemorySystem:
    def __init__(self, pinecone_service, vector_operations):
        self.settings = get_settings()
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        
    async def create_memory_from_request(
        self,
        request: CreateMemoryRequest
    ) -> MemoryResponse:
        """Create a memory from a validated request."""
        try:
            memory_id = await self.add_memory(
                content=request.content,
                memory_type=request.memory_type,
                metadata=request.metadata,
                window_id=request.window_id
            )
            
            # Retrieve the created memory to return as response
            memory = await self.get_memory_by_id(memory_id)
            return MemoryResponse(
                id=memory.id,
                content=memory.content,
                memory_type=memory.memory_type,
                created_at=memory.created_at,
                metadata=memory.metadata,
                window_id=memory.window_id
            )
            
        except Exception as e:
            logger.error(f"Error creating memory from request: {e}")
            raise MemoryOperationError(f"Failed to create memory: {str(e)}")
        
    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        window_id: Optional[str] = None
    )  -> str:
        """Add a memory to the system."""
        try:
            # Generate semantic vector
            semantic_vector = await self.vector_operations.create_semantic_vector(content)
        
            # Generate memory ID
            memory_id = f"mem_{uuid.uuid4()}"
        
            # Create ISO format timestamp
            created_at = datetime.utcnow().isoformat()
        
            # Prepare metadata with explicit ISO format timestamp
            full_metadata = {
                "content": content,
                "created_at": created_at,
                "memory_type": memory_type.value,
                **(metadata or {}),
            }
        
            if window_id:
                full_metadata["window_id"] = window_id
        
            # Create memory object
            memory = Memory(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                semantic_vector=semantic_vector,
                metadata=full_metadata,
                created_at=created_at,
                window_id=window_id
            )
        
            # Store in Pinecone
            await self.pinecone_service.upsert_memory(
                memory_id=memory.id,
                vector=semantic_vector,
                metadata=memory.metadata
            )
        
            return memory.id
        
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise MemoryOperationError(f"Failed to add memory: {str(e)}")

    async def get_memory_by_id(self, memory_id: str) -> Memory:
        """Retrieve a memory by its ID."""
        try:
            memory_data = await self.pinecone_service.fetch_memory(memory_id)
            if not memory_data:
                raise MemoryOperationError(f"Memory not found: {memory_id}")
                
            return Memory(
                id=memory_id,
                content=memory_data["metadata"]["content"],
                memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                created_at=memory_data["metadata"]["created_at"],
                semantic_vector=memory_data["vector"],
                metadata=memory_data["metadata"],
                window_id=memory_data["metadata"].get("window_id")
            )
            
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            raise MemoryOperationError(f"Failed to retrieve memory: {str(e)}")

    async def query_memories(
        self,
        request: QueryRequest
    ) -> QueryResponse:
        """Query memories based on a query request."""
        try:
            # Generate query vector
            query_vector = await self.vector_operations.create_semantic_vector(request.prompt)
            
            # Build filter
            filter_dict = {}
            if request.window_id:
                filter_dict["window_id"] = request.window_id
            
            # Query Pinecone
            results = await self.pinecone_service.query_memory(
                query_vector=query_vector,
                top_k=request.top_k,
                filter=filter_dict
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
                    window_id=memory_data["metadata"].get("window_id")
                )
                matches.append(memory_response)
                similarity_scores.append(score)
            
            return QueryResponse(
                matches=matches,
                similarity_scores=similarity_scores
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
            window_id=window_id
        )
        return memory_id