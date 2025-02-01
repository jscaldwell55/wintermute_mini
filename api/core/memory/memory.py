from datetime import datetime
import uuid
from typing import List, Dict, Optional, Tuple
import logging
from api.core.memory.models import Memory, MemoryType
from api.utils.config import get_settings

logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(self, pinecone_service, vector_operations):
        self.settings = get_settings()
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        
    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a memory to the system."""
        try:
            # Generate semantic vector
            semantic_vector = await self.vector_operations.create_semantic_vector(content)
            
            # Generate memory ID
            memory_id = f"mem_{uuid.uuid4()}"
            
            # Prepare metadata
            full_metadata = {
                "content": content,
                "created_at": datetime.now().isoformat(),
                "memory_type": memory_type.value,
                **(metadata or {})
            }
            
            # Create memory object
            memory = Memory(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                semantic_vector=semantic_vector,
                metadata=full_metadata,
                created_at=full_metadata["created_at"]
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
            raise

    async def query_memory(
        self,
        query_vector: List[float],
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None
    ) -> List[Tuple[Memory, float]]:
        """Query memories based on vector similarity."""
        try:
            # Build filter if memory type specified
            filter_dict = {}
            if memory_type:
                filter_dict["memory_type"] = memory_type.value
                
            # Query Pinecone
            results = await self.pinecone_service.query_memory(
                query_vector=query_vector,
                top_k=top_k,
                filter=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            raise

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            await self.pinecone_service.delete_memory(memory_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False

    async def add_interaction(self, user_input: str, response: str):
        """Add an interaction as an episodic memory."""
        content = f"User: {user_input}\nAssistant: {response}"
        await self.add_memory(
            content=content,
            memory_type=MemoryType.EPISODIC,
            metadata={"interaction": True}
        )