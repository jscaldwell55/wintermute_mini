from datetime import datetime, timedelta
from typing import List, Optional
from api.core.memory.models import Memory
from api.utils.responses import MemoryOperationError
from typing import Dict, List, Optional, Any, Tuple
from pinecone import Pinecone, Index
import logging
from httpx import AsyncClient

logger = logging.getLogger(__name__)

class PineconeService:
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 1536
    ):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.pc = None
        self.index = None
        self._http_client = None

    async def initialize_index(self):
        """Initialize Pinecone index"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            
            if self.index_name not in existing_indexes.names():
                # Create index if it doesn't exist
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine"
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            # Get index instance
            self.index = self.pc.Index(self.index_name)
            self._http_client = AsyncClient()
            logger.info(f"Successfully connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            raise

    async def close_connections(self):
        """Close any open connections"""
        if self._http_client:
            await self._http_client.aclose()

    async def upsert_memory(
        self,
        memory_id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Upsert a memory vector and metadata to Pinecone"""
        try:
            self.index.upsert(
                vectors=[(memory_id, vector, metadata)],
                namespace=""
            )
            return True
        except Exception as e:
            logger.error(f"Error upserting memory to Pinecone: {e}")
            raise

    async def fetch_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific memory by ID from Pinecone"""
        try:
            response = self.index.fetch(ids=[memory_id])
            
            if not response or memory_id not in response['vectors']:
                return None
                
            vector_data = response['vectors'][memory_id]
            return {
                "id": memory_id,
                "vector": vector_data.values,
                "metadata": vector_data.metadata
            }
            
        except Exception as e:
            logger.error(f"Error fetching memory from Pinecone: {e}")
            raise

    async def query_memory(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Query similar memories from Pinecone"""
        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_values=True,
                include_metadata=True,
                filter=filter
            )
            
            results = []
            for match in response.matches:
                memory_data = {
                    "id": match.id,
                    "vector": match.values,
                    "metadata": match.metadata
                }
                results.append((memory_data, match.score))
                
            return results
            
        except Exception as e:
            logger.error(f"Error querying memories from Pinecone: {e}")
            raise

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from Pinecone"""
        try:
            self.index.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting memory from Pinecone: {e}")
            raise

    async def delete_all_memories(self) -> bool:
        """Delete all memories from the index"""
        try:
            self.index.delete(delete_all=True)
            return True
        except Exception as e:
            logger.error(f"Error deleting all memories from Pinecone: {e}")
            raise

    async def fetch_recent_memories(
        self,
        memory_type: str,
        max_age_days: int,
        limit: int = 1000
    ) -> List[Memory]:
        """
        Fetch recent memories of a specific type within the given age limit.
    
        Args:
            memory_type: Type of memory to fetch (EPISODIC or SEMANTIC)
            max_age_days: Maximum age of memories in days
            limit: Maximum number of memories to fetch
        
        Returns:
            List of Memory objects
        """
        try:
            # Calculate the cutoff date and convert to ISO format
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            cutoff_date_iso = cutoff_date.isoformat()
        
            # Query Pinecone for recent memories
            query_response = self.index.query(
                vector=[0] * self.dimension,  # Dummy vector to match all
                filter={
                    "memory_type": memory_type,
                    "created_at": {"$gte": cutoff_date_iso}
                },
                top_k=limit,
                include_metadata=True
            )
        
            # Convert to Memory objects
            memories = []
            for match in query_response.matches:
                if not match.metadata:
                    continue
                
                try:
                    memory_data = match.metadata.copy()
                    memory_data['id'] = match.id
                    memory_data['semantic_vector'] = match.values
                    memories.append(Memory(**memory_data))
                except Exception as e:
                    logger.warning(f"Failed to convert match to Memory: {str(e)}")
                    continue
        
            return memories
        
        except Exception as e:
            logger.error(f"Error fetching recent memories: {str(e)}")
            raise MemoryOperationError(
                operation="fetch_recent_memories",
                details=f"Failed to fetch recent memories: {str(e)}"
            )

    async def archive_memory(self, memory_id: str) -> None:
        """
        Archive a memory by setting its archived flag.
        
        Args:
            memory_id: ID of the memory to archive
        """
        try:
            # Update the memory's metadata to mark it as archived
            self.index.update(
                id=memory_id,
                set_metadata={
                    "archived": True,
                    "archived_at": datetime.utcnow().timestamp()
                }
            )
        except Exception as e:
            raise MemoryOperationError(
                operation="archive_memory",
                details=f"Failed to archive memory {memory_id}: {str(e)}"
            )    