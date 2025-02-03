import pinecone
from pinecone import Index, Pinecone
from typing import List, Dict, Any, Tuple, Optional
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.exceptions import PineconeError
from api.core.memory.models import Memory
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log, RetryError
import logging
import time
from datetime import datetime, timedelta
from api.core.memory.exceptions import MemoryOperationError
import asyncio
from api.core.memory.exceptions import PineconeError




logger = logging.getLogger(__name__)

class PineconeService(MemoryService):
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self._index = None
        self.initialized = False
        self.embedding_dimension = 1536  # Add default embedding dimension

    @property
    def index(self) -> Index:
        """Lazy initialization of the Pinecone index."""
        if self._index is None:
            self._initialize_pinecone()
        return self._index

    def _initialize_pinecone(self):
        """Initializes the Pinecone client and index."""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=pinecone.PodSpec(environment=self.environment)
                )
            self._index = self.pc.Index(self.index_name)
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise PineconeError(f"Failed to initialize Pinecone index: {e}") from e

    async def create_memory(self, memory_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Creates a memory in Pinecone."""
        try:
            self.index.upsert(vectors=[(memory_id, vector, metadata)])
            return True
        except Exception as e:
            logger.error(f"Failed to create memory in Pinecone: {e}")
            raise PineconeError(f"Failed to create memory: {e}") from e
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a memory from Pinecone by its ID."""
        try:
            response = self.index.fetch(ids=[memory_id])
            if response and memory_id in response['vectors']:
                vector_data = response['vectors'][memory_id]
                return {
                    'id': memory_id,
                    'vector': vector_data['values'],
                    'metadata': vector_data['metadata']
                }
            else:
                logger.warning(f"Memory with ID '{memory_id}' not found in Pinecone.")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory from Pinecone: {e}")
            raise PineconeError(f"Failed to retrieve memory: {e}") from e
    
    async def query_memories(
        self, 
        query_vector: List[float], 
        top_k: int = 10, 
        filter: Dict = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Queries Pinecone for memories similar to the query vector."""
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_values=True,  # Changed to True to include vector values
                include_metadata=True,
                filter=filter
            )
            
            memories_with_scores = []
            for result in results['matches']:
                memory_data = {
                    'id': result['id'],
                    'metadata': result['metadata'],
                    'vector': result.get('values', [0.0] * self.embedding_dimension),  # Ensure vector is always present
                    'content': result['metadata'].get('content', ''),  # Add content field
                    'memory_type': result['metadata'].get('memory_type', 'EPISODIC')  # Add memory_type field
                }
                memories_with_scores.append((memory_data, result['score']))
            return memories_with_scores
            
        except Exception as e:
            logger.error(f"Failed to query memories from Pinecone: {e}")
            raise PineconeError(f"Failed to query memories: {e}") from e
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Deletes a memory from Pinecone by its ID."""
        try:
            self.index.delete(ids=[memory_id])
            logger.info(f"Memory deleted successfully: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory from Pinecone: {e}")
            raise PineconeError(f"Failed to delete memory: {e}") from e
    
    async def update_memory(self, memory_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Updates a memory in Pinecone."""
        try:
            self.index.update(id=memory_id, values=vector, set_metadata=metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to update memory in Pinecone: {e}")
            raise PineconeError(f"Failed to update memory: {e}") from e
        
    async def batch_upsert_memories(self, memories: List[Tuple[str, List[float], Dict[str, Any]]]) -> Dict[str, Any]:
        """Upserts multiple memories to Pinecone in a batch."""
        try:
            result = {"upserted_count": 0}
            self.index.upsert(vectors=memories)
            result["upserted_count"] += len(memories)
            return result
        except Exception as e:
            logger.error(f"Failed to batch upsert memories to Pinecone: {e}")
            raise PineconeError(f"Failed to batch upsert memories: {e}") from e

    async def close_connections(self):
        """Closes the Pinecone index connection."""
        if self._index:
            self._index = None

    async def health_check(self) -> Dict[str, Any]:
        """Checks the health of the Pinecone service."""
        try:
            if not self.initialized:
                await self._initialize_pinecone()
            pinecone.init(api_key=self.api_key, environment=self.environment)
            pinecone.list_indexes()
            return {
                "status": "healthy",
                "initialized": self.initialized,
                "index": self.index_name
            }
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            return {
                "status": "unhealthy", 
                "error": str(e),
                "initialized": self.initialized
            }

    async def fetch_recent_memories(
        self,
        memory_type: str,
        max_age_days: int,
        limit: int = 1000
    ) -> List[Memory]:
        """Fetch recent memories with retry logic"""
        if not self.initialized:
            raise MemoryOperationError("Pinecone service not initialized")
            
        retries = 3
        while retries > 0:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
                cutoff_timestamp = int(time.mktime(cutoff_date.timetuple()))

                query_response = self.index.query(
                    vector=[0] * self.embedding_dimension,
                    filter={
                        "memory_type": memory_type,
                        "created_at": {"$gte": cutoff_timestamp}
                    },
                    top_k=limit,
                    include_metadata=True
                )

                memories = []
                for match in query_response.matches:
                    if not match.metadata:
                        continue

                    try:
                        memory = Memory(
                            id=match.id,
                            content=match.metadata["content"],
                            memory_type=match.metadata["memory_type"],
                            created_at=match.metadata["created_at"],
                            semantic_vector=match.values,
                            metadata=match.metadata,
                            window_id=match.metadata.get("window_id")
                        )
                        memories.append(memory)
                    except Exception as e:
                        logger.warning(f"Failed to convert match to Memory: {e}")
                        continue

                logger.info(
                    f"Fetched {len(memories)} recent memories of type '{memory_type}' "
                    f"within the last {max_age_days} days."
                )
                return memories

            except pinecone_exceptions.PineconeException as e:
                retries -= 1
                if retries == 0:
                    logger.error(f"Pinecone error during fetch recent memories: {e}")
                    raise MemoryOperationError(f"Failed to fetch recent memories: {e}")
                logger.warning(f"Retrying fetch recent memories, attempts remaining: {retries}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error fetching recent memories: {e}")
                raise MemoryOperationError(f"Failed to fetch recent memories: {e}")

    async def archive_memory(self, memory_id: str) -> bool:
        """Archive a memory with retry logic"""
        if not self.initialized:
            raise MemoryOperationError("Pinecone service not initialized")
            
        retries = 3
        while retries > 0:
            try:
                self.index.update(
                    id=memory_id,
                    set_metadata={
                        "archived": True,
                        "archived_at": datetime.utcnow().isoformat()
                    }
                )
                logger.info(f"Memory archived successfully: {memory_id}")
                return True
            except pinecone_exceptions.PineconeException as e:
                retries -= 1
                if retries == 0:
                    logger.error(f"Failed to archive memory {memory_id}: {str(e)}")
                    raise MemoryOperationError(f"Failed to archive memory: {str(e)}")
                logger.warning(f"Retrying archive, attempts remaining: {retries}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Failed to archive memory {memory_id}: {str(e)}")
                raise MemoryOperationError(f"Failed to archive memory: {str(e)}")