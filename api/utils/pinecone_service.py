# pinecone_service.py
import pinecone
from pinecone import Index, Pinecone
from typing import List, Dict, Any, Tuple, Optional
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.exceptions import PineconeError, MemoryOperationError
from api.core.memory.models import Memory
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log, RetryError
import logging
import time
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class PineconeService(MemoryService):
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self._index = None
        self.initialized = False
        self.embedding_dimension = 1536  # Default embedding dimension

        self._initialize_pinecone()  # Ensure initialization at instantiation

    def _initialize_pinecone(self):
        """Initializes the Pinecone client and index with error handling."""
        try:
            if not self.api_key or not self.environment or not self.index_name:
                raise ValueError("Missing Pinecone API credentials!")

            logger.info("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=self.api_key)

            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.warning(f"Index '{self.index_name}' not found. Creating it...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=pinecone.PodSpec(environment=self.environment)
                )

            self._index = self.pc.Index(self.index_name)
            if self._index is None:
                raise RuntimeError("Pinecone index creation failed!")

            self.initialized = True
            logger.info(f"✅ Pinecone index '{self.index_name}' initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self._index = None
            self.initialized = False
            raise PineconeError(f"Failed to initialize Pinecone: {e}")

    @property
    def index(self) -> Index:
        """Lazy initialization of the Pinecone index with proper error handling."""
        if self._index is None:
            logger.warning("Pinecone index is not initialized. Attempting to initialize...")
            self._initialize_pinecone()

        if self._index is None:
            raise PineconeError("Pinecone index failed to initialize.")
    
        return self._index

    async def create_memory(self, memory_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Creates a memory in Pinecone with error handling."""
        try:
            if self.index is None:
                logger.error("❌ Pinecone index is not initialized. Cannot create memory.")
                raise PineconeError("Pinecone index is not initialized.")

            logger.info(f"📝 Creating memory in Pinecone: {memory_id}")
            self.index.upsert(vectors=[(memory_id, vector, metadata)])
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create memory: {e}")
            raise PineconeError(f"Failed to create memory: {e}") from e

    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        try:
            if self.index is None:
                logger.error("❌ Pinecone index is None! Cannot fetch memory.")
                return None
        
            logger.info(f"🔍 Fetching memory from Pinecone: {memory_id}")
        
            # Make sure to use synchronous fetch since Pinecone's Python client doesn't support async
            response = self.index.fetch(ids=[memory_id])

            if response is None:
                logger.error(f"❌ Pinecone returned None for memory_id '{memory_id}'")
                return None

            if memory_id in response['vectors']:
                vector_data = response['vectors'][memory_id]
                return {
                    'id': memory_id,
                    'vector': vector_data['values'],
                    'metadata': vector_data['metadata']
                }
            else:
                logger.warning(f"⚠️ Memory with ID '{memory_id}' not found in Pinecone.")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to retrieve memory from Pinecone: {e}")
            raise PineconeError(f"Failed to retrieve memory: {e}") from e


    async def update_memory(self, memory_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Updates an existing memory in Pinecone."""
        try:
            self.index.update(id=memory_id, values=vector, set_metadata=metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to update memory in Pinecone: {e}")
            raise PineconeError(f"Failed to update memory: {e}") from e

    async def delete_memory(self, memory_id: str) -> bool:
        """Deletes a memory by its ID."""
        try:
            self.index.delete(ids=[memory_id])
            logger.info(f"Memory deleted successfully: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory from Pinecone: {e}")
            raise PineconeError(f"Failed to delete memory: {e}") from e

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
                include_values=True,
                include_metadata=True,
                filter=filter
            )
            
            memories_with_scores = []
            for result in results['matches']:
                memory_data = {
                    'id': result['id'],
                    'metadata': result['metadata'],
                    'vector': result.get('values', [0.0] * self.embedding_dimension),
                    'content': result['metadata'].get('content', ''),
                    'memory_type': result['metadata'].get('memory_type', 'EPISODIC')
                }
                memories_with_scores.append((memory_data, result['score']))
            return memories_with_scores
            
        except Exception as e:
            logger.error(f"Failed to query memories from Pinecone: {e}")
            raise PineconeError(f"Failed to query memories: {e}") from e

    async def close_connections(self):
        """Closes the Pinecone index connection."""
        if self._index:
            self._index = None

    async def health_check(self) -> Dict[str, Any]:
        """Checks the health of the Pinecone service."""
        try:
            if not self.initialized or self._index is None:
                logger.error("Pinecone service is not initialized properly.")
                return {"status": "unhealthy", "error": "Pinecone not initialized"}

            response = self.pc.list_indexes()
            if self.index_name in response.names():
                return {"status": "healthy", "index": self.index_name}
            else:
                return {"status": "unhealthy", "error": "Index not found"}

        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
