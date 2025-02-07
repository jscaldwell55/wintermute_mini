# api/utils/pinecone_service.py
import pinecone
from pinecone import Index, Pinecone, ServerlessSpec, PodSpec
from typing import List, Dict, Any, Tuple, Optional
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.exceptions import PineconeError, MemoryOperationError
from api.core.memory.models import Memory
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log, RetryError
import logging
logging.basicConfig(level=logging.INFO)
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

        # Call initialize during init but handle failures gracefully
        try:
            self._initialize_pinecone()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone during service creation: {e}")
            # Don't raise here - let the service be created but not initialized

    @property
    def index(self) -> Index:
        """Lazy initialization of the Pinecone index with proper error handling."""
        if not self._index:
            logger.warning("Pinecone index is not initialized. Attempting to initialize...")
            self._initialize_pinecone()

        if not self._index:
            raise PineconeError("Pinecone index failed to initialize.")

        return self._index

    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self._index = None
        self.initialized = False
        self.embedding_dimension = 1536  # Default embedding dimension

        logger.info(f"PineconeService init: api_key={api_key}, environment={environment}, index_name={index_name}") # Add logging

        # Call initialize during init but handle failures gracefully
        try:
            self._initialize_pinecone()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone during service creation: {e}")
            # Don't raise here - let the service be created but not initialized

    def _initialize_pinecone(self):
        """Initializes the Pinecone client and index with error handling."""
        try:
            if not self.api_key or not self.environment or not self.index_name:
                raise ValueError("Missing Pinecone API credentials!")

            logger.info("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=self.api_key)
            logger.info(f"Pinecone client initialized: {self.pc}") # Log the client object


            # Check existing indexes and create if needed
            existing_indexes = self.pc.list_indexes().names()
            logger.info(f"Existing indexes = {existing_indexes}") #Log the existing indexes
            if self.index_name not in existing_indexes:
                logger.warning(f"Index '{self.index_name}' not found. Creating it...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=pinecone.PodSpec(environment=self.environment) # Changed to PodSpec
                )
                logger.info(f"Index '{self.index_name}' created.") #log
            else: #Log if index already created
                logger.info("Index already exists")

            # Initialize the index and test connection
            self._index = self.pc.Index(self.index_name)
            logger.info(f"Pinecone index initialized: {self._index}")# Log the index object
            if self._index is None:
                raise RuntimeError("Pinecone index creation failed!")

            # Test the connection immediately
            try:
                self._index.describe_index_stats()
                self.initialized = True
                logger.info(f"✅ Pinecone index '{self.index_name}' initialized and connected successfully.")
            except Exception as e:
                logger.error(f"Failed to connect to Pinecone index: {e}")
                self._index = None
                raise

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self._index = None
            self.initialized = False
            raise PineconeError(f"Failed to initialize Pinecone: {e}")

    async def create_memory(self, memory_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Creates a memory in Pinecone with error handling."""
        try:
            if self.index is None:
                logger.error("❌ Pinecone index is not initialized. Cannot create memory.")
                raise PineconeError("Pinecone index is not initialized.")

            # Sanitize metadata
            cleaned_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    cleaned_metadata[key] = value
                elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                    cleaned_metadata[key] = value
                else:
                    cleaned_metadata[key] = str(value)

            logger.info(f"📝 Creating memory in Pinecone: {memory_id}")
            self.index.upsert(vectors=[(memory_id, vector, cleaned_metadata)])
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create memory: {e}")
            raise PineconeError(f"Failed to create memory: {e}") from e

    async def batch_upsert_memories(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Upserts a batch of memories to Pinecone.

        Args:
            vectors: A list of tuples.  Each tuple should contain:
                (memory_id: str, vector: List[float], metadata: Dict[str, Any]).
        """
        try:
            if self.index is None:
                raise PineconeError("Pinecone index is not initialized.")

            # The Pinecone 'upsert' method *already* handles batching efficiently.
            # We just need to make sure the data is in the correct format.
            logger.info(f"📝 Upserting batch of {len(vectors)} memories to Pinecone.")
            self.index.upsert(vectors=vectors) # Changed to 'vectors'
            logger.info("✅ Batch upsert successful.")

        except Exception as e:
            logger.error(f"❌ Failed to batch upsert memories: {e}")
            raise PineconeError(f"Failed to batch upsert memories: {e}") from e


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
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = False
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Queries the Pinecone index, now with include_metadata."""
        try:
            logger.info(f"Querying Pinecone with filter: {filter}, include_metadata: {include_metadata}")
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_values=True,
                include_metadata=include_metadata,
                filter=filter
            )

            logger.info(f"Raw Pinecone results: {str(results)[:200]}")  # Log first 200 chars

            memories_with_scores = []
            for result in results['matches']:
                logger.info(f"Processing result: {type(result)} - {str(result)[:100]}")
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
        
async def delete_old_episodic_memories(self, max_age_days: int = 7) -> None:
        """
        Deletes episodic memories older than max_age_days.

        Args:
            max_age_days: The maximum age (in days) of memories to keep.
        """
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        cutoff_timestamp = int(cutoff_time.timestamp())

        try:
            # We use a filter to find old episodic memories.  Note the $lt (less than) operator.
            delete_filter = {
                "memory_type": "EPISODIC",
                "created_at": {"$lt": cutoff_timestamp}  # Find memories *older* than the cutoff
            }

            logger.info(f"Deleting episodic memories older than {cutoff_time.isoformat()} (timestamp: {cutoff_timestamp})")

            # Fetch IDs to delete, to avoid issues with large result sets.  We *don't* need the content.
            to_delete = await self.query_memories(
                query_vector=[0.0] * self.embedding_dimension,  # Dummy vector
                top_k=10000, # Fetch many
                filter=delete_filter,
                include_metadata=False, # no need for metadata
                include_values=False, #no need for values
            )
            if to_delete:
                ids_to_delete = [mem[0]['id'] for mem in to_delete]
                logger.info(f"Deleting {len(ids_to_delete)} old episodic memories.")

            # The actual deletion is done with the standard 'delete' method
                if ids_to_delete: # Check that it isn't empty
                    await self.delete_memories(ids_to_delete) # Use the existing method in main
                else:
                    logger.info("No episodic memories matched to be deleted") #If list is empty

            else:
                logger.info("No episodic memories to delete.")



        except Exception as e:
            logger.error(f"Error deleting old episodic memories: {e}")
            raise PineconeError(f"Failed to delete old episodic memories: {e}") from e

async def delete_memories(self, ids_to_delete: List[str]) -> bool:
    """Deletes a memory by its ID."""
    try:
        self.index.delete(ids=ids_to_delete)
        logger.info(f"Memories deleted successfully: {ids_to_delete}")
        return True
    except Exception as e:
            logger.error(f"Failed to delete memory from Pinecone: {e}")
            raise PineconeError(f"Failed to delete memory: {e}") from e
    

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