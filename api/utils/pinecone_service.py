# api/utils/pinecone_service.py 

import pinecone
from pinecone import Index, Pinecone, PodSpec
from typing import List, Dict, Any, Tuple, Optional
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.exceptions import PineconeError, MemoryOperationError
from api.core.memory.models import Memory
# No tenacity needed here anymore
import logging
logging.basicConfig(level=logging.INFO)
import time
from datetime import datetime, timezone  # Import timezone
import asyncio
from api.utils.utils import normalize_timestamp # Import the helper

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

        logger.info(f"PineconeService init: api_key={api_key}, environment={environment}, index_name={index_name}")

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

    def _initialize_pinecone(self):
        """Initializes the Pinecone client and index with error handling."""
        try:
            if not self.api_key or not self.environment or not self.index_name:
                raise ValueError("Missing Pinecone API credentials!")

            logger.info("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=self.api_key)
            logger.info(f"Pinecone client initialized: {self.pc}")


            # Check existing indexes and create if needed
            existing_indexes = self.pc.list_indexes().names()
            logger.info(f"Existing indexes = {existing_indexes}")
            if self.index_name not in existing_indexes:
                logger.warning(f"Index '{self.index_name}' not found. Creating it...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=pinecone.PodSpec(environment=self.environment)  # Corrected
                )
                logger.info(f"Index '{self.index_name}' created.")
            else:
                logger.info("Index already exists")

            # Initialize the index and test connection
            self._index = self.pc.Index(self.index_name)
            logger.info(f"Pinecone index initialized: {self._index}")
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
                    cleaned_metadata[key] = str(value)  # Convert to string

            logger.info(f"📝 Creating memory in Pinecone: {memory_id}")
            self.index.upsert(vectors=[(memory_id, vector, cleaned_metadata)])
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create memory: {e}")
            raise PineconeError(f"Failed to create memory: {e}") from e

    async def batch_upsert_memories(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Upserts a batch of memories to Pinecone."""
        try:
            if self.index is None:
                raise PineconeError("Pinecone index is not initialized.")

            logger.info(f"📝 Upserting batch of {len(vectors)} memories to Pinecone.")
            self.index.upsert(vectors=vectors)
            logger.info("✅ Batch upsert successful.")

        except Exception as e:
            logger.error(f"❌ Failed to batch upsert memories: {e}")
            raise PineconeError(f"Failed to batch upsert memories: {e}") from e


    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a memory from Pinecone by its ID, parsing created_at."""
        try:
            if self.index is None:
                logger.error("❌ Pinecone index is None! Cannot fetch memory.")
                return None

            logger.info(f"🔍 Fetching memory from Pinecone: {memory_id}")
            response = self.index.fetch(ids=[memory_id])

            if response is None:
                logger.error(f"❌ Pinecone returned None for memory_id '{memory_id}'")
                return None

            if 'vectors' in response and memory_id in response['vectors']:
                vector_data = response['vectors'][memory_id]
                metadata = vector_data['metadata']

                # Use normalize_timestamp here
                created_at_raw = metadata.get("created_at")
                if isinstance(created_at_raw, str):
                    created_at = datetime.fromisoformat(normalize_timestamp(created_at_raw))
                    metadata['created_at'] = created_at

                return {
                    'id': memory_id,
                    'vector': vector_data['values'],
                    'metadata': metadata  # Return metadata with parsed datetime
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
            # Use the 'upsert' method, which acts as an update if the ID already exists
            self.index.upsert(vectors=[(memory_id, vector, metadata)])
            logger.info(f"📝 Memory updated in Pinecone: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update memory in Pinecone: {e}")
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
        include_metadata: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Queries the Pinecone index, parsing created_at in metadata."""
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
            # Correct fix for pinecone_service.py (around line ~225) - Fully specified:
            # In the query_memories method, where you're processing result metadata:

            for result in results['matches']:
                metadata = result['metadata']
                created_at_raw = metadata.get("created_at")
            
                # Handle different timestamp formats with better error handling
                try:
                    if isinstance(created_at_raw, (int, float)):
                        # If timestamp is numeric (Unix timestamp), convert to datetime
                        created_at = datetime.fromtimestamp(created_at_raw, tz=timezone.utc)
                    elif isinstance(created_at_raw, str):
                        # If timestamp is string, normalize and convert
                        normalized = created_at_raw.replace('Z', '').replace('+00:00', '')
                        created_at = datetime.fromisoformat(normalized + '+00:00')
                    else:
                        logger.warning(f"Memory {result['id']}: Unexpected created_at format: {type(created_at_raw)}")
                        created_at = datetime.now(timezone.utc)  # Fallback
                except Exception as e:
                    logger.warning(f"Error processing timestamp for memory {result['id']}: {e}")
                    created_at = datetime.now(timezone.utc)  # Fallback on any error
            
                # Update metadata with processed datetime
                metadata['created_at'] = created_at
            
                memory_data = {
                    'id': result['id'],
                    'metadata': metadata,  # Use the updated metadata
                    'vector': result.get('values', [0.0] * self.embedding_dimension),
                    'content': metadata.get('content', ''),
                    'memory_type': metadata.get('memory_type', 'EPISODIC')
                }
                memories_with_scores.append((memory_data, result['score']))
            return memories_with_scores

        except Exception as e:
            logger.error(f"Failed to query memories from Pinecone: {e}")
            raise PineconeError(f"Failed to query memories: {e}") from e


    async def delete_all_episodic_memories(self) -> None:
        """Deletes ALL episodic memories."""

        try:
            delete_filter = {"memory_type": "EPISODIC"}
            logger.info("Deleting ALL episodic memories.")

            to_delete = await self.query_memories(
                query_vector=[0.0] * self.embedding_dimension,
                top_k=10000,
                filter=delete_filter,
                include_metadata=False,
                include_values=False,
            )
            if to_delete:
                ids_to_delete = [mem[0]['id'] for mem in to_delete]
                logger.info(f"Deleting {len(ids_to_delete)} episodic memories.")
                if ids_to_delete:
                    await self.delete_memories(ids_to_delete)
                else:
                    logger.info("No episodic memories matched for deletion.")
            else:
                logger.info("No episodic memories to delete.")

        except Exception as e:
            logger.error(f"Error deleting episodic memories: {e}")
            raise PineconeError(f"Failed to delete episodic memories: {e}") from e



    async def delete_memories(self, ids_to_delete: List[str]) -> bool:
        """Deletes memories by their IDs."""
        try:
            self.index.delete(ids=ids_to_delete)
            logger.info(f"Memories deleted successfully: {ids_to_delete}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memories from Pinecone: {e}")
            raise PineconeError(f"Failed to delete memories: {e}") from e
        
    
    async def _check_recent_duplicate(self, content: str, window_minutes: int = 30) -> bool:
        """Safe duplicate check with proper timestamp handling."""
        try:
            # Extract core message
            normalized_content = ' '.join(content.lower().split())
            if "User:" in normalized_content:
                normalized_content = normalized_content.split("User:", 1)[1]
            if "Assistant:" in normalized_content:
                normalized_content = normalized_content.split("Assistant:", 1)[0]
            
            vector = await self.vector_operations.create_semantic_vector(normalized_content)
        
            # Use timestamp as number (epoch) for Pinecone
            cutoff_time = int(time.time() - (window_minutes * 60))
        
            results = await self.pinecone_service.query_memories(
                query_vector=vector,
                top_k=5,
                filter={
                    "memory_type": "EPISODIC",
                    "created_at": {"$gte": cutoff_time}  # Use epoch timestamp
                },
                include_metadata=True
            )
        
            for memory_data, score in results:
                if score > self.settings.duplicate_similarity_threshold:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Duplicate check failed, proceeding with storage: {e}")
            return False

    async def close_connections(self):
        """Closes the Pinecone index connection."""
        if self._index:
                self._index = None

    async def health_check(self) -> Dict[str, Any]: # Add async here
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