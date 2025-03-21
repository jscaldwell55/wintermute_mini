# api/utils/pinecone_service.py

import pinecone
from pinecone import Index, Pinecone, PodSpec
from typing import List, Dict, Any, Tuple, Optional
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.exceptions import PineconeError, MemoryOperationError
from api.core.memory.models import Memory, MemoryType
import logging

logging.basicConfig(level=logging.INFO)
import time
from datetime import datetime, timezone
import asyncio
from api.utils.utils import normalize_timestamp
import math
import random

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

        logger.info(
            f"PineconeService init: api_key={api_key}, environment={environment}, index_name={index_name}"
        )

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
                    spec=pinecone.PodSpec(environment=self.environment),
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
                logger.info(
                    f"✅ Pinecone index '{self.index_name}' initialized and connected successfully."
                )
            except Exception as e:
                logger.error(f"Failed to connect to Pinecone index: {e}")
                self._index = None
                raise

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self._index = None
            self.initialized = False
            raise PineconeError(f"Failed to initialize Pinecone: {e}")

    async def create_memory(
        self, memory_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> bool:
        """Creates a memory in Pinecone with standardized timestamp handling."""
        try:
            if self.index is None:
                logger.error("❌ Pinecone index is not initialized. Cannot create memory.")
                raise PineconeError("Pinecone index is not initialized.")

            # Ensure consistent timestamp format in all new memories
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is None:  # skip None values
                    continue
                elif key == "created_at" and isinstance(value, str):
                    # Use the normalize_timestamp function which returns a datetime
                    dt = normalize_timestamp(value)
                    cleaned_metadata["created_at_iso"] = dt.isoformat(timespec='milliseconds') + "Z" # Store ISO with Z
                    cleaned_metadata["created_at_unix"] = int(dt.timestamp()) # Store Unix timestamp
                elif key == "created_at" and isinstance(value, datetime):
                    cleaned_metadata["created_at_iso"] = value.isoformat(timespec='milliseconds') + "Z" # Store ISO with Z
                    cleaned_metadata["created_at_unix"] = int(value.timestamp()) # Store Unix timestamp
                elif isinstance(value, (str, int, float, bool, list)):
                    cleaned_metadata[key] = value
                elif isinstance(value, datetime):
                    cleaned_metadata[key] = value.isoformat(timespec='milliseconds') + "Z" # Ensure ISO with Z
                    if key.endswith("_at") and not key.endswith("_unix"): # Also store unix for other datetime fields
                        cleaned_metadata[f"{key}_unix"] = int(value.timestamp())
                else:
                    cleaned_metadata[key] = str(value)

            # Ensure 'created_at' (for legacy reasons) is also ISO format, but without Z - for backward compatibility if needed, though 'created_at_iso' is preferred
            if "created_at_iso" in cleaned_metadata:
                cleaned_metadata["created_at"] = datetime.fromisoformat(cleaned_metadata["created_at_iso"].rstrip('Z')).isoformat()

            logger.info(
                f"📝 Creating memory in Pinecone: {memory_id}, metadata keys: {list(cleaned_metadata.keys())}"
            )  # Log metadata keys

            self.index.upsert(vectors=[(memory_id, vector, cleaned_metadata)])
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create memory: {e}")
            raise PineconeError(f"Failed to create memory: {e}") from e

    async def batch_upsert_memories(
        self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]
    ) -> None:
        """Upserts a batch of memories to Pinecone with standardized timestamp handling."""
        try:
            if self.index is None:
                raise PineconeError("Pinecone index is not initialized.")

            batch_vectors_cleaned = []
            for memory_id, vector, metadata in vectors:
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if value is None:  # skip None values
                        continue
                    elif key == "created_at" and isinstance(value, str):
                        # Use the normalize_timestamp function which now returns a datetime
                        dt = normalize_timestamp(value)
                        cleaned_metadata["created_at_iso"] = dt.isoformat(timespec='milliseconds') + "Z" # Store ISO with Z
                        cleaned_metadata["created_at_unix"] = int(dt.timestamp()) # Store Unix timestamp
                    elif key == "created_at" and isinstance(value, datetime):
                        cleaned_metadata["created_at_iso"] = value.isoformat(timespec='milliseconds') + "Z" # Store ISO with Z
                        cleaned_metadata["created_at_unix"] = int(value.timestamp()) # Store Unix timestamp
                    elif isinstance(value, (str, int, float, bool, list)):
                        cleaned_metadata[key] = value
                    elif isinstance(value, datetime):
                        cleaned_metadata[key] = value.isoformat(timespec='milliseconds') + "Z" # Ensure ISO with Z
                        if key.endswith("_at") and not key.endswith("_unix"): # Also store unix for other datetime fields
                            cleaned_metadata[f"{key}_unix"] = int(value.timestamp())
                    else:
                        cleaned_metadata[key] = str(value)

                # Ensure 'created_at' is also ISO format (without Z) for legacy reasons
                if "created_at_iso" in cleaned_metadata:
                    cleaned_metadata["created_at"] = datetime.fromisoformat(cleaned_metadata["created_at_iso"].rstrip('Z')).isoformat()


                batch_vectors_cleaned.append((memory_id, vector, cleaned_metadata))

            # Log batch information
            logger.info(f"📝 Upserting batch of {len(batch_vectors_cleaned)} memories to Pinecone.")

            # Handle batch size limits (typically 100 for Pinecone)
            batch_size = 100
            for i in range(0, len(batch_vectors_cleaned), batch_size):
                batch_chunk = batch_vectors_cleaned[i:i + batch_size]
                self.index.upsert(vectors=batch_chunk)
                logger.info(f"✅ Batch chunk {i//batch_size + 1} upsert successful ({len(batch_chunk)} vectors).")

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

            # Handle response as a FetchResponse object (new SDK)
            if hasattr(response, "vectors") and memory_id in response.vectors:
                vector_data = response.vectors[memory_id]
                metadata = (
                    vector_data.metadata if hasattr(vector_data, "metadata") else {}
                )
                vector_values = (
                    vector_data.values if hasattr(vector_data, "values") else []
                )

                # Fix timestamp parsing - handle both +00:00Z and other formats robustly
                created_at_raw = metadata.get("created_at")
                if isinstance(created_at_raw, str):
                    try:
                        created_at = normalize_timestamp(created_at_raw) # Updated line: Use normalize_timestamp directly
                        metadata["created_at"] = created_at
                    except ValueError as e:
                        logger.warning(f"Error parsing timestamp '{created_at_raw}': {e}, using current time")
                        metadata["created_at"] = datetime.now(timezone.utc)

                return {
                    "id": memory_id,
                    "vector": vector_values,
                    "metadata": metadata,
                }
            # Fall back to dictionary-style access (old SDK)
            elif (
                isinstance(response, dict)
                and "vectors" in response
                and memory_id in response["vectors"]
            ):
                vector_data = response["vectors"][memory_id]
                metadata = vector_data["metadata"]

                # Use normalize_timestamp here
                created_at_raw = metadata.get("created_at")
                if isinstance(created_at_raw, str):
                    try:
                        # Check what normalize_timestamp returns
                        normalized_value = normalize_timestamp(created_at_raw)

                        # Handle based on return type
                        if isinstance(normalized_value, str):
                            created_at = datetime.fromisoformat(normalized_value.rstrip("Z"))
                        else:
                            # Already a datetime object
                            created_at = normalized_value

                        metadata["created_at"] = created_at
                    except ValueError as e:
                        logger.warning(f"Error parsing timestamp '{created_at_raw}': {e}, using current time")
                        metadata["created_at"] = datetime.now(timezone.utc)

                return {
                    "id": memory_id,
                    "vector": vector_data["values"],
                    "metadata": metadata,  # Return metadata with parsed datetime
                }
            else:
                logger.warning(
                    f"⚠️ Memory with ID '{memory_id}' not found in Pinecone."
                )
                return None
        except Exception as e:
            logger.error(f"❌ Failed to retrieve memory from Pinecone: {e}")
            raise PineconeError(f"Failed to retrieve memory: {e}") from e

    async def update_memory(
        self, memory_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> bool:
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
        include_metadata: bool = True,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Queries the Pinecone index with DYNAMIC filter logic restored and enhanced logging."""
        try:
            # Always include metadata in this implementation
            include_metadata = True
            logger.info(
                f"Querying Pinecone with filter (raw): {filter}, include_metadata: {include_metadata}"
            )  # Log raw filter

            query_filter = filter or {} # Use filter directly as we expect it to be correctly formatted with 'created_at_unix'

            # Log the filter being sent
            logger.info(f"Filter being sent to Pinecone QUERY: {query_filter}") # Log the FILTER!
            if query_filter and "created_at_unix" in query_filter and isinstance(query_filter["created_at_unix"], dict):
                for op, ts in query_filter["created_at_unix"].items():
                    logger.info(f"  {op.upper()} timestamp (Unix): {ts}") # Log the timestamp range
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    logger.info(f"  {op.upper()} as date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            elif query_filter and "created_at_unix" in query_filter: # Log for single value as well
                created_at_unix_filter = query_filter["created_at_unix"]
                logger.info(
                    f"Data type of created_at_unix filter: {type(created_at_unix_filter)}"
                )  # Log the type

            # Execute the query
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_values=True,
                include_metadata=include_metadata,
                filter=query_filter,
            )

            # Add null check for results
            if not results or "matches" not in results or not results["matches"]:
                logger.warning("Pinecone query returned no matches or empty result structure")
                return []

            # Process results
            logger.info(
                f"Pinecone query returned {len(results.get('matches', []))} matches"
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Raw Pinecone results: {str(results)[:200]}")

            memories_with_scores = []

            for result in results["matches"]:
                metadata = result["metadata"]

                # Process created_at timestamp
                created_at_raw = metadata.get("created_at")
                created_at_iso = metadata.get("created_at_iso")

                # Try ISO format first if available
                if created_at_iso:
                    try:
                        created_at = normalize_timestamp(created_at_iso) # Updated line: Use normalize_timestamp
                        metadata["created_at"] = created_at
                    except Exception as e:
                        logger.warning(
                            f"Error parsing created_at_iso for memory {result['id']}: {e}"
                        )

                # Fall back to created_at
                elif created_at_raw:
                    try:
                        if isinstance(created_at_raw, (int, float)):
                            # If timestamp is numeric (Unix timestamp), convert to datetime
                            created_at = datetime.fromtimestamp(
                                created_at_raw, tz=timezone.utc
                            )
                        elif isinstance(created_at_raw, str):
                            # If timestamp is string, convert directly (expecting ISO now)
                            created_at = normalize_timestamp(created_at_raw) # Updated line: Use normalize_timestamp
                        else:
                            logger.warning(
                                f"Memory {result['id']}: Unexpected created_at format: {type(created_at_raw)}"
                            )
                            created_at = datetime.now(timezone.utc)  # Fallback

                        metadata["created_at"] = created_at
                    except Exception as e:
                        logger.warning(
                            f"Error processing timestamp for memory {result['id']}: {e}"
                        )
                        metadata["created_at"] = datetime.now(timezone.utc)  # Fallback

                # Create memory data
                memory_data = {
                    "id": result["id"],
                    "metadata": metadata,
                    "vector": result.get("values", [0.0] * self.embedding_dimension),
                    "content": metadata.get("content", ""),
                    "memory_type": metadata.get("memory_type", "EPISODIC"),
                }

                memories_with_scores.append((memory_data, result["score"]))

            if not memories_with_scores: # If NO matches returned, try to fetch some episodic memories for timestamp inspection
                logger.warning("No matches returned. Attempting to fetch a few episodic memories for timestamp inspection...")
                try:
                    # Fetch a few episodic memory IDs (replace with actual IDs if you know some)
                    sample_memory_ids = ["mem_3af60f6237b64650bd0236ea2248507e", "mem_e94c36f3694a409691bd6166a1c1b736", "mem_ef70a9c068ee43228b551e9ce52871c0"] # Replace with actual IDs, using IDs from your initial logs
                    sample_fetch_response = self.index.fetch(ids=sample_memory_ids)
                    if sample_fetch_response and sample_fetch_response.vectors:
                        for mem_id, vec_data in sample_fetch_response.vectors.items():
                            if vec_data.metadata.get("memory_type") == "EPISODIC":
                                created_at_unix_val = vec_data.metadata.get("created_at_unix")
                                logger.info(f"  Sample EPISODIC memory ID: {mem_id}, created_at_unix: {created_at_unix_val}") # Log sample memory timestamps
                except Exception as e_fetch_sample:
                    logger.warning(f"Error fetching sample memories for timestamp inspection: {e_fetch_sample}")


            return memories_with_scores

        except Exception as e:
            logger.error(f"Failed to query memories from Pinecone: {e}")
            raise PineconeError(f"Failed to query memories: {e}") from e
        
    async def update_memory_metadata(self, memory_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """Updates specific metadata fields for a memory without changing the vector."""
        try:
            # First, get the current memory to preserve the vector and other metadata
            memory_data = await self.get_memory_by_id(memory_id)
            if not memory_data:
                logger.warning(f"Cannot update metadata for memory {memory_id}: Memory not found")
                return False
                
            # Merge the current metadata with the updates
            updated_metadata = memory_data["metadata"].copy()
            updated_metadata.update(metadata_updates)
            
            # Convert any datetime objects to ISO strings for Pinecone
            for key, value in updated_metadata.items():
                if isinstance(value, datetime):
                    updated_metadata[key] = value.isoformat() + "Z"
            
            # Re-upsert the memory with the updated metadata
            self.index.upsert(vectors=[(
                memory_id, 
                memory_data["vector"], 
                updated_metadata
            )])
            
            logger.info(f"📝 Updated metadata for memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update memory metadata: {e}")
            return False

    async def sample_memories(
        self, limit: int = 1000, include_vector: bool = False
    ) -> List[Memory]:
        """
        Get a sample of memories for building indexes

        Args:
            limit: Maximum number of memories to retrieve
            include_vector: Whether to include vectors in the results

        Returns:
            List of Memory objects
        """
        try:
            logger.info(
                f"Sampling up to {limit} memories for keyword index initialization"
            )
            memories = []

            # Distribute the limit across memory types
            per_type_limit = limit // 3

            # Define memory types to sample
            memory_types = ["SEMANTIC", "EPISODIC", "LEARNED"]

            for memory_type in memory_types:
                try:
                    # Query with a random vector to get a diverse sample
                    # This is more efficient than listing all vectors and sampling
                    import random

                    random_vector = [random.uniform(-1, 1) for _ in range(1536)]

                    # Normalize the vector
                    magnitude = math.sqrt(sum(x * x for x in random_vector))
                    normalized_vector = [x / magnitude for x in random_vector]

                    results = await self.query_memories(
                        query_vector=normalized_vector,
                        top_k=per_type_limit,
                        filter={"memory_type": memory_type},
                        include_metadata=True,
                    )

                    logger.info(f"Retrieved {len(results)} {memory_type} memories")

                    # Convert results to Memory objects
                    for memory_data, _ in results:
                        try:
                            # Parse the created_at timestamp
                            created_at_raw = memory_data["metadata"].get("created_at")

                            if isinstance(created_at_raw, str):
                                created_at = datetime.fromisoformat(
                                    normalize_timestamp(created_at_raw)
                                )
                            else:
                                # Handle the case where it might be an int (Unix timestamp)
                                created_at = datetime.fromtimestamp(
                                    created_at_raw, tz=timezone.utc
                                )

                            memory = Memory(
                                id=memory_data["id"],
                                content=memory_data["metadata"]["content"],
                                memory_type=MemoryType(
                                    memory_data["metadata"]["memory_type"]
                                ),
                                created_at=created_at,
                                metadata=memory_data["metadata"],
                                window_id=memory_data["metadata"].get("window_id"),
                                semantic_vector=memory_data.get("vector")
                                if include_vector
                                else None,
                            )
                            memories.append(memory)
                        except Exception as e:
                            logger.warning(
                                f"Error converting memory data to Memory: {e}",
                                exc_info=True,
                            )

                except Exception as e:
                    logger.warning(
                        f"Error sampling {memory_type} memories: {e}", exc_info=True
                    )

            logger.info(f"Successfully sampled {len(memories)} total memories")
            return memories

        except Exception as e:
            logger.error(f"Error sampling memories: {e}", exc_info=True)
            return []

    async def get_all_memories(
        self, filter: Optional[Dict[str, Any]] = None, limit: int = 1000
    ) -> List[Memory]:
        """
        Retrieves all memories matching the given filter (with a reasonable limit)

        Args:
            filter: Optional filter criteria
            limit: Maximum number of memories to return

        Returns:
            List of Memory objects
        """
        try:
            logger.info(f"Retrieving all memories with filter: {filter}, limit: {limit}")

            # We need a dummy vector for the query - use all zeros
            dummy_vector = [0.0] * self.embedding_dimension

            # Query Pinecone with a high top_k
            results = await self.query_memories(
                query_vector=dummy_vector,
                top_k=limit,
                filter=filter,
                include_metadata=True,
            )

            # Convert to Memory objects
            memories = []

            for memory_data, _ in results:
                try:
                    memory_type_str = memory_data["metadata"].get(
                        "memory_type", "EPISODIC"
                    )

                    memory = Memory(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_type_str),
                        created_at=memory_data["metadata"][
                            "created_at"
                        ],  # Already a datetime
                        metadata=memory_data["metadata"],
                        window_id=memory_data["metadata"].get("window_id"),
                        semantic_vector=memory_data.get("vector"),
                    )
                    memories.append(memory)
                except Exception as e:
                    logger.error(f"Error converting memory {memory_data['id']}: {e}")
                    continue

            logger.info(f"Retrieved {len(memories)} memories")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve all memories: {e}")
            raise PineconeError(f"Failed to retrieve all memories: {e}") from e

    async def get_memories_by_window_id(
        self, window_id: str, limit: int = 100
    ) -> List[Memory]:
        """Retrieves memories by window ID."""
        try:
            logger.info(f"Retrieving memories by window ID: {window_id}")

            # Use get_all_memories with a filter for window_id
            memories = await self.get_all_memories(
                filter={"window_id": window_id}, limit=limit
            )

            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories by window ID: {e}")
            return []

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
            )
            if to_delete:
                ids_to_delete = [mem[0]["id"] for mem in to_delete]
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

    # Removed the _check_recent_duplicate method as it referenced non-existent properties

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