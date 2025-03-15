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
        """Creates a memory in Pinecone with error handling."""
        try:
            if self.index is None:
                logger.error("❌ Pinecone index is not initialized. Cannot create memory.")
                raise PineconeError("Pinecone index is not initialized.")

            # Sanitize metadata - be more selective
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is None:  # skip None values
                    continue
                elif isinstance(
                    value, (str, int, float, bool, list)
                ):  # Allow these types directly
                    cleaned_metadata[key] = value
                elif isinstance(
                    value, datetime
                ):  # Handle datetime objects specifically - convert to ISO string for Pinecone metadata (if you want to store string representation as well)
                    cleaned_metadata[
                        key
                    ] = value.isoformat() + "Z"  # Store datetime as ISO string (optional - you might not need to store string representation if you have created_at_unix)
                else:
                    cleaned_metadata[key] = str(
                        value
                    )  # Fallback to string for other types

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
        """Upserts a batch of memories to Pinecone."""
        try:
            if self.index is None:
                raise PineconeError("Pinecone index is not initialized.")

            batch_vectors_cleaned = []
            for memory_id, vector, metadata in vectors:
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if value is None:  # skip None values
                        continue
                    elif isinstance(
                        value, (str, int, float, bool, list)
                    ):  # Allow these types directly
                        cleaned_metadata[key] = value
                    elif isinstance(
                        value, datetime
                    ):  # Handle datetime objects specifically - convert to ISO string (optional)
                        cleaned_metadata[
                            key
                        ] = value.isoformat() + "Z"  # Store datetime as ISO string (optional)
                    else:
                        cleaned_metadata[key] = str(
                            value
                        )  # Fallback to string for other types
                batch_vectors_cleaned.append((memory_id, vector, cleaned_metadata))

            logger.info(
                f"📝 Upserting batch of {len(batch_vectors_cleaned)} memories to Pinecone."
            )  # Use cleaned vector count
            self.index.upsert(vectors=batch_vectors_cleaned)  # Use cleaned vectors for upsert
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

            # Handle response as a FetchResponse object (new SDK)
            if hasattr(response, "vectors") and memory_id in response.vectors:
                vector_data = response.vectors[memory_id]
                metadata = (
                    vector_data.metadata if hasattr(vector_data, "metadata") else {}
                )
                vector_values = (
                    vector_data.values if hasattr(vector_data, "values") else []
                )

                # Use normalize_timestamp here
                created_at_raw = metadata.get("created_at")
                if isinstance(created_at_raw, str):
                    created_at = datetime.fromisoformat(
                        normalize_timestamp(created_at_raw)
                    )
                    metadata["created_at"] = created_at

                return {
                    "id": memory_id,
                    "vector": vector_values,
                    "metadata": metadata,  # Return metadata with parsed datetime
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
                    created_at = datetime.fromisoformat(
                        normalize_timestamp(created_at_raw)
                    )
                    metadata["created_at"] = created_at

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
        """Queries the Pinecone index, normalizing timestamp formats in filters."""
        try:
            logger.info(
                f"Querying Pinecone with filter (raw): {filter}, include_metadata: {include_metadata}"
            )  # Log raw filter

            # Normalize timestamp formats in filter
            normalized_filter = None
            if filter:
                normalized_filter = filter.copy()

                # Handle created_at filters specifically
                if "created_at" in normalized_filter:
                    # If created_at is a dictionary (range query)
                    if isinstance(normalized_filter["created_at"], dict):
                        # Process each operator in the range query
                        unix_ranges = {}
                        for op, val in normalized_filter["created_at"].items():
                            if isinstance(val, datetime):
                                unix_ranges[op] = int(val.timestamp())
                            elif isinstance(val, str):
                                dt = datetime.fromisoformat(
                                    normalize_timestamp(val).rstrip("Z")
                                )
                                unix_ranges[op] = int(dt.timestamp())
                            else:
                                # Already a number (likely a timestamp)
                                unix_ranges[op] = val

                        # Replace with created_at_unix for better filtering
                        normalized_filter["created_at_unix"] = unix_ranges
                        del normalized_filter["created_at"]
                    else:
                        # Single value created_at (exact match)
                        val = normalized_filter["created_at"]
                        if isinstance(val, datetime):
                            normalized_filter["created_at_unix"] = int(val.timestamp())
                        elif isinstance(val, str):
                            dt = datetime.fromisoformat(
                                normalize_timestamp(val).rstrip("Z")
                            )
                            normalized_filter["created_at_unix"] = int(dt.timestamp())

                        # Remove original created_at after converting
                        del normalized_filter["created_at"]

            # Use normalized filter or original if no normalization was needed
            query_filter = (
                normalized_filter if normalized_filter is not None else filter
            )

            # Log the normalized filter for debugging - CRITICAL LOGGING ADDITION
            logger.info(f"Normalized filter (sent to Pinecone): {query_filter}")

            if query_filter and "created_at_unix" in query_filter:  # Add check for query_filter
                created_at_unix_filter = query_filter["created_at_unix"]
                logger.info(
                    f"Data type of created_at_unix filter: {type(created_at_unix_filter)}"
                )  # Log the type
                if isinstance(created_at_unix_filter, dict):
                    for op, val in created_at_unix_filter.items():
                        logger.info(
                            f"  Operator: {op}, Value: {val}, Value Data Type: {type(val)}"
                        )  # Log type of values within range query
                else:
                    logger.info(
                        f"  Value: {created_at_unix_filter}, Data Type: {type(created_at_unix_filter)}"
                    )  # Log type of single value

            # Execute the query
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_values=True,
                include_metadata=include_metadata,
                filter=query_filter,
            )

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
                        created_at = datetime.fromisoformat(
                            normalize_timestamp(created_at_iso)
                        )
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
                            # If timestamp is string, normalize and convert
                            created_at = datetime.fromisoformat(
                                normalize_timestamp(created_at_raw)
                            )
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

            return memories_with_scores

        except Exception as e:
            logger.error(f"Failed to query memories from Pinecone: {e}")
            raise PineconeError(f"Failed to query memories: {e}") from e

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
                        include_vector=include_vector,
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