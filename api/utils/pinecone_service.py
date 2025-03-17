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
                    # Always store created_at as ISO string for readability
                    cleaned_metadata["created_at"] = value

                    # Ensure we always have created_at_unix for filtering
                    if "created_at_unix" not in metadata:
                        # Parse the ISO string and convert to unix timestamp
                        dt = datetime.fromisoformat(normalize_timestamp(value))
                        cleaned_metadata["created_at_unix"] = int(dt.timestamp())
                elif key == "created_at" and isinstance(value, datetime):
                    # Convert datetime to ISO string and store unix timestamp
                    cleaned_metadata["created_at"] = value.isoformat() + "Z"
                    cleaned_metadata["created_at_unix"] = int(value.timestamp())
                elif isinstance(value, (str, int, float, bool, list)):
                    cleaned_metadata[key] = value
                elif isinstance(value, datetime):
                    cleaned_metadata[key] = value.isoformat() + "Z"

                    # If this is some other datetime field, also store a unix version
                    if key.endswith("_at") and not key.endswith("_unix"):
                        unix_key = f"{key}_unix"
                        cleaned_metadata[unix_key] = int(value.timestamp())
                else:
                    cleaned_metadata[key] = str(value)

            # This is now properly outside the loop
            # Ensure created_at_unix is always present
            if "created_at_unix" not in cleaned_metadata and "created_at" in cleaned_metadata:
                try:
                    dt = datetime.fromisoformat(normalize_timestamp(cleaned_metadata["created_at"]))
                    cleaned_metadata["created_at_unix"] = int(dt.timestamp())
                except Exception as e:
                    logger.warning(f"Failed to create created_at_unix from created_at: {e}")
                    # Fallback to current time
                    cleaned_metadata["created_at_unix"] = int(time.time())

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
                        # Always store created_at as ISO string for readability
                        cleaned_metadata["created_at"] = value

                        # Ensure we always have created_at_unix for filtering
                        if "created_at_unix" not in metadata:
                            # Parse the ISO string and convert to unix timestamp
                            dt = datetime.fromisoformat(normalize_timestamp(value))
                            cleaned_metadata["created_at_unix"] = int(dt.timestamp())
                    elif key == "created_at" and isinstance(value, datetime):
                        # Convert datetime to ISO string and store unix timestamp
                        cleaned_metadata["created_at"] = value.isoformat() + "Z"
                        cleaned_metadata["created_at_unix"] = int(value.timestamp())
                    elif isinstance(value, (str, int, float, bool, list)):
                        cleaned_metadata[key] = value
                    elif isinstance(value, datetime):
                        cleaned_metadata[key] = value.isoformat() + "Z"

                        # If this is some other datetime field, also store a unix version
                        if key.endswith("_at") and not key.endswith("_unix"):
                            unix_key = f"{key}_unix"
                            cleaned_metadata[unix_key] = int(value.timestamp())
                    else:
                        cleaned_metadata[key] = str(value)

                # This is now properly outside the inner loop
                # Ensure created_at_unix is always present
                if "created_at_unix" not in cleaned_metadata and "created_at" in cleaned_metadata:
                    try:
                        dt = datetime.fromisoformat(normalize_timestamp(cleaned_metadata["created_at"]))
                        cleaned_metadata["created_at_unix"] = int(dt.timestamp())
                    except Exception as e:
                        logger.warning(f"Failed to create created_at_unix from created_at for memory {memory_id}: {e}")
                        # Fallback to current time
                        cleaned_metadata["created_at_unix"] = int(time.time())

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
                        # Fix double timezone indicator: either remove Z or +00:00
                        if created_at_raw.endswith('Z') and '+00:00' in created_at_raw:
                            created_at_raw = created_at_raw.replace('+00:00Z', 'Z')

                        # Now parse the fixed timestamp
                        created_at = datetime.fromisoformat(
                            normalize_timestamp(created_at_raw)
                        )
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
        """Queries the Pinecone index with DYNAMIC filter logic restored and enhanced logging."""
        try:
            # Always include metadata in this implementation
            include_metadata = True
            logger.info(
                f"Querying Pinecone with filter (raw): {filter}, include_metadata: {include_metadata}"
            )  # Log raw filter

            # --- RESTORED DYNAMIC FILTER LOGIC (Original, but with logging) ---
            normalized_filter = None
            if filter:
                normalized_filter = filter.copy()

                # Handle created_at filters specifically
                if "created_at" in normalized_filter:
                    logger.info("Original filter had 'created_at', normalizing...") # Log when normalization is triggered
                    # If created_at is a dictionary (range query)
                    if isinstance(normalized_filter["created_at"], dict):
                        logger.info("  created_at is a dict (range query)") # Log range query detection
                        unix_ranges = {}
                        for op, val in normalized_filter["created_at"].items():
                            if isinstance(val, datetime):
                                unix_ranges[op] = int(val.timestamp())
                                logger.info(f"  Operator: {op}, Datetime Value: {val}, Converted Unix Timestamp: {unix_ranges[op]}") # Log datetime conversion
                            elif isinstance(val, str):
                                dt = datetime.fromisoformat(
                                    normalize_timestamp(val).rstrip("Z")
                                )
                                unix_ranges[op] = int(dt.timestamp())
                                logger.info(f"  Operator: {op}, String Value: {val}, Normalized Datetime: {dt}, Converted Unix Timestamp: {unix_ranges[op]}") # Log string conversion and normalization
                            else:
                                # Already a number (likely a timestamp)
                                unix_ranges[op] = val
                                logger.info(f"  Operator: {op}, Numeric Value (assumed timestamp): {val}") # Log numeric value assumption

                        # Replace with created_at_unix for better filtering
                        normalized_filter["created_at_unix"] = unix_ranges
                        del normalized_filter["created_at"]
                    else:
                        # Single value created_at (exact match - unlikely in temporal queries, but handling just in case)
                        logger.info("  created_at is NOT a dict (single value - exact match?)") # Log single value case
                        val = normalized_filter["created_at"]
                        if isinstance(val, datetime):
                            normalized_filter["created_at_unix"] = int(val.timestamp())
                            logger.info(f"  Datetime Value: {val}, Converted Unix Timestamp: {normalized_filter['created_at_unix']}") # Log datetime conversion
                        elif isinstance(val, str):
                            dt = datetime.fromisoformat(
                                normalize_timestamp(val).rstrip("Z")
                            )
                            normalized_filter["created_at_unix"] = int(dt.timestamp())
                            logger.info(f"  String Value: {val}, Normalized Datetime: {dt}, Converted Unix Timestamp: {normalized_filter['created_at_unix']}") # Log string conversion and normalization
                        else:
                            # Already a number (likely a timestamp)
                            normalized_filter["created_at_unix"] = val
                            logger.info(f"  Numeric Value (assumed timestamp): {val}") # Log numeric value assumption

                        # Remove original created_at after converting
                        del normalized_filter["created_at"]
            else:
                logger.info("No filter or no 'created_at' in filter - no normalization needed.") # Log no normalization case


            # Use normalized filter or original if no normalization was needed
            query_filter = (
                normalized_filter if normalized_filter is not None else filter
            )

            # --- End of RESTORED DYNAMIC FILTER LOGIC ---

            # Log the normalized filter for debugging
            logger.info(f"Normalized filter (sent to Pinecone): {query_filter}")
            logger.info(f"Filter being sent to Pinecone QUERY: {query_filter}") # Log the FILTER!
            if query_filter and "created_at_unix" in query_filter and isinstance(query_filter["created_at_unix"], dict):
                for op, ts in query_filter["created_at_unix"].items():
                    logger.info(f"  {op.upper()} timestamp (Unix): {ts}") # Log the timestamp range


            if query_filter and "created_at_unix" in query_filter:  # Add check for query_filter
                created_at_unix_filter = query_filter["created_at_unix"]
                logger.info(
                    f"Data type of created_at_unix filter: {type(created_at_unix_filter)}"
                )  # Log the type

                # Check if this is a temporal query (looking for when something was discussed)
                is_temporal_query = False
                if hasattr(self, 'current_query') and self.current_query:
                    is_temporal_query = any(term in self.current_query.lower()
                                        for term in ["when did we", "when have we", "what time", "how long ago"])

                # Handle range queries (most common for temporal filters)
                if isinstance(created_at_unix_filter, dict):
                    # Log original values before any modifications
                    logger.info(f"Original created_at_unix filter: {created_at_unix_filter}")

                    for op, val in created_at_unix_filter.items():
                        logger.info(
                            f"  Operator: {op}, Value: {val}, Value Data Type: {type(val)}"
                        )  # Log type of values within range query

                        # Within the query_memories method, when handling created_at_unix filters
                        if is_temporal_query and op == "$gte":
                            # Check if it's a "when did we discuss X" type query
                            is_topic_history_query = any(pattern in self.current_query.lower()
                                                    for pattern in ["when did we discuss", "when did we talk about", "when have we"])

                            # For "when did we discuss X" queries, use a much wider time window (7 days)
                            if is_topic_history_query:
                                # Widen the time window by moving back the start time to 7 days earlier
                                original_val = val
                                adjusted_val = original_val - (7 * 24 * 3600)  # 7 days earlier
                                created_at_unix_filter[op] = adjusted_val
                                logger.info(
                                    f"  ADJUSTED for topic history query: {op} value from {original_val} to {adjusted_val} "
                                    f"(-7 days to improve historical memory recall)"
                                )
                            else:
                                # For other temporal queries, use the existing 24-hour extension
                                original_val = val
                                adjusted_val = original_val - (24 * 3600)  # 24 hours earlier
                                created_at_unix_filter[op] = adjusted_val
                                logger.info(
                                    f"  ADJUSTED for temporal query: {op} value from {original_val} to {adjusted_val} "
                                    f"(-24h to improve memory recall)"
                                )
                else:
                    logger.info(
                        f"  Value: {created_at_unix_filter}, Data Type: {type(created_at_unix_filter)}"
                    )  # Log type of single value

                    # If it's a single value (exact match) and temporal query, convert to a range
                    if is_temporal_query:
                        exact_val = created_at_unix_filter
                        # Convert to a range query extending 48 hours before and after the timestamp
                        query_filter["created_at_unix"] = {
                            "$gte": exact_val - (48 * 3600),
                            "$lte": exact_val + (48 * 3600)
                        }
                        logger.info(
                            f"  Converted exact timestamp {exact_val} to range query for temporal query: "
                            f"{query_filter['created_at_unix']}"
                        )

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
                            created_at_iso # Removed normalize_timestamp here
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
                            # If timestamp is string, convert directly (expecting ISO now)
                            created_at = datetime.fromisoformat(
                                created_at_raw # Removed normalize_timestamp here
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