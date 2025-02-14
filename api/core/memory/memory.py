# api/core/memory/memory.py
import uuid
from typing import List, Dict, Optional, Tuple, Any
import logging
logging.basicConfig(level=logging.INFO)
import asyncio
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError, before_sleep_log
from collections import OrderedDict
import time
import math
from datetime import datetime, timedelta, timezone  # Import timezone

# Corrected imports:  Import from the correct locations
from api.utils.config import get_settings, Settings
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.interfaces.vector_operations import VectorOperations
from api.core.memory.models import (Memory, MemoryType, CreateMemoryRequest,
                                      MemoryResponse, QueryRequest, QueryResponse,
                                      RequestMetadata, OperationType, ErrorDetail)
from api.core.memory.exceptions import MemoryOperationError

logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(
        self,
        pinecone_service: MemoryService,
        vector_operations: VectorOperations,
        settings: Optional[Settings] = None  # Allow optional settings
    ):
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        self.settings = settings or get_settings()  # Use provided settings or get defaults
        self._initialized = False


    async def initialize(self) -> bool:
        """Initialize the memory system and its components."""
        try:
            # Verify vector operations initialization (if it has an initialize method)
            if hasattr(self.vector_operations, 'initialize'):
                await self.vector_operations.initialize()

            # Verify Pinecone initialization (if it has an initialize method)
            if hasattr(self.pinecone_service, 'initialize'):
                await self.pinecone_service.initialize()


            self._initialized = True
            logger.info("Memory system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            self._initialized = False
            return False

    async def ensure_initialized(self) -> bool:
        """Ensure the memory system is initialized."""
        if not self._initialized:
            return await self.initialize()
        return True


    async def batch_create_memories(
        self,
        requests: List[CreateMemoryRequest]
    ) -> List[str]:
        """Create multiple memories in batch."""
        try:
            memory_ids = []
            vectors_batch = []

            # Process in chunks of 100
            chunk_size = 100
            for i in range(0, len(requests), chunk_size):
                chunk = requests[i:i + chunk_size]

                # Generate vectors in parallel
                vector_tasks = [
                    self.vector_operations.create_semantic_vector(req.content)
                    for req in chunk
                ]
                vectors = await asyncio.gather(*vector_tasks)

                # Prepare batch upsert data
                for req, vector in zip(chunk, vectors):
                    memory_id = f"mem_{uuid.uuid4()}"
                    memory_ids.append(memory_id)

                    metadata = {
                        "content": req.content,
                        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                        "memory_type": req.memory_type.value,
                        **(req.metadata or {}),
                    }

                    if req.window_id:
                        metadata["window_id"] = req.window_id

                    vectors_batch.append((memory_id, vector, metadata))

                # Batch upsert to Pinecone
                await self.pinecone_service.batch_upsert_memories(vectors_batch)

            return memory_ids

        except Exception as e:
            logger.error(f"Batch memory creation failed: {e}")
            raise MemoryOperationError(f"Failed to create memories in batch: {str(e)}")


    async def create_memory_from_request(
        self,
        request: CreateMemoryRequest
    ) -> MemoryResponse:
        try:
            if not request.request_metadata:  # Ensure request_metadata exists
                request.request_metadata = RequestMetadata(operation_type=OperationType.STORE)
            else:
                request.request_metadata.operation_type = OperationType.STORE

            logger.info(f"Creating memory from request: {request}")
            memory_id = str(uuid.uuid4())
            created_at = datetime.utcnow().isoformat() + "Z" #consistent and timezone aware

            semantic_vector = await self.vector_operations.create_semantic_vector(request.content)

            metadata = {
                "content": request.content,
                "created_at": created_at,
                "memory_type": request.memory_type.value,
                **(request.metadata or {}),  # Include any extra metadata from the request
            }
            if request.window_id: #add window id
                metadata["window_id"] = request.window_id

            memory = Memory(
                id=memory_id,
                content=request.content,
                memory_type=request.memory_type,
                created_at=created_at,
                metadata=metadata,
                semantic_vector=semantic_vector,
                window_id=request.window_id,
                trace_id=request.request_metadata.trace_id
            )

            success = await self.pinecone_service.create_memory(
                memory_id=memory.id,
                vector=semantic_vector,
                metadata=memory.metadata
            )
            if not success:
                raise MemoryOperationError("Failed to store memory in vector database")

            logger.info(f"Successfully created memory with ID: {memory_id}")
            return memory.to_response()

        except Exception as e:
            logger.error(f"Error during memory creation process: {e}", exc_info=True)
            #  return an error response
            return MemoryResponse(
                id=memory_id,
                content=request.content,
                memory_type=request.memory_type,
                created_at=created_at,
                metadata=request.metadata,
                error=ErrorDetail(code="MEMORY_CREATION_FAILED", message=str(e)),
            )


    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
      try:
        logger.info(f"Retrieving memory by ID: {memory_id}")
        memory_data = await self.pinecone_service.get_memory_by_id(memory_id)
        if memory_data:
            # Create and return a Memory object
            return Memory(**memory_data)
        else:
            logger.info(f"Memory with ID '{memory_id}' not found.")
            return None
      except Exception as e:
        logger.error(f"Failed to retrieve memory by ID: {e}", exc_info=True)
        raise
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            logger.info(f"Deleting memory with ID: {memory_id}")
            return await self.pinecone_service.delete_memory(memory_id) #removed extra call
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}", exc_info=True)
            raise MemoryOperationError("delete_memory", str(e))

    async def query_memories(self, request: QueryRequest) -> QueryResponse:
        """Query memories based on the given request, with filtering and sorting."""

        try:
            logger.info(f"Starting memory query with request: {request}")

            # Generate query vector
            query_vector = await self.vector_operations.create_semantic_vector(request.prompt)
            logger.info(f"Query vector generated (first 10 elements): {query_vector[:10]}")

            # Prepare the filter for Pinecone
            pinecone_filter = {"memory_type": "SEMANTIC"}
            if request.window_id:
                pinecone_filter["window_id"] = request.window_id

            logger.info(f"Querying Pinecone with filter: {pinecone_filter}")

            # Query Pinecone, including metadata and values
            results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=request.top_k,
                filter=pinecone_filter,
                include_metadata=True
            )

            logger.info(f"Received {len(results)} raw results from Pinecone.")


            # Process and filter the results
            matches: List[MemoryResponse] = []
            similarity_scores: List[float] = []

            current_time = datetime.utcnow()

            for memory_data, similarity_score in results:
                logger.debug(f"Processing memory data: {memory_data}")
                try:
                    # Basic similarity threshold check
                    if similarity_score < 0.6:  # Adjust threshold as needed
                        logger.info(f"Skipping memory {memory_data['id']} due to low similarity score: {similarity_score}")
                        continue

                    # Apply time weighting
                    created_at_str = memory_data["metadata"].get("created_at")
                    if not created_at_str:
                        logger.warning(f"Skipping memory {memory_data['id']} due to missing created_at")
                        continue
                    
                    #Handle timestamps
                    if isinstance(created_at_str, str):
                        # Handle ISO 8601 strings (with or without 'Z')
                        if not created_at_str.endswith("Z"):
                            created_at_str += "Z"
                        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    elif isinstance(created_at_str, (int, float)):
                        # Handle numeric timestamps (assume seconds since epoch)
                        created_at = datetime.fromtimestamp(created_at_str, tz=timezone.utc)
                    else:
                        # Handle unexpected types (shouldn't happen, but be safe)
                        logger.warning(f"Unexpected created_at type: {type(created_at_raw)} in memory {memory_data['id']}")
                        created_at = datetime.now(timezone.utc)

                    age_days = (current_time - created_at).total_seconds() / (60*60*24)
                    # Example time weighting function (adjust as needed):
                    time_weight = 0.7 + (0.3 / (1 + math.exp(age_days/180 - 2))) # Example weighting function

                    # Combine similarity and time weighting (adjust weights as needed)
                    final_score = (similarity_score * 0.8) + (time_weight * 0.2)
                    logger.info(f"Memory ID {memory_data['id']}: Raw Score={similarity_score:.3f}, Time Weight={time_weight:.3f}, Final Score={final_score:.3f}")
                    # Convert to MemoryResponse
                    memory_response = MemoryResponse(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),  # Ensure this is an Enum
                        created_at= created_at.isoformat() + "Z",  # to string
                        metadata=memory_data["metadata"],
                        window_id=memory_data["metadata"].get("window_id"),
                        semantic_vector=memory_data["vector"],  # Include the vector
                    )
                    matches.append(memory_response)
                    similarity_scores.append(final_score)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing timestamp for memory {memory_data['id']}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing memory {memory_data['id']}: {e}", exc_info=True)
                    continue

            # Sort by final score and limit to requested number
            sorted_results = sorted(zip(matches, similarity_scores), key=lambda x: x[1], reverse=True)[:request.top_k]
            matches, similarity_scores = zip(*sorted_results) if sorted_results else ([], [])

            # Create and return the QueryResponse
            return QueryResponse(
                matches=list(matches),
                similarity_scores=list(similarity_scores),
            )

        except Exception as e:
            logger.error(f"Error querying memories: {e}", exc_info=True)
            raise MemoryOperationError(f"Failed to query memories: {str(e)}")

    async def store_interaction(self, query: str, response: str, window_id: Optional[str] = None) -> Memory:
        """Stores a user interaction (query + response) as a new episodic memory."""
        try:
            logger.info(f"Storing interaction with query: '{query[:50]}...' and response: '{response[:50]}...'")
            # Create a semantic vector for the interaction.  Consider combining query + response.
            interaction_text = f"{query} {response}"
            semantic_vector = await self.vector_operations.create_semantic_vector(interaction_text)

            memory_id = f"mem_{uuid.uuid4().hex}"
            metadata = {
                "content": interaction_text,
                "memory_type": "EPISODIC",
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "window_id": window_id,
                "source": "user_interaction"  # Indicate the source of this memory
            }

            await self.pinecone_service.create_memory(
                memory_id=memory_id,
                vector=semantic_vector,
                metadata=metadata
            )
            logger.info(f"Episodic memory stored successfully: {memory_id}")

            # Construct and return the Memory object (important for consistency)
            return Memory(
                id=memory_id,
                content=interaction_text,
                memory_type=MemoryType.EPISODIC,
                created_at=metadata["created_at"], # Use consistent format
                metadata=metadata,
                window_id=window_id,
                semantic_vector=semantic_vector,
            )

        except Exception as e:
            logger.error(f"Failed to store interaction: {e}", exc_info=True)
            raise MemoryOperationError("store_interaction", str(e))

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
            window_id=window_id,
        )
        return memory_id

    async def health_check(self):
        """Checks the health of the memory system."""
        return {"status": "healthy"}