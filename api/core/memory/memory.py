# api/core/memory/memory.py
from datetime import datetime, timezone, timedelta
import uuid
from typing import List, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
import asyncio
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import math

# Corrected imports:  Import from the correct locations
from api.utils.config import get_settings, Settings
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.interfaces.vector_operations import VectorOperations
from api.core.memory.models import (Memory, MemoryType, CreateMemoryRequest,
                                      MemoryResponse, QueryRequest, QueryResponse,
                                      RequestMetadata, OperationType, ErrorDetail)
from api.core.memory.exceptions import MemoryOperationError
from api.utils.utils import normalize_timestamp  # Import the helper


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
                        **(req.metadata or {}),  # Use .get() for optional fields
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
                created_at=created_at,  # Pass created_at string
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
            # we are now getting a datetime object from pinecone service
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
            logger.error(f"Error deleting memory: {e}")
            return False

    async def query_memories(self, request: QueryRequest) -> QueryResponse:
        """Query memories based on the given request, with filtering and sorting."""
        try:
            logger.info(f"Starting memory query with request: {request}")
            query_vector = await self.vector_operations.create_semantic_vector(request.prompt)
            logger.info(f"Query vector generated (first 10 elements): {query_vector[:10]}")

            pinecone_filter = {"memory_type": "SEMANTIC"}
            if request.window_id:
                pinecone_filter["window_id"] = request.window_id
            logger.info(f"Querying Pinecone with filter: {pinecone_filter}")

            results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=request.top_k,
                filter=pinecone_filter,
                include_metadata=True
            )
            logger.info(f"Received {len(results)} raw results from Pinecone.")

            matches: List[MemoryResponse] = []
            similarity_scores: List[float] = []
            current_time = datetime.utcnow()

            for memory_data, similarity_score in results:
                logger.debug(f"Processing memory data: {memory_data}")
                try:
                    if similarity_score < 0.6:
                        logger.info(f"Skipping memory {memory_data['id']} due to low similarity score: {similarity_score}")
                        continue

                    created_at = memory_data["metadata"].get("created_at")  # Already a datetime object!
                    if not created_at:
                        logger.warning(f"Skipping memory {memory_data['id']} due to missing created_at")
                        continue

                    # No more type checking/string parsing

                    age_days = (current_time - created_at).total_seconds() / (60*60*24)
                    time_weight = 0.7 + (0.3 / (1 + math.exp(age_days/180 - 2)))

                    final_score = (similarity_score * 0.8) + (time_weight * 0.2)
                    logger.info(f"Memory ID {memory_data['id']}: Raw Score={similarity_score:.3f}, Time Weight={time_weight:.3f}, Final Score={final_score:.3f}")

                    memory_response = MemoryResponse(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                        created_at=memory_data["metadata"]["created_at"].isoformat() + "Z",  # Format as ISO string with Z here
                        metadata=memory_data["metadata"],
                        window_id=memory_data["metadata"].get("window_id"),
                        semantic_vector=memory_data["vector"],
                    )
                    matches.append(memory_response)
                    similarity_scores.append(final_score)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing timestamp for memory {memory_data['id']}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing memory {memory_data['id']}: {e}", exc_info=True)
                    continue

            sorted_results = sorted(zip(matches, similarity_scores), key=lambda x: x[1], reverse=True)[:request.top_k]
            matches, similarity_scores = zip(*sorted_results) if sorted_results else ([], [])

            return QueryResponse(
                matches=list(matches),
                similarity_scores=list(similarity_scores),
            )

        except Exception as e:
            logger.error(f"Error querying memories: {e}", exc_info=True)
            raise MemoryOperationError(f"Failed to query memories: {str(e)}")

    async def _check_recent_duplicate(self, content: str, window_minutes: int = 30) -> bool:
        """Improved duplicate detection."""
        try:
            # Normalize the content for comparison (lowercase, remove extra spaces)
            normalized_content = ' '.join(content.lower().split())

            # Extract core message (remove "User:" and "Assistant:" prefixes)
            if "user:" in normalized_content:  # Use lowercase
                normalized_content = normalized_content.split("user:", 1)[1]
            if "assistant:" in normalized_content:  # Use lowercase
                normalized_content = normalized_content.split("assistant:", 1)[0]

            vector = await self.vector_operations.create_semantic_vector(normalized_content)

            recent_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            # Convert recent_time to a Unix timestamp (seconds since epoch)
            recent_timestamp = int(recent_time.timestamp())

            results = await self.pinecone_service.query_memories(
                query_vector=vector,
                top_k=5,  # Check more potential matches
                filter={
                    "memory_type": "EPISODIC",
                    "created_at": {"$gte": recent_timestamp}  # Use the integer timestamp
                },
                include_metadata=True
            )

            for memory_data, score in results:
                if score > self.settings.duplicate_similarity_threshold:
                    logger.info(f"Duplicate found: {memory_data['id']} with score {score}")
                    return True  # Duplicate found
            return False  # No duplicates found

        except Exception as e:
            logger.warning(f"Duplicate check failed, proceeding with storage: {e}")
            return False  # Assume not a duplicate on error
    async def store_interaction_enhanced(self, query: str, response: str, window_id: Optional[str] = None) -> Memory:
        """Stores a user interaction (query + response) as a new episodic memory."""
        try:
            logger.info(f"Storing interaction with query: '{query[:50]}...' and response: '{response[:50]}...'")
            # Combine query and response for embedding.  Correct format.
            interaction_text = f"User: {query}\nAssistant: {response}"

            # Check for duplicates *before* creating the memory object
            if await self._check_recent_duplicate(interaction_text):
                logger.warning("Duplicate interaction detected. Skipping storage.")
                return None  # Or raise an exception, depending on desired behavior


            semantic_vector = await self.vector_operations.create_semantic_vector(interaction_text)

            memory_id = f"mem_{uuid.uuid4().hex}"
            created_at = datetime.now(timezone.utc).isoformat() + "Z"  # Use consistent format here
            metadata = {
                "content": interaction_text,  # Store the COMBINED text
                "memory_type": "EPISODIC",
                "created_at": created_at, # Use consistent format here
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
                created_at= datetime.fromisoformat(normalize_timestamp(created_at)), # Pass datetime object
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
        memory_id = await self.add_memory( #add_memory to be deprecated
            content=content,
            memory_type=MemoryType.EPISODIC,
            metadata={"interaction": True},
            window_id=window_id,
        )
        return memory_id

    async def health_check(self):
        """Checks the health of the memory system."""
        return {"status": "healthy"} 