# memory.py
from datetime import datetime
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


# Updated imports:
from api.utils.config import Settings
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.interfaces.vector_operations import VectorOperations
from api.core.memory.models import Memory, MemoryType, CreateMemoryRequest, MemoryResponse, QueryRequest, QueryResponse, RequestMetadata, OperationType, ErrorDetail
from api.core.memory.exceptions import MemoryOperationError

logger = logging.getLogger(__name__)

# Constants for retry mechanism
RETRY_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 1

class MemorySystem:
    def __init__(
        self,
        pinecone_service: MemoryService,
        vector_operations: VectorOperations,
        settings: Any
    ):
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        self.settings = settings or Settings()
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the memory system and its components."""
        try:
            # Verify vector operations initialization
            if hasattr(self.vector_operations, 'initialize'):
                await self.vector_operations.initialize()

            # Verify Pinecone initialization
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
                        "created_at": datetime.utcnow().isoformat(),
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
    
    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_fixed(RETRY_WAIT_SECONDS),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def create_memory_from_request(
        self,
        request: CreateMemoryRequest
    ) -> MemoryResponse:
        """Create a memory from a validated request."""
        try:
            request.request_metadata = RequestMetadata(
                operation_type=OperationType.STORE,
                window_id=request.window_id
            )

            logger.info(f"Creating memory from request: {request}")

            # Generate memory ID with 'mem_' prefix
            memory_id = f"mem_{uuid.uuid4()}"
            created_at_timestamp = int(datetime.utcnow().timestamp()) # Get integer timestamp NOW

            try:
                # Generate semantic vector
                semantic_vector = await self.vector_operations.create_semantic_vector(request.content)

                if hasattr(semantic_vector, 'tolist'):
                    semantic_vector = semantic_vector.tolist()

                # Prepare metadata - Use the INTEGER timestamp here.
                full_metadata = {
                    "content": request.content,
                    "created_at": created_at_timestamp,  # <--- Store as INTEGER timestamp
                    "memory_type": request.memory_type.value,
                    **(request.metadata or {}),
                }

                if request.window_id:
                    full_metadata["window_id"] = request.window_id

                # Create memory object -- The string conversion happens HERE
                memory = Memory(
                    id=memory_id,
                    content=request.content,
                    memory_type=request.memory_type,
                    semantic_vector=semantic_vector,
                    created_at=datetime.fromtimestamp(created_at_timestamp, tz=timezone.utc).isoformat() + "Z", #to string
                    metadata=full_metadata,
                    window_id=request.window_id
                )

                # Store directly in Pinecone
                success = await self.pinecone_service.create_memory(
                    memory_id=memory.id,
                    vector=semantic_vector,
                    metadata=memory.metadata, #metadata passed to pinecone
                )

                if not success:
                    raise MemoryOperationError("Failed to store memory in vector database")

                logger.info(f"Successfully created memory with ID: {memory_id}")

                return MemoryResponse.from_memory(memory) # Use class method for conversion


            except Exception as e:
                logger.error(f"Error during memory creation process: {e}")
                raise MemoryOperationError(f"Failed to add memory during vector creation or upsertion: {e}")

        except Exception as e:  # This outer exception seems redundant
            logger.error(f"Error creating memory from request: {e}", exc_info=True)
            return MemoryResponse( #I dont think you ever hit this because of the inner exception.
                error=ErrorDetail(
                    code="MEMORY_CREATION_FAILED",
                    message=str(e),
                    trace_id=request.request_metadata.trace_id,
                    operation_type=OperationType.STORE
                )
            )

    async def query_memories(
        self,
        request: QueryRequest
    ) -> QueryResponse:
        try:
            logger.info(f"Starting memory query with request: {request}")
            query_vector = await self.vector_operations.create_semantic_vector(
                request.prompt
            )
            logger.info(f"Query vector: {query_vector}")
            # Build filter for semantic memories
            semantic_filter = {
                "memory_type": "SEMANTIC",
                **({"window_id": request.window_id} if request.window_id else {})
            }
    
            results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=request.top_k,
                filter=semantic_filter,
            )
            logger.info(f"Received results from pinecone service: {results[:2]}")
            logger.info(f"Query returned {len(results)} raw results")
            for i, (memory_data, score) in enumerate(results):
                logger.info(f"Memory {i} - Type: {type(memory_data)}, Score: {score}")
                logger.info(f"Memory {i} content: {str(memory_data)[:100]}")

        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            raise MemoryOperationError(f"Failed to query memories: {str(e)}")
        
        try:
            # Generate query vector
            query_vector = await self.vector_operations.create_semantic_vector(
                request.prompt
            )

            # Get more initial results to account for filtering
            initial_top_k = min(request.top_k * 2, 20)

            semantic_filter = {
                "memory_type": "SEMANTIC",
                **({"window_id": request.window_id} if request.window_id else {})
            }

            # Query semantic memories
            semantic_results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=initial_top_k,
                filter=semantic_filter,
            )

            # Process and filter results
            matches = []
            similarity_scores = []
            current_time = datetime.utcnow()

            for memory_data, similarity_score in semantic_results:
                # Basic similarity threshold
                if similarity_score < 0.6:
                    continue

                try:
                    # Apply gentle time weighting
                    consolidated_at = datetime.fromisoformat(
                        memory_data["metadata"].get("consolidated_at", "").replace("Z", "+00:00")
                    )
                    age_days = (current_time - consolidated_at).days
                    time_weight = 0.7 + (0.3 / (1 + math.exp(age_days / 180 - 2)))
                
                    # Combine scores (80% similarity, 20% time)
                    final_score = (similarity_score * 0.8) + (time_weight * 0.2)
                
                    # Add source information
                    source_info = ""
                    if source_memories := memory_data["metadata"].get("source_memories"):
                        source_info = f"\nDerived from {len(source_memories)} related interactions."

                    memory_response = MemoryResponse(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"] + source_info,
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                        created_at=memory_data["metadata"]["created_at"],
                        metadata={
                            **memory_data["metadata"],
                            "similarity_score": similarity_score,
                        },
                        window_id=memory_data["metadata"].get("window_id"),
                        semantic_vector=memory_data["vector"]
                    )
                    matches.append(memory_response)
                    similarity_scores.append(final_score)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing timestamp for memory {memory_data['id']}: {e}")
                    continue

            # Sort by final score and limit to requested number
            sorted_results = sorted(zip(matches, similarity_scores), 
                              key=lambda x: x[1], 
                              reverse=True)[:request.top_k]
            matches, similarity_scores = zip(*sorted_results) if sorted_results else ([], [])

            return QueryResponse(
                matches=list(matches),
                similarity_scores=list(similarity_scores),
            )

        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            raise MemoryOperationError(f"Failed to query memories: {str(e)}")
    
    async def store_interaction(
        self,
        query: str,
        response: str,
        window_id: Optional[str] = None
    ) -> MemoryResponse:
        """Store query and response as an episodic memory with enhanced error handling."""
        trace_id = f"store_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        logger.info(f"[{trace_id}] Starting interaction storage")
    
        try:
            # Format and validate interaction content
            if not query or not response:
                raise ValueError("Query and response must not be empty")
            
            interaction_content = f"Query: {query}\nResponse: {response}"
            if len(interaction_content) > 32000:  # Add reasonable limit
                logger.warning(f"[{trace_id}] Large interaction content: {len(interaction_content)} chars")
    
            # Create memory request
            try:
                request = CreateMemoryRequest(
                    content=interaction_content,
                    memory_type=MemoryType.EPISODIC,
                    metadata={
                        "interaction_type": "query_response",
                        "query": query,
                        "response": response,
                        "trace_id": trace_id,
                        "stored_at": datetime.utcnow().isoformat()
                    },
                    window_id=window_id
                )
            except ValidationError as e:
                logger.error(f"[{trace_id}] Failed to create memory request: {str(e)}")
                raise MemoryOperationError(f"Invalid memory request: {str(e)}")
    
            # Store the memory
            try:
                result = await self.create_memory_from_request(request)
                logger.info(f"[{trace_id}] Successfully stored interaction with ID: {result.id}")
                return result
            except Exception as e:
                logger.error(f"[{trace_id}] Failed to store memory: {str(e)}", exc_info=True)
                raise MemoryOperationError(f"Failed to store interaction: {str(e)}")
            
        except Exception as e:
            logger.error(f"[{trace_id}] Unexpected error in store_interaction: {str(e)}", exc_info=True)
            raise MemoryOperationError(f"Unexpected error storing interaction: {str(e)}")

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