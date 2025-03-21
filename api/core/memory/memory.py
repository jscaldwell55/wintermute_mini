# memory.py
from datetime import datetime, timezone, timedelta
import uuid
from typing import List, Dict, Optional, Any, Tuple, Union
import logging
import math
import asyncio
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import re
from ortools.linear_solver import pywraplp

# Corrected imports: Import from the correct locations
from api.utils.config import get_settings, Settings
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.vector.vector_operations import VectorOperations
from api.core.memory.models import (Memory, MemoryType, CreateMemoryRequest,
                                      MemoryResponse, QueryRequest, QueryResponse,
                                      RequestMetadata, OperationType, ErrorDetail)
from api.core.memory.exceptions import MemoryOperationError
from api.utils.utils import normalize_timestamp
from api.utils.llm_service import LLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(
        self,
        pinecone_service: MemoryService,
        vector_operations: VectorOperations,
        llm_service: LLMService,
        settings: Optional[Settings] = None  # Allow optional settings
    ):
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        self.llm_service = llm_service
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
                        "created_at": datetime.now(timezone.utc).isoformat(),
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
            created_at = datetime.now(timezone.utc).isoformat() + "Z"

            semantic_vector = await self.vector_operations.create_semantic_vector(request.content)

            metadata = {
                "content": request.content,
                "created_at": created_at,
                "memory_type": request.memory_type.value,
                **(request.metadata or {}),  # Include any extra metadata from the request
            }

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

            if success and hasattr(self, 'memory_graph'):
                self.memory_graph.add_memory_node(memory)

                # Also add relationships to existing memories
                if hasattr(self, 'relationship_detector'):
                    # Get a sample of existing memories as candidates
                    candidate_memories = await self.get_sample_memories(50, exclude_id=memory.id)
                    relationships_by_type = await self.relationship_detector.analyze_memory_relationships(
                        memory, candidate_memories
                    )

                    # Add all detected relationships
                    for rel_type, rel_list in relationships_by_type.items():
                        for related_memory, strength in rel_list:
                            self.memory_graph.add_relationship(
                                source_id=memory.id,
                                target_id=related_memory.id,
                                rel_type=rel_type,
                                weight=strength
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

    # Add to MemorySystem class in memory_system.py
    async def get_sample_memories(self, sample_size: int, exclude_id: Optional[str] = None) -> List[Memory]:
        """Get a random sample of memories for relationship detection."""
        try:
            # Create dummy vector for query
            dummy_vector = [0.0] * self.pinecone_service.embedding_dimension

            # Query for memories
            results = await self.pinecone_service.query_memories(
                query_vector=dummy_vector,
                top_k=sample_size * 2,  # Request more to account for filtering
                filter={},  # No filter to get all memory types
                include_metadata=True
            )

            # Convert to Memory objects
            memories = []
            for memory_data, _ in results:
                if exclude_id and memory_data["id"] == exclude_id:
                    continue

                memory = Memory(
                    id=memory_data["id"],
                    content=memory_data["metadata"]["content"],
                    memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                    created_at=memory_data["metadata"]["created_at"],
                    metadata=memory_data["metadata"],
                    semantic_vector=memory_data.get("vector")
                )
                memories.append(memory)

                if len(memories) >= sample_size:
                    break

            return memories

        except Exception as e:
            logger.error(f"Error getting sample memories: {e}")
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID and remove it from keyword index."""
        try:
            logger.info(f"Deleting memory with ID: {memory_id}")


            # Then delete from Pinecone
            result = await self.pinecone_service.delete_memory(memory_id)
            return result
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False

    async def get_memories_by_window_id(self, window_id: str) -> List[Memory]:
        """Retrieve memories by window ID."""
        try:
            logger.info(f"Retrieving memories by window ID: {window_id}")
            memories = await self.pinecone_service.get_memories_by_window_id(window_id)
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories by window ID: {e}")
            return []

    async def query_memories(self, request: QueryRequest) -> QueryResponse:
        """Query memories based on the given request, with hybrid vector and keyword search."""
        try:
            # Store the current query for time-based filtering
            self.current_query = request.prompt
            logger.info(f"Starting memory query with request: {request}")

            # Set a limit on processing attempts to avoid infinite loops
            max_attempts = 3
            attempt_count = 0

            # Create a request-specific embedding cache
            # This will be used for this query only
            embedding_cache = {}

            # Function to get embedding with caching
            async def get_cached_embedding(text: str) -> List[float]:
                if text in embedding_cache:
                    logger.info(f"Using cached embedding for: {text[:50]}...")
                    return embedding_cache[text]

                logger.info(f"Generating new embedding for: {text[:50]}...")
                embedding = await self.vector_operations.create_semantic_vector(text)
                embedding_cache[text] = embedding
                return embedding

            # Generate query vector (using cache)
            query_vector = await get_cached_embedding(request.prompt)
            logger.info(f"Query vector generated (first 5 elements): {query_vector[:5]}")


            # Determine memory type filter
            memory_type_value = request.memory_type.value if hasattr(request, 'memory_type') and request.memory_type else None
            logger.info(f"Using memory type filter: {memory_type_value}")

            # Execute vector and keyword searches in parallel
            tasks = []

            # Add vector search task
            if not hasattr(request, 'memory_type') or request.memory_type is None:
                # Default case - query all types
                pinecone_filter = {}
                logger.info(f"Querying ALL memory types with filter: {pinecone_filter}")

                vector_task = asyncio.create_task(
                    self.pinecone_service.query_memories(
                        query_vector=query_vector,
                        top_k=request.top_k,
                        filter=pinecone_filter,
                        include_metadata=True
                    )
                )
                tasks.append(("vector_all", vector_task))

            elif request.memory_type == MemoryType.SEMANTIC:
                # For semantic memories
                pinecone_filter = {"memory_type": "SEMANTIC"}
                logger.info(f"Querying SEMANTIC memories with filter: {pinecone_filter}")

                vector_task = asyncio.create_task(
                    self.pinecone_service.query_memories(
                        query_vector=query_vector,
                        top_k=request.top_k,
                        filter=pinecone_filter,
                        include_metadata=True
                    )
                )
                tasks.append(("vector_semantic", vector_task))

            elif request.memory_type == MemoryType.EPISODIC:
                # For episodic memories with time filtering
                seven_days_ago = datetime.now(timezone.utc) - timedelta(days=14)
                seven_days_ago_timestamp = int(seven_days_ago.timestamp())

                pinecone_filter = {
                    "memory_type": "EPISODIC",
                    "created_at_unix": {  # Corrected line: using created_at_unix
                        "$gte": seven_days_ago_timestamp
                    }
                }
                logger.info(f"Querying EPISODIC memories with filter: {pinecone_filter}")

                vector_task = asyncio.create_task(
                    self.pinecone_service.query_memories(
                        query_vector=query_vector,
                        top_k=request.top_k,
                        filter=pinecone_filter,
                        include_metadata=True
                    )
                )
                tasks.append(("vector_episodic", vector_task))

                # Also prepare fallback task if needed
                if request.window_id:
                    fallback_filter = {
                        "memory_type": "EPISODIC",
                        "created_at_unix": {  # Corrected line: using created_at_unix
                            "$gte": seven_days_ago_timestamp
                        },
                        "window_id": request.window_id
                    }
                    fallback_task = asyncio.create_task(
                        self.pinecone_service.query_memories(
                            query_vector=query_vector,
                            top_k=request.top_k,
                            filter=fallback_filter,
                            include_metadata=True
                        )
                    )
                    tasks.append(("vector_episodic_fallback", fallback_task))

            elif request.memory_type == MemoryType.LEARNED:
                # For learned memories
                pinecone_filter = {"memory_type": "LEARNED"}
                logger.info(f"Querying LEARNED memories with filter: {pinecone_filter}")

                vector_task = asyncio.create_task(
                    self.pinecone_service.query_memories(
                        query_vector=query_vector,
                        top_k=request.top_k,
                        filter=pinecone_filter,
                        include_metadata=True
                    )
                )
                tasks.append(("vector_learned", vector_task))

            else:
                # Fallback for unknown types
                pinecone_filter = {}
                logger.info(f"Querying with unknown memory type, using ALL types with filter: {pinecone_filter}")

                vector_task = asyncio.create_task(
                    self.pinecone_service.query_memories(
                        query_vector=query_vector,
                        top_k=request.top_k,
                        filter=pinecone_filter,
                        include_metadata=True
                    )
                )
                tasks.append(("vector_unknown", vector_task))


            # Wait for all tasks to complete
            task_results = {}
            for name, task in tasks:
                try:
                    task_results[name] = await task
                except Exception as e:
                    logger.error(f"Task {name} failed: {e}")
                    task_results[name] = []

            # Process vector results
            vector_results = []

            # First check main vector results
            if "vector_all" in task_results:
                vector_results = task_results["vector_all"]
            elif "vector_semantic" in task_results:
                vector_results = task_results["vector_semantic"]
            elif "vector_episodic" in task_results:
                vector_results = task_results["vector_episodic"]

                # If no results and we have a fallback, use that
                if len(vector_results) == 0 and "vector_episodic_fallback" in task_results:
                    vector_results = task_results["vector_episodic_fallback"]
            elif "vector_learned" in task_results:
                vector_results = task_results["vector_learned"]
            elif "vector_unknown" in task_results:
                vector_results = task_results["vector_unknown"]

            logger.info(f"Received {len(vector_results)} raw results from vector search.")



            # Just use vector results directly since keyword search is removed
            combined_results = [(memory_data, score) for memory_data, score in vector_results]


            matches = []
            similarity_scores = []
            current_time = datetime.now(timezone.utc)


            for memory_data, similarity_score in combined_results:
                logger.debug(f"Processing combined result: {memory_data}")
                try:
                    # Basic similarity threshold
                    if similarity_score < self.settings.min_similarity_threshold:
                        logger.info(f"Skipping memory {memory_data['id']} due to low similarity score: {similarity_score}")
                        continue

                    created_at_raw = memory_data["metadata"].get("created_at")
                    if not created_at_raw:
                        logger.warning(f"Skipping memory {memory_data['id']} due to missing created_at")
                        continue

                    # Convert string timestamp to datetime if needed
                    if isinstance(created_at_raw, str):
                        created_at = normalize_timestamp(created_at_raw)
                    else:
                        created_at = created_at_raw

                    memory_type = memory_data["metadata"].get("memory_type", "UNKNOWN")
                    final_score = similarity_score  # Default score is just similarity

                    # Apply type-specific scoring adjustments with the new weighting system
                    if memory_type == "EPISODIC":
                        # Ensure compatible timezone handling
                        if current_time.tzinfo is None and created_at.tzinfo is not None:
                            # Make current_time timezone-aware if it isn't already
                            current_time = current_time.replace(tzinfo=timezone.utc)
                        elif current_time.tzinfo is not None and created_at.tzinfo is None:
                            # Make created_at timezone-aware if it isn't already
                            created_at = created_at.replace(tzinfo=timezone.utc)

                        # Calculate age in hours for episodic memories
                        try:
                            age_hours = (current_time - created_at).total_seconds() / (60*60)
                        except Exception as e:
                            logger.warning(f"Error calculating age for memory {memory_data['id']}: {e}")
                            # Default to recent memory (1 hour old) to avoid filtering
                            age_hours = 1.0

                        # Use bell curve recency scoring if enabled, otherwise use original method
                        use_bell_curve = getattr(self.settings, 'episodic_bell_curve_enabled', True)

                        if use_bell_curve:
                            # Apply bell curve scoring
                            recency_score = self._calculate_bell_curve_recency(age_hours)
                        else:
                            # Original method (linear then exponential decay)
                            if age_hours <= self.settings.episodic_recent_hours:
                                recency_score = 1.0 - (age_hours / self.settings.episodic_recent_hours) * 0.3
                            else:
                                max_age_hours = self.settings.episodic_max_age_days * 24
                                relative_age = (age_hours - self.settings.episodic_recent_hours) / (max_age_hours - self.settings.episodic_recent_hours)
                                recency_score = 0.7 * (0.1/0.7) ** relative_age

                        # Ensure recency score is between 0-1
                        recency_score = max(0.0, min(1.0, recency_score))

                        # Calculate combined score: (1-w)*similarity + w*recency
                        relevance_weight = 1 - self.settings.episodic_recency_weight
                        combined_score = (
                            relevance_weight * similarity_score +
                            self.settings.episodic_recency_weight * recency_score
                        )

                        # Apply memory type weight
                        final_score = combined_score * self.settings.episodic_memory_weight

                        logger.info(f"Memory ID {memory_data['id']} (EPISODIC): Raw={similarity_score:.3f}, "
                            f"Age={age_hours:.1f}h, Recency={recency_score:.3f}, Final={final_score:.3f}")

                    elif memory_type == "SEMANTIC":
                        # Extract creation timestamp
                        created_at_raw = memory_data["metadata"].get("created_at")
                        if isinstance(created_at_raw, str):
                            created_at = normalize_timestamp(created_at_raw)
                        else:
                            created_at = created_at_raw

                        # Add this timezone handling code here
                        if current_time.tzinfo is None and created_at.tzinfo is not None:
                            current_time = current_time.replace(tzinfo=timezone.utc)
                        elif current_time.tzinfo is not None and created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)

                        # Calculate age in days
                        age_days = (current_time - created_at).total_seconds() / (86400)  # Seconds in a day

                        # Calculate recency score (using a slower decay than episodic)
                        recency_score = max(0.4, 1.0 - (math.log(1 + age_days) / 10))

                        # Combine scores (80% relevance, 20% recency)
                        semantic_recency_weight = 0.2  # Could add this to settings if you want it configurable
                        relevance_weight = 1 - semantic_recency_weight
                        combined_score = (
                            relevance_weight * similarity_score +
                            semantic_recency_weight * recency_score
                        )

                        # Apply memory type weight
                        final_score = combined_score * self.settings.semantic_memory_weight

                        logger.info(f"Memory ID {memory_data['id']} (SEMANTIC): Raw={similarity_score:.3f}, "
                            f"Age={age_days:.1f}d, Recency={recency_score:.3f}, Final={final_score:.3f}")


                    elif memory_type == "LEARNED":
                        confidence = memory_data["metadata"].get("confidence", 0.5)  # Default confidence if not present
                        combined_score = (similarity_score * 0.8) + (confidence * 0.2)  # Weight by confidence
                        final_score = combined_score * self.settings.learned_memory_weight
                        logger.info(f"Memory ID {memory_data['id']} (LEARNED): Raw={similarity_score:.3f}, "
                            f"Confidence={confidence:.2f}, Final={final_score:.3f}")

                    memory_response = MemoryResponse(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                        created_at=created_at.isoformat() + "Z",  # Format as ISO string with Z here
                        metadata=memory_data["metadata"],
                        window_id=memory_data["metadata"].get("window_id"),
                        semantic_vector=memory_data.get("vector"),
                    )
                    memory_response.time_ago = self._format_time_ago(created_at)
                    matches.append(memory_response)
                    similarity_scores.append(final_score)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing timestamp for memory {memory_data['id']}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing memory {memory_data['id']}: {e}", exc_info=True)
                    continue

            # Sort results by final score
            if matches and similarity_scores:
                # Create pairs of memories and scores
                memory_scores = list(zip(matches, similarity_scores))

                # Apply time-based weighting
                memory_scores = self.apply_time_weighting(memory_scores)

                # Extract the top_k results
                memory_scores = memory_scores[:request.top_k]

                # Unzip the results
                matches, similarity_scores = zip(*memory_scores)
            else:
                matches, similarity_scores = [], []

            return QueryResponse(
                matches=list(matches),
                similarity_scores=list(similarity_scores),
            )

        except Exception as e:
            logger.error(f"Error querying memories: {e}", exc_info=True)
            raise MemoryOperationError(f"Failed to query memories: {str(e)}")



    def _get_past_weekday_date(self, base_time: datetime, weekday_name: str) -> datetime:
            """Helper to get the datetime for the most recent past weekday (e.g., last Monday)."""
            weekday_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6
            }
            weekday_num = weekday_map[weekday_name]
            days_diff = (base_time.weekday() - weekday_num) % 7
            target_date = base_time - timedelta(days=days_diff)
            return target_date.replace(hour=0, minute=0, second=0, microsecond=0) # Set time to midnight

    async def process_temporal_query(self, query: str, window_id: Optional[str] = None) -> Dict[str, str]:
            """Process queries about past conversations with temporal references."""
            temporal_context = None  # Initialize temporal_context here to avoid NameError

            # Enhanced regex patterns for temporal expressions with better coverage
            temporal_patterns = [
                # Simple temporal references
                r"(?:what|when|how) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (\d+ days? ago)",
                r"(?:what|when|how) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (yesterday)",
                r"(?:what|when) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (last week)",
                r"(?:what) (?:happened|occurred|took place) (yesterday|last week|\d+ days? ago)",
                r"(?:what|remember) (?:we've|we have|have we) (?:talk|discuss|chat)(?:ed|) (?:about)? (?:the past|in the past|over the past) (\d+ days?)",

                # Time of day references
                r"(?:what|when|how) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (this morning)",
                r"(?:what|when|how) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (today)",
                r"(?:what) (?:have we discussed|did we discuss) (today|this morning|this afternoon|this evening)",
                r"(?:what) (?:happened|occurred|took place) (this morning|today|this afternoon)",

                # More specific times
                r"(?:what|when|how) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (yesterday) (morning|afternoon|evening|night)",
                r"(?:what|when|how) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (last week)",
                r"(?:what|when|how) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (?:on) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)",

                # Exact time references
                r"(?:what|when|how) (?:did|have) (?:we|you|I) (?:talk|discuss|chat|mention) (?:about)? (?:at|around|about) (\d{1,2}[:.]\d{2}) *(am|pm)",

                # Simple "What did we discuss yesterday?" pattern
                r"what did we discuss (yesterday)"
            ]

            # Check for temporal patterns
            matched_expr = None
            for pattern in temporal_patterns:
                match = re.search(pattern, query.lower(), re.IGNORECASE)
                if match:
                    matched_expr = match.group(1)
                    # If we have a second capturing group (e.g., "morning" in "yesterday morning")
                    if len(match.groups()) > 1 and match.group(2):
                        matched_expr += f" {match.group(2)}"
                    break

            if not matched_expr:
                # No temporal expression found, return empty so normal processing continues
                return {}

            # Parse the temporal expression
            start_time, end_time = await self.parse_time_expression(matched_expr)
            logger.info(f"Parsed start_time: {start_time.isoformat()}, end_time: {end_time.isoformat()}") # ADD LOGGING

            # Add buffer to the time window (e.g., ±1 hours for more precise queries)
            buffer = timedelta(hours=1)
            start_time = start_time - buffer
            end_time = end_time + buffer
            logger.info(f"Timeframe with buffer: start_time: {start_time.isoformat()}, end_time: {end_time.isoformat()}") # ADD LOGGING

            logger.info(f"Temporal query detected: '{matched_expr}' - Timeframe with buffer: {start_time} to {end_time}")

            # Always calculate Unix timestamps for consistency
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            logger.info(f"Unix timestamps: start_timestamp: {start_timestamp}, end_timestamp: {end_timestamp}") # ADD LOGGING

            # Generate temporal context for the prompt template
            temporal_context = f"Note: This query is specifically about conversations from {matched_expr}, between {start_time.strftime('%Y-%m-%d %H:%M')} and {end_time.strftime('%Y-%m-%d %H:%M')}."

            # Create primary filter WITHOUT window_id for broader search
            filter_dict = {
                "memory_type": "EPISODIC",
                "created_at_unix": {
                    "$gte": start_timestamp,
                    "$lte": end_timestamp
                }
            }
            logger.info(f"Initial filter_dict: {filter_dict}") # ADD LOGGING

            # Add time-of-day filters if present in the query
            if "morning" in matched_expr.lower():
                filter_dict["is_morning"] = True
                logger.info(f"Added 'is_morning' to filter: {filter_dict}") # ADD LOGGING
            elif "afternoon" in matched_expr.lower():
                filter_dict["is_afternoon"] = True
                logger.info(f"Added 'is_afternoon' to filter: {filter_dict}") # ADD LOGGING
            elif "evening" in matched_expr.lower() or "night" in matched_expr.lower():
                filter_dict["is_evening"] = True
                logger.info(f"Added 'is_afternoon' to filter: {filter_dict}") # ADD LOGGING

            # For specific day of week queries
            days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            for day in days_of_week:
                if day in matched_expr.lower():
                    filter_dict["day_of_week"] = day
                    logger.info(f"Added 'day_of_week' ({day}) to filter: {filter_dict}") # ADD LOGGING
                    break

            # Log the actual filter being used
            logger.info(f"Using temporal query filter: {filter_dict}")

            # Perform vector search with the time-filtered query
            memories = await self.query_by_timeframe_enhanced(
                query=query,
                filter_dict=filter_dict,
                top_k=10
            )

            # Unzip the results returned from query_by_timeframe_enhanced
            episodic_memories, episodic_scores = [], []
            if memories:
                episodic_memories, episodic_scores = zip(*memories)

            # If no results, progressively relax constraints
            if not episodic_memories:
                logger.info(f"No memories found with primary filter, trying expanded time range")

                # 1. Double the time window
                time_span = end_timestamp - start_timestamp
                expanded_filter = {
                    "memory_type": "EPISODIC",
                    "created_at_unix": {
                        "$gte": start_timestamp - time_span//2,
                        "$lte": end_timestamp + time_span//2
                    }
                }

                # Remove time-of-day constraints but keep day of week if present
                if "is_morning" in expanded_filter:
                    del expanded_filter["is_morning"]
                    logger.info(f"Removed 'is_morning' from expanded_filter") # ADD LOGGING
                if "is_afternoon" in expanded_filter:
                    del expanded_filter["is_afternoon"]
                    logger.info(f"Removed 'is_afternoon' from expanded_filter") # ADD LOGGING
                if "is_evening" in expanded_filter:
                    del expanded_filter["is_evening"]
                    logger.info(f"Removed 'is_evening' from expanded_filter") # ADD LOGGING

                logger.info(f"Using expanded filter: {expanded_filter}")

                memories = await self.query_by_timeframe_enhanced(
                    query=query,
                    filter_dict=expanded_filter,
                    top_k=10
                )
                # Unzip the results returned from query_by_timeframe_enhanced
                episodic_memories, episodic_scores = [], []
                if memories:
                    episodic_memories, episodic_scores = zip(*memories)


            # For fallback approach, use a very liberal timeframe
            if not episodic_memories:
                logger.info(f"No memories found with expanded filter, using fallback approach")

                # For fallback, use a very wide time range (7 days before to 1 day after the specified period)
                fallback_filter = {
                    "memory_type": "EPISODIC",
                    "created_at_unix": {
                        "$gte": start_timestamp - (7 * 24 * 3600),  # 7 days before
                        "$lte": end_timestamp + (24 * 3600)         # 1 day after
                    }
                }

                # Remove all time constraints
                if "day_of_week" in fallback_filter:
                    del fallback_filter["day_of_week"]
                    logger.info(f"Removed 'day_of_week' from fallback_filter") # ADD LOGGING

                logger.info(f"Using fallback filter: {fallback_filter}")

                memories = await self.query_by_timeframe_enhanced(
                    query=query,
                    filter_dict=fallback_filter,
                    top_k=10
                )
                # Unzip the results returned from query_by_timeframe_enhanced
                episodic_memories, episodic_scores = [], []
                if memories:
                    episodic_memories, episodic_scores = zip(*memories)


            if not episodic_memories:
                # Return a polite "no memories found" response
                return {"episodic": f"I don't recall discussing anything {matched_expr}. Is there something else I can help you with?"}

            # Process memories for summarization
            episodic_memories_with_scores = list(zip(episodic_memories, episodic_scores))

            # Use the existing memory summarization agent
            summary = await self.memory_summarization_agent(
                query=query,
                semantic_memories=[],
                episodic_memories=episodic_memories_with_scores,
                learned_memories=[],
                time_expression=matched_expr,
                temporal_context=temporal_context,
                window_id=window_id
            )

            return summary


    async def query_by_timeframe_enhanced(
        self,
        query: str,
        window_id: Optional[str] = None,  # Keep parameter for backward compatibility
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Tuple[MemoryResponse, float]]:
        """
        Enhanced timeframe query with precise filtering and improved temporal boost logic.
        """
        try:
            # Create vector for semantic search
            query_vector = await self.vector_operations.create_semantic_vector(query)

            # Start with the provided filter or default to episodic memories
            filter_dict = filter_dict or {"memory_type": "EPISODIC"}

            # Log the filter for debugging
            logger.info(f"Temporal query with filter: {filter_dict}")

            # Log readable timestamp range if present
            if "created_at_unix" in filter_dict and "$gte" in filter_dict["created_at_unix"]:
                try:
                    gte_timestamp = filter_dict["created_at_unix"]["$gte"]
                    lte_timestamp = filter_dict["created_at_unix"].get("$lte", int(datetime.now(timezone.utc).timestamp()))

                    from_time = datetime.fromtimestamp(gte_timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    to_time = datetime.fromtimestamp(lte_timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                    logger.info(f"Searching for memories from {from_time} to {to_time} (Unix timestamps: {gte_timestamp} to {lte_timestamp})")
                except Exception as e:
                    logger.error(f"Error formatting timestamp range: {e}")

            logger.info(f"Querying Pinecone with filter (raw): {filter_dict}")
            results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=top_k * 3,  # Request more to account for filtering and boost adjustments
                filter=filter_dict,
                include_metadata=True
            )

            logger.info(f"Raw Pinecone query results (before processing): {results}")  # ADD THIS LOGGING LINE

            # Log sample memory timestamps for debugging
            if results:
                logger.info(f"Sample memories from query results:")
                for i, memory in enumerate(results.matches[:3]):  # Log first 3 memories
                    metadata = memory.metadata
                    logger.info(f"Memory {i+1}: {memory.id}")
                    logger.info(f"  created_at_unix: {metadata.get('created_at_unix', 'missing')}")
                    logger.info(f"  created_at_iso: {metadata.get('created_at_iso', 'missing')}")
                    logger.info(f"  created_at: {metadata.get('created_at', 'missing')}")
                    logger.info(f"  time_of_day: {metadata.get('time_of_day', 'missing')}")
                    logger.info(f"  day_of_week: {metadata.get('day_of_week', 'missing')}")

                    # Log human-readable time if available
                    if "created_at_unix" in metadata:
                        unix_ts = metadata["created_at_unix"]
                        if isinstance(unix_ts, (int, float)):
                            try:
                                dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
                                logger.info(f"  human-readable time: {dt.strftime('%Y-%m-%d %H:%M:%S')} ({metadata.get('time_of_day', 'unknown')})")
                            except Exception as e:
                                logger.error(f"Error converting Unix timestamp {unix_ts}: {e}")

            # Log the number of results
            logger.info(f"Temporal query returned {len(results.matches if results and results.matches else 0)} initial results")

            # Extract temporal references from query for better boosting
            query_lower = query.lower()
            time_references = {
                "morning": "morning" in query_lower,
                "afternoon": "afternoon" in query_lower,
                "evening": "evening" in query_lower or "night" in query_lower,
                "yesterday": "yesterday" in query_lower,
                "today": "today" in query_lower,
                "specific_time": bool(re.search(r"(?:at|around|about) \d{1,2}(?::\d{2})?\s*(?:am|pm)", query_lower))
            }

            # Check for day of week references
            day_of_week_reference = None
            for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                if day in query_lower:
                    day_of_week_reference = day
                    break

            # Process results with temporal boosting
            processed_results = []
            current_time = datetime.now(timezone.utc)

            for memory in results.matches:
                memory_data = memory.metadata
                score = memory.score
                try:
                    metadata = memory_data.get("metadata", {})

                    # Skip memories without proper metadata
                    if not metadata or "content" not in metadata:
                        logger.warning(f"Skipping memory {memory_data.get('id', 'unknown')} - missing metadata or content")
                        continue

                    # Get timestamp in most reliable format
                    memory_timestamp = None
                    created_at_object = None

                    # Try Unix timestamp first (most reliable for comparisons)
                    if "created_at_unix" in metadata and isinstance(metadata["created_at_unix"], (int, float)):
                        memory_timestamp = metadata["created_at_unix"]
                        created_at_object = datetime.fromtimestamp(memory_timestamp, tz=timezone.utc)
                    # Then try ISO string
                    elif "created_at_iso" in metadata and isinstance(metadata["created_at_iso"], str):
                        try:
                            created_at_object = datetime.fromisoformat(metadata["created_at_iso"].rstrip('Z'))
                            if created_at_object.tzinfo is None:
                                created_at_object = created_at_object.replace(tzinfo=timezone.utc)
                            memory_timestamp = int(created_at_object.timestamp())
                        except Exception as e:
                            logger.warning(f"Error parsing created_at_iso for memory {memory_data.get('id', 'unknown')}: {e}")
                    # Finally try created_at field
                    elif "created_at" in metadata:
                        try:
                            if isinstance(metadata["created_at"], (int, float)):
                                memory_timestamp = metadata["created_at"]
                                created_at_object = datetime.fromtimestamp(metadata["created_at"], tz=timezone.utc)
                            elif isinstance(metadata["created_at"], str):
                                created_at_object = normalize_timestamp(metadata["created_at"])
                        except Exception as e:
                            logger.warning(f"Error processing created_at for memory {memory_data.get('id', 'unknown')}: {e}")

                    # Skip if we couldn't determine a valid timestamp
                    if created_at_object is None:
                        logger.warning(f"Skipping memory {memory_data.get('id', 'unknown')} - couldn't determine timestamp")
                        continue

                    # Calculate memory age for recency boosts
                    memory_age_hours = (current_time - created_at_object).total_seconds() / 3600

                    # Start with base score
                    adjusted_score = score
                    applied_boosts = []

                    # Apply temporal relevance boosts based on query
                    # 1. Time of day, 2. Day of week, 3. "Yesterday", 4. "Today", 5. Specific Time, 6. Recency (very recent) - [Implementation of boosts as previously defined]

                    # Apply recency adjustment for very recent memories
                    if memory_age_hours < 1:  # Less than 1 hour old
                        adjusted_score *= 1.2  # 20% boost for very recent
                        applied_boosts.append("very recent (<1 hour)")

                    memory_response = MemoryResponse(
                        id=memory_data.id,
                        content=metadata["content"],
                        memory_type=MemoryType(metadata.get("memory_type", "EPISODIC")),
                        created_at=created_at_object.isoformat(),
                        metadata=metadata,
                        window_id=metadata.get("window_id")
                    )

                    memory_response.time_ago = self._format_time_ago(created_at_object)

                    # Log boosts if applied
                    if applied_boosts:
                        logger.info(f"Memory {memory_data.id} - Applied boosts: {', '.join(applied_boosts)}, " +
                                    f"Final score: {score:.3f} → {adjusted_score:.3f}")

                    processed_results.append((memory_response, adjusted_score))

                except Exception as e:
                    logger.error(f"Error processing memory {memory_data.get('id', 'unknown')} in timeframe query: {e}")
                    continue

            # Sort by adjusted score and limit to top_k
            processed_results.sort(key=lambda x: x[1], reverse=True)
            final_results = processed_results[:top_k]

            logger.info(f"Returning {len(final_results)}/{len(processed_results)} processed results for temporal query")
            return final_results

        except Exception as e:
            logger.error(f"Error in query_by_timeframe_enhanced: {e}", exc_info=True)
            return []

    async def parse_time_expression(
        self,
        time_expr: str,
        base_time: Optional[datetime] = None
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse natural language time expressions into start/end datetime objects.
        Handles various temporal references like today, yesterday, specific times of day,
        days ago, last week, specific weekdays, and exact times.

        Args:
            time_expr: Natural language time expression (e.g., "yesterday", "this morning")
            base_time: Reference time for relative expressions (defaults to current UTC time)

        Returns:
            Tuple of (start_time, end_time) as datetime objects with timezone info
        """
        # Ensure base_time is set and has timezone info
        base_time = base_time or datetime.now(timezone.utc)
        if base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=timezone.utc)

        # Normalize time expression for easier matching
        time_expr = time_expr.lower().strip()

        logger.info(f"Parsing time expression: '{time_expr}' (base time: {base_time.isoformat()})")

        # Log sample memories for debugging timestamp formats
        try:
            dummy_vector = [0.0] * 1536  # Adjust to match your vector dimension
            sample_results = await self.pinecone_service.query_memories(
                query_vector=dummy_vector,
                top_k=3,
                filter={"memory_type": "EPISODIC"},
                include_metadata=True
            )
            for i, (mem, _) in enumerate(sample_results):
                logger.info(f"Sample memory #{i+1}: created_at_unix={mem['metadata'].get('created_at_unix')}, " +
                            f"ISO={mem['metadata'].get('created_at_iso')}, " +
                            f"time_of_day={mem['metadata'].get('time_of_day', 'missing')}")
        except Exception as e:
            logger.error(f"Error sampling memories for timestamp format reference: {e}")

        # Check for combined expressions first (more specific matches)
        # === YESTERDAY WITH TIME OF DAY ===
        if "yesterday morning" in time_expr:
            yesterday = base_time - timedelta(days=1)
            start_time = yesterday.replace(hour=5, minute=0, second=0, microsecond=0)
            end_time = yesterday.replace(hour=12, minute=0, second=0, microsecond=0)
            logger.info(f"Yesterday morning timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time
        elif "yesterday afternoon" in time_expr:
            yesterday = base_time - timedelta(days=1)
            start_time = yesterday.replace(hour=12, minute=0, second=0, microsecond=0)
            end_time = yesterday.replace(hour=17, minute=0, second=0, microsecond=0)
            logger.info(f"Yesterday afternoon timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time
        elif "yesterday evening" in time_expr or "yesterday night" in time_expr:
            yesterday = base_time - timedelta(days=1)
            start_time = yesterday.replace(hour=17, minute=0, second=0, microsecond=0)
            end_time = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            logger.info(f"Yesterday evening/night timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        # Check for specific time mentions like "at 3pm yesterday"
        time_pattern = re.search(r"(?:at|around|about) (\d{1,2})[:\.]?(\d{2})?\s*(am|pm)?(?:\s+(\w+))?", time_expr)
        if time_pattern:
            # Extract time components
            hour = int(time_pattern.group(1))
            minute = int(time_pattern.group(2) or 0)
            am_pm = time_pattern.group(3)
            day_ref = time_pattern.group(4)  # Could be "yesterday", "today", etc.

            # Adjust hour for AM/PM
            if am_pm:
                if am_pm.lower() == 'pm' and hour < 12:
                    hour += 12
                elif am_pm.lower() == 'am' and hour == 12:
                    hour = 0

            # Determine which day
            target_day = base_time
            if day_ref and "yesterday" in day_ref:
                target_day = base_time - timedelta(days=1)

            # Create a 2-hour window centered on the specific time
            start_time = target_day.replace(hour=max(0, hour-1), minute=minute, second=0, microsecond=0)
            end_time = target_day.replace(hour=min(23, hour+1), minute=minute, second=59, microsecond=999999)

            logger.info(f"Specific time timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        # === TODAY REFERENCES ===
        if "today" in time_expr:
            logger.info("Detected 'today' expression")
            start_time = base_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = base_time.replace(hour=23, minute=59, second=59, microsecond=999999)
            logger.info(f"Today timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        # === THIS MORNING/AFTERNOON/EVENING ===
        elif "this morning" in time_expr:
            logger.info("Detected 'this morning' expression")
            start_time = base_time.replace(hour=5, minute=0, second=0, microsecond=0)
            end_time = base_time.replace(hour=12, minute=0, second=0, microsecond=0)
            logger.info(f"This morning timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time
        elif "this afternoon" in time_expr:
            start_time = base_time.replace(hour=12, minute=0, second=0, microsecond=0)
            end_time = base_time.replace(hour=17, minute=0, second=0, microsecond=0)
            logger.info(f"This afternoon timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time
        elif "this evening" in time_expr or "tonight" in time_expr:
            start_time = base_time.replace(hour=17, minute=0, second=0, microsecond=0)
            end_time = base_time.replace(hour=23, minute=59, second=59, microsecond=999999)
            logger.info(f"This evening timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        # === YESTERDAY ===
        elif "yesterday" in time_expr:
            logger.info("Detected 'yesterday' expression")
            yesterday = base_time - timedelta(days=1)
            start_time = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            logger.info(f"Yesterday timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        # === DAYS AGO ===
        elif "day" in time_expr and "ago" in time_expr:
            logger.info("Detected 'days ago' expression")
            days_pattern = re.search(r"(\d+)\s*days?", time_expr)
            if days_pattern:
                days_ago = int(days_pattern.group(1))
                # For multi-day windows, include the full period
                if days_ago > 1:
                    # Start from X days ago
                    start_time = (base_time - timedelta(days=days_ago)).replace(hour=0, minute=0, second=0, microsecond=0)
                    # End at yesterday
                    end_time = (base_time - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
                else:
                    # Just the specific day (yesterday for "1 day ago")
                    target_date = base_time - timedelta(days=days_ago)
                    start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_time = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

                logger.info(f"Days ago timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
                return start_time, end_time

        # === LAST WEEK ===
        elif "last week" in time_expr:
            logger.info("Detected 'last week' expression")
            # Get first day of current week (Monday)
            days_to_monday = base_time.weekday()
            start_of_this_week = base_time - timedelta(days=days_to_monday)
            start_of_this_week = start_of_this_week.replace(hour=0, minute=0, second=0, microsecond=0)

            # Start of last week is 7 days before
            start_time = start_of_this_week - timedelta(days=7)
            # End of last week is the day before this week started
            end_time = start_of_this_week - timedelta(microseconds=1)

            logger.info(f"Last week timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        # === SPECIFIC WEEKDAY ===
        weekday_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }

        for day_name, day_num in weekday_map.items():
            if day_name in time_expr:
                logger.info(f"Detected reference to '{day_name}'")

                # Handle "last Monday" vs just "Monday"
                if "last" in time_expr:
                    # Find the most recent occurrence of this day before the previous one
                    days_since = (base_time.weekday() - day_num) % 7  # Days since the last occurrence
                    if days_since == 0:  # If today is that day, go back a week
                        days_since = 7
                    days_back = days_since + 7  # Go back an extra week for "last X"
                else:
                    # Find the most recent occurrence of this day
                    days_since = (base_time.weekday() - day_num) % 7  # Days since the last occurrence
                    if days_since == 0:  # If today is that day, go back a week
                        days_since = 7
                    days_back = days_since

                target_date = base_time - timedelta(days=days_back)
                start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

                logger.info(f"Weekday timeframe for {day_name}: {start_time.isoformat()} to {end_time.isoformat()}")
                return start_time, end_time

        # === EARLY, RECENT EXPRESSIONS ===
        if "just now" in time_expr or "a moment ago" in time_expr:
            # Very recent - last 10 minutes
            end_time = base_time
            start_time = end_time - timedelta(minutes=10)
            logger.info(f"Very recent timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        elif "earlier today" in time_expr:
            # Earlier part of today
            now_hour = base_time.hour
            start_time = base_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = base_time.replace(hour=max(0, now_hour-1), minute=59, second=59, microsecond=999999)
            logger.info(f"Earlier today timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        elif "an hour ago" in time_expr:
            # About an hour ago - 40-80 minute window
            end_time = base_time - timedelta(minutes=40)
            start_time = base_time - timedelta(minutes=80)
            logger.info(f"About an hour ago timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
            return start_time, end_time

        # === DEFAULT FALLBACK: LAST 24 HOURS ===
        logger.warning(f"No specific time expression pattern matched for '{time_expr}'. Using default (last 24 hours)")
        end_time = base_time
        start_time = end_time - timedelta(days=1)

        logger.info(f"Default timeframe: {start_time.isoformat()} to {end_time.isoformat()}")
        return start_time, end_time

    async def query_by_timeframe_enhanced(
        self,
        query: str,
        window_id: Optional[str] = None,  # Keep parameter for backward compatibility
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Tuple[MemoryResponse, float]]:
        """
        Enhanced timeframe query with precise filtering and improved temporal boost logic.
        """
        try:
            # Create vector for semantic search
            query_vector = await self.vector_operations.create_semantic_vector(query)

            # Start with the provided filter or default to episodic memories
            filter_dict = filter_dict or {"memory_type": "EPISODIC"}

            # Log the filter for debugging
            logger.info(f"Temporal query with filter: {filter_dict}")

            # Log readable timestamp range if present
            if "created_at_unix" in filter_dict and "$gte" in filter_dict["created_at_unix"]:
                try:
                    gte_timestamp = filter_dict["created_at_unix"]["$gte"]
                    lte_timestamp = filter_dict["created_at_unix"].get("$lte", int(datetime.now(timezone.utc).timestamp()))

                    from_time = datetime.fromtimestamp(gte_timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    to_time = datetime.fromtimestamp(lte_timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                    logger.info(f"Searching for memories from {from_time} to {to_time} (Unix timestamps: {gte_timestamp} to {lte_timestamp})")
                except Exception as e:
                    logger.error(f"Error formatting timestamp range: {e}")

            logger.info(f"Querying Pinecone with filter (raw): {filter_dict}")
            results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=top_k * 3,  # Request more to account for filtering and boost adjustments
                filter=filter_dict,
                include_metadata=True
            )

            logger.info(f"Raw Pinecone query results (before processing): {results}")

            # Log sample memory timestamps for debugging
            if results:  # Changed from 'results and results.matches'
                logger.info(f"Sample memories from query results:")
                for i, (memory_data, _score) in enumerate(results[:3]):  # Changed to iterate through the list of tuples
                    metadata = memory_data["metadata"]  # Changed from memory.metadata
                    logger.info(f"Memory {i+1}: {memory_data['id']}")  # Changed from memory.id
                    logger.info(f"  created_at_unix: {metadata.get('created_at_unix', 'missing')}")
                    logger.info(f"  created_at_iso: {metadata.get('created_at_iso', 'missing')}")
                    logger.info(f"  created_at: {metadata.get('created_at', 'missing')}")
                    logger.info(f"  time_of_day: {metadata.get('time_of_day', 'missing')}")
                    logger.info(f"  day_of_week: {metadata.get('day_of_week', 'missing')}")

                    # Log human-readable time if available
                    if "created_at_unix" in metadata:
                        unix_ts = metadata["created_at_unix"]
                        if isinstance(unix_ts, (int, float)):
                            try:
                                dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
                                logger.info(f"  human-readable time: {dt.strftime('%Y-%m-%d %H:%M:%S')} ({metadata.get('time_of_day', 'unknown')})")
                            except Exception as e:
                                logger.error(f"Error converting Unix timestamp {unix_ts}: {e}")

            # Log the number of results
            logger.info(f"Temporal query returned {len(results)} initial results")  # Updated

            # Extract temporal references from query for better boosting
            query_lower = query.lower()
            time_references = {
                "morning": "morning" in query_lower,
                "afternoon": "afternoon" in query_lower,
                "evening": "evening" in query_lower or "night" in query_lower,
                "yesterday": "yesterday" in query_lower,
                "today": "today" in query_lower,
                "specific_time": bool(re.search(r"(?:at|around|about) \d{1,2}(?::\d{2})?\s*(?:am|pm)", query_lower))
            }

            # Check for day of week references
            day_of_week_reference = None
            for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                if day in query_lower:
                    day_of_week_reference = day
                    break

            # Process results with temporal boosting
            processed_results = []
            current_time = datetime.now(timezone.utc)

            # Changed to iterate over results as a list of tuples (memory_data, score)
            for memory_data, score in results:
                try:
                    metadata = memory_data["metadata"]  # Changed from memory_data.get("metadata", {})

                    # Skip memories without proper metadata
                    if not metadata or "content" not in metadata:
                        logger.warning(f"Skipping memory {memory_data.get('id', 'unknown')} - missing metadata or content")
                        continue

                    # Get timestamp in most reliable format
                    memory_timestamp = None
                    created_at_object = None

                    # Try Unix timestamp first (most reliable for comparisons)
                    if "created_at_unix" in metadata and isinstance(metadata["created_at_unix"], (int, float)):
                        memory_timestamp = metadata["created_at_unix"]
                        created_at_object = datetime.fromtimestamp(memory_timestamp, tz=timezone.utc)
                    # Then try ISO string
                    elif "created_at_iso" in metadata and isinstance(metadata["created_at_iso"], str):
                        try:
                            created_at_object = datetime.fromisoformat(metadata["created_at_iso"].rstrip('Z'))
                            if created_at_object.tzinfo is None:
                                created_at_object = created_at_object.replace(tzinfo=timezone.utc)
                            memory_timestamp = int(created_at_object.timestamp())
                        except Exception as e:
                            logger.warning(f"Error parsing created_at_iso for memory {memory_data.get('id', 'unknown')}: {e}")
                    # Finally try created_at field
                    elif "created_at" in metadata:
                        try:
                            if isinstance(metadata["created_at"], (int, float)):
                                memory_timestamp = metadata["created_at"]
                                created_at_object = datetime.fromtimestamp(metadata["created_at"], tz=timezone.utc)
                            elif isinstance(metadata["created_at"], str):
                                created_at_object = normalize_timestamp(metadata["created_at"])
                        except Exception as e:
                            logger.warning(f"Error processing created_at for memory {memory_data.get('id', 'unknown')}: {e}")

                    # Skip if we couldn't determine a valid timestamp
                    if created_at_object is None:
                        logger.warning(f"Skipping memory {memory_data.get('id', 'unknown')} - couldn't determine timestamp")
                        continue

                    # Calculate memory age for recency boosts
                    memory_age_hours = (current_time - created_at_object).total_seconds() / 3600

                    # Start with base score
                    adjusted_score = score
                    applied_boosts = []

                    # Apply temporal relevance boosts based on query
                    # 1. Time of day, 2. Day of week, 3. "Yesterday", 4. "Today", 5. Specific Time, 6. Recency (very recent) - [Implementation of boosts as previously defined]

                    # Apply recency adjustment for very recent memories
                    if memory_age_hours < 1:  # Less than 1 hour old
                        adjusted_score *= 1.2  # 20% boost for very recent
                        applied_boosts.append("very recent (<1 hour)")

                    memory_response = MemoryResponse(
                        id=memory_data["id"],  # Changed from memory_data.id
                        content=metadata["content"],
                        memory_type=MemoryType(metadata.get("memory_type", "EPISODIC")),
                        created_at=created_at_object.isoformat(),
                        metadata=metadata,
                        window_id=metadata.get("window_id")
                    )

                    memory_response.time_ago = self._format_time_ago(created_at_object)

                    # Log boosts if applied
                    if applied_boosts:
                        logger.info(f"Memory {memory_data['id']} - Applied boosts: {', '.join(applied_boosts)}, " +
                                    f"Final score: {score:.3f} → {adjusted_score:.3f}")

                    processed_results.append((memory_response, adjusted_score))

                except Exception as e:
                    logger.error(f"Error processing memory {memory_data.get('id', 'unknown')} in timeframe query: {e}")
                    continue

            # Sort by adjusted score and limit to top_k
            processed_results.sort(key=lambda x: x[1], reverse=True)
            final_results = processed_results[:top_k]

            logger.info(f"Returning {len(final_results)}/{len(processed_results)} processed results for temporal query")
            return final_results

        except Exception as e:
            logger.error(f"Error in query_by_timeframe_enhanced: {e}", exc_info=True)
            return []

    async def batch_query_memories(
        self,
        query: str,
        window_id: Optional[str] = None,  # Keep parameter for compatibility
        top_k_per_type: Union[int, Dict[MemoryType, int]] = 5,  # Accept either int or dict
        request_metadata: Optional[RequestMetadata] = None
    ) -> Dict[MemoryType, List[Tuple[MemoryResponse, float]]]:
        """
        Query all memory types in a single batched operation.
        """
        logger.info(f"Starting batch memory query: '{query[:50]}...'")

        # First check if this is a temporal query, but don't pass window_id
        temporal_summaries = await self.process_temporal_query(query)
        if temporal_summaries and "episodic" in temporal_summaries:
            # If it's a temporal query and we got results, return them in the format expected by caller
            logger.info("Processed as temporal query, returning specialized results")

            # Return the episodic memories directly without wrapping in a dummy object.
            # episodic_memories should already be a list of (MemoryResponse, score) tuples
            return {MemoryType.EPISODIC: temporal_summaries['episodic']} # Corrected line

        # Create embedding cache for this query
        embedding_cache = {}
        async def get_cached_embedding(text: str) -> List[float]:
            if text in embedding_cache:
                return embedding_cache[text]
            embedding = await self.vector_operations.create_semantic_vector(text)
            embedding_cache[text] = embedding
            return embedding

        # Generate query vector once
        query_vector = await get_cached_embedding(query)

        # Extract keywords once

        # Create tasks for each memory type
        tasks = []
        for memory_type in [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.LEARNED]:
            # Determine top_k for this memory type
            if isinstance(top_k_per_type, dict):
                top_k = top_k_per_type.get(memory_type, 5)  # Default to 5 if not specified
            else:
                top_k = top_k_per_type

            # Create query request
            request = QueryRequest(
                prompt=query,
                top_k=top_k,  # Use the type-specific top_k
                window_id=window_id,
                memory_type=memory_type,
                request_metadata=request_metadata or RequestMetadata(
                    operation_type=OperationType.QUERY,
                    window_id=window_id
                )
            )

            # Use the cached vector
            task = asyncio.create_task(
                self._query_memory_type(
                    request=request,
                    pre_computed_vector=query_vector
                )
            )
            tasks.append((memory_type, task))

        # Wait for all tasks to complete
        results = {}
        for memory_type, task in tasks:
            try:
                query_response = await task
                results[memory_type] = list(zip(query_response.matches, query_response.similarity_scores))
            except Exception as e:
                logger.error(f"Error querying {memory_type} memories: {e}")
                results[memory_type] = []

        return results

    # In the MemorySystem class, add the function first
    def apply_time_weighting(self, memory_scores, decay_factor=0.03):  # Further reduced decay factor
        """Apply even less aggressive time-based decay to memory relevance scores."""
        now = datetime.now(timezone.utc)

        for i, (memory, score) in enumerate(memory_scores):
            # Calculate age in days - ensure timestamp is compatible
            if isinstance(memory.created_at, str):
                # Handle string format
                created_at = datetime.fromisoformat(memory.created_at.rstrip('Z'))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            else:
                # Already a datetime object
                created_at = memory.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)

            age_in_days = (now - created_at).days

            # Apply gentler exponential decay based on age
            time_weight = math.exp(-decay_factor * age_in_days)

            # Memory type-aware weighting
            if hasattr(memory, 'memory_type') and memory.memory_type:
                if memory.memory_type.value == "SEMANTIC":
                    # For semantic memories, focus almost entirely on relevance
                    adjusted_score = score * 0.95 + time_weight * 0.05
                elif memory.memory_type.value == "EPISODIC":
                    # For episodic, use moderate time weighting
                    adjusted_score = score * 0.8 + time_weight * 0.2
                else:
                    # For learned or other types, use balanced approach
                    adjusted_score = score * 0.9 + time_weight * 0.1
            else:
                # Default case if memory_type not available
                adjusted_score = score * 0.9 + time_weight * 0.1

            # Update the score
            memory_scores[i] = (memory, adjusted_score)

        # Re-sort based on adjusted scores
        memory_scores.sort(key=lambda x: x[1], reverse=True)

        return memory_scores

    def _calculate_bell_curve_recency(self, age_hours, is_temporal_query=False):
        """Calculate recency score with special handling for temporal queries."""
        # For temporal queries, use a broader bell curve
        if is_temporal_query:
            # Use different parameters optimized for finding past discussions
            peak_hours = getattr(self.settings, 'temporal_query_peak_hours', 72)
            steepness = getattr(self.settings, 'temporal_query_steepness', 1.5)  # Less steep
            very_recent_threshold = getattr(self.settings, 'temporal_very_recent_threshold', 2.0)
            recent_threshold = getattr(self.settings, 'temporal_recent_threshold', 36.0)
        else:
            # Regular parameters
            peak_hours = getattr(self.settings, 'episodic_peak_hours', 36)
            steepness = getattr(self.settings, 'episodic_bell_curve_steepness', 2.5)
            very_recent_threshold = getattr(self.settings, 'episodic_very_recent_threshold', 1.0)
            recent_threshold = getattr(self.settings, 'episodic_recent_threshold', 24.0)

        # Check if this is a temporal query (looking for a specific time period)
        query = getattr(self, 'current_query', '').lower()

        # More specific temporal query handling (binary approach for better filtering)
        if query:
            # Extract time period from query
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Get memory timestamp data for comparison
            memory_date = getattr(self, 'current_memory_date', None)
            memory_time_of_day = getattr(self, 'current_memory_time_of_day', None)

            # Apply strict binary scoring for temporal queries
            if 'this morning' in query and memory_date == today_str and memory_time_of_day == 'morning':
                return 1.0  # Perfect match for "this morning" query
            elif 'today' in query and memory_date == today_str:
                return 1.0  # Perfect match for "today" query
            elif ('yesterday' in query and
                memory_date == (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")):
                return 1.0  # Perfect match for "yesterday" query
            elif any(x in query for x in ['this morning', 'today', 'yesterday', 'last week']):
                return 0.2  # Significant penalty for non-matching memories in temporal queries

        # Add special handling for very recent memories (less than 10 minutes old)
        if age_hours < 0.17:  # Less than 10 minutes (0.17 hours)
            return 0.95  # Very high recency score for just-happened conversations

        # Standard bell curve logic for non-temporal queries remains the same
        if age_hours < very_recent_threshold:
            return 0.2 + (0.2 * age_hours / very_recent_threshold)
        elif age_hours < recent_threshold:
            relative_position = (age_hours - very_recent_threshold) / (recent_threshold - very_recent_threshold)
            return 0.4 + (0.4 * relative_position)

        # Bell curve peak and decay
        else:
            # Calculate distance from peak (in hours)
            distance_from_peak = abs(age_hours - peak_hours)

            # Convert to a bell curve shape (Gaussian-inspired)
            max_distance = getattr(self.settings, 'episodic_max_age_days', 7) * 24 - peak_hours

            # Normalized distance from peak (0-1)
            normalized_distance = min(1.0, distance_from_peak / max_distance)

            # Apply bell curve formula (variant of Gaussian)
            bell_value = math.exp(-(normalized_distance ** 2) * steepness)

            # Scale between 0.8 (peak) and 0.2 (oldest)
            return 0.8 * bell_value + 0.2
        
    

    async def _query_memory_type(
        self,
        request: QueryRequest,
        pre_computed_vector: Optional[List[float]] = None,
        pinecone_filter: Optional[Dict[str, Any]] = None  # COMMA ADDED HERE
    ) -> QueryResponse:
        """
        Query a specific memory type using pre-computed vector if available.
        This is an internal helper for batch_query_memories.
        """
        try:
            # Use pre-computed vector if provided
            if pre_computed_vector is not None:
                query_vector = pre_computed_vector
                logger.info("Using pre-computed vector for memory retrieval")
            else:
                query_vector = await self.vector_operations.create_semantic_vector(request.prompt)
                logger.info(f"Generated new vector for memory retrieval (first 5 elements): {query_vector[:5]}")

            # Determine memory type filter and prepare filter
            memory_type_value = request.memory_type.value
            logger.info(f"Querying memory type: {memory_type_value}")

            # Define the filter based on memory type
            if request.memory_type == MemoryType.SEMANTIC:
                pinecone_filter = {"memory_type": "SEMANTIC"}
            elif request.memory_type == MemoryType.EPISODIC:
                seven_days_ago = datetime.now(timezone.utc) - timedelta(days=14)
                seven_days_ago_timestamp = int(seven_days_ago.timestamp())
                pinecone_filter = {
                    "memory_type": "EPISODIC",
                    "created_at_unix": {"$gte": seven_days_ago_timestamp}
                }
                if request.window_id: # ADDED WINDOW_ID FILTER HERE
                    pinecone_filter["window_id"] = request.window_id
            elif request.memory_type == MemoryType.LEARNED:
                pinecone_filter = {"memory_type": "LEARNED"}

            # Execute vector search
            vector_results = await self.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=request.top_k,
                filter=pinecone_filter,
                include_metadata=True
            )
            logger.info(f"Vector search returned {len(vector_results)} results")

            # Process vector results
            matches = []
            similarity_scores = []
            current_time = datetime.utcnow()

            for memory_data, similarity_score in vector_results:
                try:
                    # Apply basic similarity threshold
                    if similarity_score < self.settings.min_similarity_threshold:
                        continue

                    # Process created_at timestamp
                    created_at_raw = memory_data["metadata"].get("created_at")
                    if not created_at_raw:
                        continue

                    # Add the is_very_recent check here
                    if memory_data["metadata"].get("is_very_recent", False):
                        created_at = datetime.fromisoformat(memory_data["metadata"].get("created_at_iso", "").rstrip('Z'))
                        if datetime.now(timezone.utc) - created_at > timedelta(minutes=10):
                            # No longer very recent
                            memory_data["metadata"]["is_very_recent"] = False
                            # Optionally update in database
                            await self.pinecone_service.update_memory_metadata(memory_data["id"],
                                                                            {"is_very_recent": False})

                    # Convert timestamp to datetime
                    if isinstance(created_at_raw, str):
                        created_at = normalize_timestamp(created_at_raw)
                    else:
                        created_at = created_at_raw

                    # Get memory type and calculate final score based on memory type
                    memory_type = memory_data["metadata"].get("memory_type", "UNKNOWN")
                    final_score = similarity_score

                    # Check for empty/negative responses and apply severe penalty
                    content = memory_data["metadata"].get("content", "").lower()
                    empty_response_penalty = getattr(self.settings, 'empty_response_penalty', 0.01)

                    # Check if this memory has already been pre-filtered
                    if not hasattr(memory_data, 'pre_filtered') or not memory_data.pre_filtered:
                        # More comprehensive negative phrases detection
                        negative_phrases = [
                            "i don't recall", "i'm sorry", "no relevant", "haven't discussed",
                            "don't remember", "i do not recall", "not recall any",
                            "not remember any", "doesn't look like", "nothing specific"
                        ]

                        if any(phrase in content for phrase in negative_phrases):
                            # Apply severe penalty to empty/negative responses
                            original_score = final_score
                            final_score *= empty_response_penalty
                            logger.info(f"Applied empty response penalty to memory {memory_data['id']}: " +
                                    f"Score reduced from {original_score:.4f} to {final_score:.4f}")

                    # More comprehensive negative phrases detection
                    negative_phrases = [
                        "i don't recall", "i'm sorry", "no relevant", "haven't discussed",
                        "don't remember", "i do not recall", "not recall any",
                        "not remember any", "doesn't look like", "nothing specific"
                    ]

                    if any(phrase in content for phrase in negative_phrases): # <---- ENSURE THIS LINE IS EXACTLY LIKE THIS
                        # Apply severe penalty to empty/negative responses
                        original_score = final_score
                        final_score *= empty_response_penalty  # Default 0.01 = 1% of original score
                        logger.info(f"Applied empty response penalty to memory {memory_data['id']}: " +
                                f"Score reduced from {original_score:.4f} to {final_score:.4f}")

                    # Apply memory-type specific scoring
                    if memory_type == "EPISODIC":
                        # Adjust for timezone if needed
                        if current_time.tzinfo is None and created_at.tzinfo is not None:
                            current_time = current_time.replace(tzinfo=timezone.utc)
                        elif current_time.tzinfo is not None and created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)

                        # Calculate age in hours
                        age_hours = (current_time - created_at).total_seconds() / (60*60)

                        # Use bell curve recency scoring if enabled, otherwise use original method
                        use_bell_curve = getattr(self.settings, 'episodic_bell_curve_enabled', True)

                        if use_bell_curve:
                            # Apply bell curve scoring
                            recency_score = self._calculate_bell_curve_recency(age_hours)
                        else:
                            # Original method (linear then exponential decay)
                            if age_hours <= self.settings.episodic_recent_hours:
                                recency_score = 1.0 - (age_hours / self.settings.episodic_recent_hours) * 0.15
                            else:
                                max_age_hours = self.settings.episodic_max_age_days * 24
                                relative_age = (age_hours - self.settings.episodic_recent_hours) / (max_age_hours - self.settings.episodic_recent_hours)
                                recency_score = 0.85 * (0.3/0.85) ** relative_age

                        recency_score = max(0.0, min(1.0, recency_score))

                        # Combine relevance and recency using settings
                        relevance_weight = 1 - self.settings.episodic_recency_weight
                        combined_score = (
                            relevance_weight * similarity_score +
                            self.settings.episodic_recency_weight * recency_score
                        )

                        # Apply memory type weight
                        combined_score = combined_score * self.settings.episodic_memory_weight

                        # Check for substantive content and boost if found
                        content_boost = getattr(self.settings, 'content_response_boost', 5.0)
                        if (not any(phrase in content for phrase in negative_phrases) and
                            len(content) > 100):  # Threshold for meaningful content
                            original_score = combined_score
                            combined_score *= content_boost  # Default 5.0 = 5x boost for real content
                            logger.info(f"Applied content boost to memory {memory_data['id']}: " +
                                    f"Score increased from {original_score:.4f} to {combined_score:.4f}")

                        final_score = combined_score

                        logger.info(f"Memory ID {memory_data['id']} (EPISODIC): Raw={similarity_score:.3f}, " +
                            f"Age={age_hours:.1f}h, Recency={recency_score:.3f}, Final={final_score:.3f}")

                    elif memory_type == "SEMANTIC":
                        # Extract creation timestamp
                        created_at_raw = memory_data["metadata"].get("created_at")
                        if isinstance(created_at_raw, str):
                            created_at = normalize_timestamp(created_at_raw)
                        else:
                            created_at = created_at_raw

                        # Add this timezone handling code here
                        if current_time.tzinfo is None and created_at.tzinfo is not None:
                            current_time = current_time.replace(tzinfo=timezone.utc)
                        elif current_time.tzinfo is not None and created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)

                        # Calculate age in days
                        age_days = (current_time - created_at).total_seconds() / (86400)  # Seconds in a day

                        # Calculate recency score with much slower decay for semantic memories
                        # Increase minimum score and slow down the decay rate significantly
                        recency_score = max(0.6, 1.0 - (math.log(1 + age_days) / 20))

                        # Combine scores using semantic_recency_weight from settings
                        semantic_recency_weight = getattr(self.settings, 'semantic_recency_weight', 0.15)  # Default if not set
                        relevance_weight = 1 - semantic_recency_weight
                        combined_score = (
                            relevance_weight * similarity_score +
                            semantic_recency_weight * recency_score
                        )

                        # Apply memory type weight
                        final_score = combined_score * self.settings.semantic_memory_weight

                        logger.info(f"Memory ID {memory_data['id']} (SEMANTIC): Raw={similarity_score:.3f}, "
                            f"Age={age_days:.1f}d, Recency={recency_score:.3f}, Final={final_score:.3f}")

                    elif memory_type == "LEARNED":
                        confidence = memory_data["metadata"].get("confidence", 0.5)
                        combined_score = (similarity_score * 0.8) + (confidence * 0.2)
                        final_score = combined_score * self.settings.learned_memory_weight
                        logger.info(f"Memory ID {memory_data['id']} (LEARNED): Raw={similarity_score:.3f}, " +
                            f"Confidence={confidence:.2f}, Final={final_score:.3f}")

                    # Create memory response
                    memory_response = MemoryResponse(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                        created_at=created_at.isoformat() + "Z",
                        metadata=memory_data["metadata"],
                        window_id=memory_data["metadata"].get("window_id"),
                        semantic_vector=memory_data.get("vector"),
                    )
                    memory_response.time_ago = self._format_time_ago(created_at)
                    matches.append(memory_response)
                    similarity_scores.append(final_score)

                except Exception as e:
                    logger.error(f"Error processing memory {memory_data.get('id', 'unknown')}: {e}")
                    continue

            # Sort results by final score
            if matches and similarity_scores:
                # Create pairs of memories and scores
                memory_scores = list(zip(matches, similarity_scores))

                # Apply time-based weighting with memory-type awareness
                memory_scores = self.apply_time_weighting(memory_scores, decay_factor=0.03)

                # Extract the top_k results
                memory_scores = memory_scores[:request.top_k]

                # Unzip the results
                matches, similarity_scores = zip(*memory_scores)
            else:
                matches, similarity_scores = [], []

            return QueryResponse(
                matches=list(matches),
                similarity_scores=list(similarity_scores),
            )

        except Exception as e:
            logger.error(f"Error in _query_memory_type: {e}", exc_info=True)
            return QueryResponse(matches=[], similarity_scores=[])

    async def _combine_search_results(
        self,
        vector_results: List[Tuple[Dict[str, Any], float]],
        keyword_results: List[Tuple[Memory, float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Combine and re-rank results from vector and keyword searches

        Args:
            vector_results: Results from vector search (memory_data, score)
            keyword_results: Results from keyword search (Memory, score)

        Returns:
            Combined and re-ranked list of memory data and scores
        """
        try:
            # Add detailed debug logging to identify the call stack and recursive calls
            logger.info(f"Combining vector ({len(vector_results)}) and keyword ({len(keyword_results)}) search results")

            # Create a dictionary to combine results by memory_id
            combined_dict = {}

            # Process vector results
            for memory_data, vector_score in vector_results:
                memory_id = memory_data["id"]
                combined_dict[memory_id] = {
                    "memory_data": memory_data,
                    "vector_score": vector_score,
                    "keyword_score": 0.0
                }

            # Process keyword results - ONLY if there are any
            if keyword_results:
                for memory, keyword_score in keyword_results:
                    memory_id = memory.id
                    if memory_id in combined_dict:
                        # Update with keyword score if entry already exists
                        combined_dict[memory_id]["keyword_score"] = keyword_score
                    else:
                        # Memory exists in keyword results but not in vector results
                        # We need to create a compatible memory_data dict
                        memory_data = {
                            "id": memory.id,
                            "metadata": {
                                "content": memory.content,
                                "memory_type": memory.memory_type.value,
                                "created_at": memory.created_at,
                                "window_id": memory.window_id
                            },
                            "vector": memory.semantic_vector
                        }
                        combined_dict[memory_id] = {
                            "memory_data": memory_data,
                            "vector_score": 0.0,
                            "keyword_score": keyword_score
                        }

            # Calculate combined scores (with configurable weights)
            vector_weight = self.settings.vector_search_weight if hasattr(self.settings, 'vector_search_weight') else 0.7
            keyword_weight = self.settings.keyword_search_weight if hasattr(self.settings, 'keyword_search_weight') else 0.3

            # Calculate combined scores and prepare results
            combined_results = []
            for memory_id, data in combined_dict.items():
                combined_score = (data["vector_score"] * vector_weight) + (data["keyword_score"] * keyword_weight)
                combined_results.append((data["memory_data"], combined_score))

            # Sort by combined score (descending)
            combined_results.sort(key=lambda x: x[1], reverse=True)

            # Log result counts (only once)
            logger.info(f"Combined results: {len(combined_results)} memories")

            return combined_results

        except Exception as e:
            logger.error(f"Error combining search results: {e}", exc_info=True)
            # Return an empty list if there's an error, not None
            return []
        
    async def optimize_temporal_memories(
        self,
        candidates: List[Tuple[MemoryResponse, float]],
        max_memories: int = 5,
        time_window_hours: float = 24
    ) -> List[MemoryResponse]:
        """
        Optimize the selection of episodic memories for a temporal query.
        Only memories within the target time window (e.g., last 24 hours) are allowed.
        The optimization maximizes the total relevance score.
        
        Args:
            candidates: List of (MemoryResponse, score) tuples.
            max_memories: Maximum number of memories to select.
            time_window_hours: Only memories with age (in hours) <= this value will be considered.
        
        Returns:
            List of selected MemoryResponse objects.
        """
        # Filter candidates by time window.
        now = datetime.now(timezone.utc)
        filtered_candidates = []
        scores = []
        timestamps = []
        for mem, score in candidates:
            try:
                # Assume mem.created_at is an ISO string.
                mem_time = datetime.fromisoformat(mem.created_at.rstrip('Z'))
                if mem_time.tzinfo is None:
                    mem_time = mem_time.replace(tzinfo=timezone.utc)
                age_hours = (now - mem_time).total_seconds() / 3600
                if age_hours <= time_window_hours:
                    filtered_candidates.append(mem)
                    scores.append(score)
                    timestamps.append(mem_time.timestamp())
            except Exception as e:
                # Skip candidates with parsing errors.
                continue

        if not filtered_candidates:
            return []

        n = len(filtered_candidates)
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            # Fallback: return top scoring filtered candidates.
            sorted_by_score = sorted(zip(filtered_candidates, scores), key=lambda x: x[1], reverse=True)
            return [mem for mem, _ in sorted_by_score][:max_memories]

        # Create binary decision variables for each memory.
        memory_vars = [solver.BoolVar(f"mem_{i}") for i in range(n)]
        
        # Objective: maximize total score.
        solver.Maximize(solver.Sum([memory_vars[i] * scores[i] for i in range(n)]))
        
        # Constraint: select at most max_memories.
        solver.Add(solver.Sum(memory_vars) <= max_memories)
        
        # Optional: ensure at least one memory is very recent (e.g., within the last 2 hours).
        very_recent_threshold = now - timedelta(hours=2)
        very_recent_indices = [i for i, ts in enumerate(timestamps) if ts >= very_recent_threshold.timestamp()]
        if very_recent_indices:
            solver.Add(solver.Sum([memory_vars[i] for i in very_recent_indices]) >= 1)
        
        status = solver.Solve()
        selected = []
        if status == pywraplp.Solver.OPTIMAL:
            for i in range(n):
                if memory_vars[i].solution_value() > 0.5:
                    selected.append(filtered_candidates[i])
        else:
            # Fallback: choose the top scoring filtered candidates.
            sorted_by_score = sorted(zip(filtered_candidates, scores), key=lambda x: x[1], reverse=True)
            selected = [mem for mem, _ in sorted_by_score][:max_memories]

        return selected


    async def memory_summarization_agent(
        self,
        query: str,
        semantic_memories: List[Tuple[MemoryResponse, float]],
        episodic_memories: List[Tuple[MemoryResponse, float]],
        learned_memories: List[Tuple[MemoryResponse, float]],
        time_expression: Optional[str] = None,
        temporal_context: Optional[str] = None,
        window_id: Optional[str] = None  # Keep parameter for compatibility
    ) -> Dict[str, str]:
        """Use an LLM to process retrieved memories into human-like summaries for all memory types."""
        logger.info(f"Processing memories with summarization agent for query: {query[:50]}...")
        # Log detailed information about each episodic memory to diagnose issues
        logger.info(f"Number of episodic memories received: {len(episodic_memories)}")
        for i, (mem, score) in enumerate(episodic_memories):
            # Log a snippet of content for debugging
            content_snippet = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
            logger.info(f"Memory content snippet: {content_snippet}")

        # Pre-filter obviously empty responses
        if episodic_memories:
            filtered_episodic_memories = []
            for mem, score in episodic_memories:
                content_lower = mem.content.lower()
                if (len(mem.content) > 150 or 
                    not any(phrase in content_lower for phrase in ["i don't recall", "no relevant", "haven't discussed", "i'm sorry"])):
                    filtered_episodic_memories.append((mem, score))
                else:
                    logger.info(f"Pre-filtering empty response memory: {mem.id}")
            
            # Only replace if we have some substantial memories left
            if filtered_episodic_memories:
                episodic_memories = filtered_episodic_memories

        # **NEW STEP: Temporal Query Optimization**
        # If the query contains temporal keywords, run the OR-based optimization before further processing
        temporal_keywords = ["yesterday", "today", "last week", "this morning", "this afternoon", "this evening"]
        if any(keyword in query.lower() for keyword in temporal_keywords):
            logger.info("Temporal query detected: applying OR-based optimization to episodic memories")
            optimized = await self.optimize_temporal_memories(episodic_memories, time_window_hours=24)
            if optimized:
                episodic_memories = [(mem, 1.0) for mem in optimized]
            else:
                episodic_memories = []

        # Format semantic memories (no changes)
        semantic_content = "\n".join([
            f"- {mem.content[:300]}..." if len(mem.content) > 300 else f"- {mem.content}"
            for mem, _ in semantic_memories
        ])

        # Format learned memories (no changes)
        learned_content = "\n".join([
            f"- {mem.content[:300]}..." if len(mem.content) > 300 else f"- {mem.content}"
            for mem, _ in learned_memories
        ])

        # === Enhanced Episodic Memory Formatting for Contextual Relevance ===
        episodic_content_list = []
        if episodic_memories:
            # Log detailed information about each episodic memory to diagnose issues
            logger.info(f"Number of episodic memories received: {len(episodic_memories)}")
            for i, (mem, score) in enumerate(episodic_memories):
                # Look for mentions related to the query
                query_topic = query.lower().replace("when did we discuss ", "").replace("when did we talk about ", "").replace("?", "").strip()
                contains_topic = query_topic in mem.content.lower() if query_topic else False
                logger.info(f"Episodic memory {i+1}/{len(episodic_memories)}: "
                        f"ID={mem.id}, Score={score:.3f}, Contains '{query_topic}'={contains_topic}")

            # Get embedding of the CURRENT QUERY for relevance scoring
            query_vector = await self.vector_operations.create_semantic_vector(query)

            for mem, _score in episodic_memories:
                try:
                    if mem.metadata.get("is_very_recent", False):
                        created_at_value = mem.metadata.get("created_at_iso", "")
                        # Handle different types for created_at_value
                        if isinstance(created_at_value, str):
                            created_at = datetime.fromisoformat(created_at_value.rstrip('Z'))
                        elif isinstance(created_at_value, datetime):
                            created_at = created_at_value
                        else:
                            # Default to current time if we can't parse
                            created_at = datetime.now(timezone.utc)
                        # Ensure timezone information
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        # Check if it's still "very recent"
                        if datetime.now(timezone.utc) - created_at > timedelta(minutes=10):
                            mem.metadata["is_very_recent"] = False
                            # Optionally update in database
                            await self.pinecone_service.update_memory_metadata(mem.id,
                                                                            {"is_very_recent": False})
                    # Calculate semantic similarity between memory and CURRENT QUERY
                    memory_vector = mem.semantic_vector  # Assuming semantic_vector is populated
                    if memory_vector:
                        similarity_to_query = self.vector_operations.cosine_similarity(query_vector, memory_vector)
                        logger.info(f"Memory {mem.id} similarity to query: {similarity_to_query:.4f}")
                    else:
                        similarity_to_query = 0.0
                    # Apply a dynamic relevance boost based on similarity to query
                    relevance_boost = 1.0 + (similarity_to_query * 1.0)
                    # Apply recency weighting (existing bell curve or time-based decay)
                    if isinstance(mem.created_at, str):
                        created_at_dt = datetime.fromisoformat(mem.created_at.rstrip('Z'))
                    else:
                        created_at_dt = mem.created_at
                    if created_at_dt.tzinfo is None:
                        created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)
                    age_hours = (datetime.now(timezone.utc) - created_at_dt).total_seconds() / 3600
                    recency_score = self._calculate_bell_curve_recency(age_hours)
                    recency_weight = getattr(self.settings, 'episodic_recency_weight', 0.35)
                    final_score = (relevance_boost * _score * (1 - recency_weight)) + (recency_score * recency_weight)
                    # Enhanced time context formatting with additional metadata
                    time_context = self._format_memory_time_context(mem)
                    content = mem.content[:300] + "..." if len(mem.content) > 300 else mem.content
                    # Add more detail to help with temporal references
                    iso_time = mem.metadata.get("created_at_iso")
                    date_str = mem.metadata.get("date_str", "")
                    day_of_week = mem.metadata.get("day_of_week", "")
                    fragment = f"- ({time_context}"
                    if "what time" in query.lower() or "when exactly" in query.lower():
                        fragment += f", exact time: {iso_time}"
                    if "when did we" in query.lower() and date_str:
                        fragment += f", date: {date_str} ({day_of_week})"
                    fragment += f") {content}"
                    episodic_content_list.append((fragment, final_score))
                    logger.info(f"Memory fragment (score={final_score:.3f}): {fragment[:100]}...")
                except Exception as e:
                    logger.warning(f"Error processing episodic memory for summarization: {e}")
                    continue
            episodic_content_list.sort(key=lambda x: x[1], reverse=True)
            top_fragments = [fragment for fragment, _score in episodic_content_list[:10]]
            episodic_content = "\n".join(top_fragments) if top_fragments else "No relevant conversation history available."
            logger.info(f"Final episodic content for prompt (length={len(episodic_content)}): {episodic_content[:200]}...")
        else:
            episodic_content = "No relevant conversation history available."
            logger.warning("No episodic memories received for summarization")
        # === End Enhanced Episodic Memory Formatting ===

        # --- Retrieve immediate previous turn and slightly recent memories ---
        immediate_previous_turn_memory = await self._retrieve_immediate_previous_turn(window_id=window_id)
        slightly_recent_episodic_memories = await self._retrieve_slightly_recent_episodic_memories(window_id=window_id)

        # --- Define semantic_prompt HERE ---
        semantic_prompt = f"""
        You are an AI memory processor helping a conversational AI recall knowledge.

        **User Query:** "{query}"

        **Retrieved Knowledge Fragments:**
        {semantic_content or "**No relevant knowledge found.**"}

        **Task:**
        - Synthesize this knowledge like a human would recall facts and information.
        - Keep it concise but informative (max 150 words).
        - Prioritize details that are most relevant to the current query.
        - Connect related concepts naturally.
        - Make it feel like knowledge a person would have, not like search results.
        - If no knowledge is provided, respond with "**No relevant background knowledge available.**"

        **Output just the synthesized knowledge:**
        """

        # --- Define learned_prompt HERE ---
        learned_prompt = f"""
        You are an AI memory processor helping a conversational AI recall learned insights.

        **User Query:** "{query}"

        **Retrieved Insights:**
        {learned_content or "**No relevant insights found.**"}

        **Task:**
        - Synthesize these insights like a human would recall their own conclusions and lessons.
        - Keep it concise and focused (max 80 words).
        - Prioritize insights that are most relevant to the current query.
        - Make it feel like a personal reflection rather than a data report.
        - If no insights are provided, respond with "**No relevant insights available yet.**"

        **Output just the synthesized insights:**
        """

        # --- Define episodic_prompt HERE ---
        episodic_content_for_prompt = ""
        episodic_prompt = f"""
You are an AI memory processor recalling past conversations.

**User Query:** "{query}"
"""
        if immediate_previous_turn_memory:
            episodic_prompt += f"""
**Immediate Previous Turn (Most Recent Conversation):**
        - {immediate_previous_turn_memory[0].content[:300]}...\n
        """
        if slightly_recent_episodic_memories:
            episodic_prompt += f"""**Slightly Recent Conversation History (Last Few Minutes):**\n"""
            episodic_prompt += "\\n".join([f"- ({self._format_memory_time_context(mem)}): " + mem.content[:200] + "..." for mem, score in slightly_recent_episodic_memories]) + "\\n"
        if episodic_content:
            episodic_prompt += f"""
**Older Conversation History:**\n
        """
            episodic_prompt += episodic_content
        episodic_prompt += f"""
        **Task:**
        - Summarize relevant past conversations concisely (max 150 words).
        - Prioritize the most relevant content to the current query.
        - Focus on what was discussed rather than what wasn't discussed.
        - If you find ANY substantive content, include it in your summary.
        - Avoid starting with phrases like "I don't recall" - instead, focus on what you DO find.
        {f"- Focus on conversations from {time_expression}." if time_expression else ""}
        {temporal_context if temporal_context else ""}

        **Output the summarized memory:**
        """

        semantic_summary_task = asyncio.create_task(
            self.llm_service.generate_gpt_response_async(semantic_prompt, temperature=0.5)
        ) if semantic_content else None
        episodic_summary_task = asyncio.create_task(
            self.llm_service.generate_gpt_response_async(episodic_prompt, temperature=0.5)
        ) if episodic_content or immediate_previous_turn_memory or slightly_recent_episodic_memories else None
        learned_summary_task = asyncio.create_task(
            self.llm_service.generate_gpt_response_async(learned_prompt, temperature=0.5)
        ) if learned_content else None

        tasks_to_gather = [task for task in [semantic_summary_task, episodic_summary_task, learned_summary_task] if task is not None]
        summaries = {}
        results = await asyncio.gather(
            *tasks_to_gather,
            return_exceptions=True
        )
        task_map = {"semantic": semantic_summary_task, "episodic": episodic_summary_task, "learned": learned_summary_task}
        for memory_type, task, result in zip(task_map.keys(), task_map.values(), results):
            if isinstance(result, Exception):
                logger.error(f"Summarization task for {memory_type} failed: {result}")
                summaries[memory_type] = f"Error processing {memory_type} memories."
            elif task is not None:
                summaries[memory_type] = result.strip()
                logger.info(f"Memory summarization for {memory_type} completed: {summaries[memory_type][:100]}...")
            else:
                summaries[memory_type] = f"No relevant memories for {memory_type}."
        if "episodic" in summaries:
            negative_phrases = ["i don't recall", "no relevant", "haven't discussed", "i'm sorry"]
            if (len(summaries["episodic"]) < 100 and 
                any(phrase in summaries["episodic"].lower() for phrase in negative_phrases)):
                summaries["episodic"] = "I don't recall any relevant conversations from that time."
            if "episodic" in summaries and "i don't recall" in summaries["episodic"].lower():
                logger.info("Episodic summary contains 'I don't recall' or similar phrase")
            else:
                logger.info(f"Returning non-empty episodic summary: {summaries.get('episodic', '')[:100]}...")
        logger.info(f"Returning summaries - Episodic: {summaries.get('episodic', '')[:100]}...")
        if 'episodic' in summaries and summaries['episodic'] is None:
            summaries['episodic'] = "No relevant conversation history available."
        if "episodic" not in summaries:
            summaries["episodic"] = "No relevant conversation history available."
        logger.info(f"Returning summaries - Episodic: {summaries.get('episodic', '')[:100]}...")
        return summaries


    async def store_interaction_enhanced(self, query: str, response: str, window_id: Optional[str] = None) -> Memory:
        """Stores a user interaction (query + response) as a new episodic memory with consistent timestamp formats."""
        try:
            logger.info(f"Storing interaction with query: '{query[:50]}...' and response: '{response[:50]}...'")

            # Check for "I don't recall" responses and skip storing
            if "i don't recall" in response.lower() or "no relevant" in response.lower() or "haven't discussed" in response.lower():
                logger.info(f"Not storing 'I don't recall' response to avoid polluting memory")
                return None

            # Combine query and response for embedding. Correct format.
            interaction_text = f"User: {query}\nAssistant: {response}"

            # Check for duplicates *before* creating the memory object
            if await self._check_recent_duplicate(interaction_text):
                logger.warning("Duplicate interaction detected. Skipping storage.")
                return None  # Or raise an exception, depending on desired behavio

            semantic_vector = await self.vector_operations.create_episodic_memory_vector(interaction_text)
            memory_id = f"mem_{uuid.uuid4().hex}"

            # Get current time for enhanced time metadata with proper UTC timezone
            current_time = datetime.now(timezone.utc)

            # Generate consistent timestamp formats that work reliably with filtering
            # 1. ISO 8601 format with explicit Z notation for UTC
            current_time_iso = current_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

            # 2. Unix timestamp (seconds since epoch) for range queries
            current_time_unix = int(current_time.timestamp())

            # Add more granular time_of_day periods
            hour = current_time.hour
            if 5 <= hour < 9:
                time_of_day = "early morning"
            elif 9 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 14:
                time_of_day = "noon"
            elif 14 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

            # Create a standardized metadata dictionary with consistent timestamp formats
            metadata = {
                "content": interaction_text,
                "memory_type": "EPISODIC",

                # IMPORTANT: Three different timestamp representations for different purposes
                "created_at_iso": current_time_iso,         # ISO string with Z notation
                "created_at_unix": current_time_unix,       # Unix timestamp for range filtering
                "created_at": current_time_iso,             # Backward compatibility

                "window_id": window_id,
                "source": "user_interaction",

                # Enhanced time metadata for better temporal queries
                "time_of_day": time_of_day,
                "day_of_week": current_time.strftime("%A").lower(),  # Use lowercase for consistent queries
                "date_str": current_time.strftime("%Y-%m-%d"),

                # Boolean flags for efficient filtering
                "is_morning": 5 <= hour < 12,
                "is_afternoon": 12 <= hour < 17,
                "is_evening": 17 <= hour < 24,
                "is_today": True,  # Will be useful for "today" queries
                "is_very_recent": True,

                # Add hour for more specific filtering
                "hour_of_day": current_time.hour,

                # Add additional useful time flags
                "is_weekday": 0 <= current_time.weekday() <= 4,
                "is_weekend": current_time.weekday() >= 5,
                "month": current_time.strftime("%B").lower(),
                "year": current_time.year,
                "week_of_year": current_time.isocalendar()[1]
            }

            # Create the Memory object
            # NOTE: We now use the ISO string format for created_at to maintain consistency
            memory = Memory(
                id=memory_id,
                content=interaction_text,
                memory_type=MemoryType.EPISODIC,
                created_at=current_time_iso,  # Use ISO string for consistent format
                metadata=metadata,
                window_id=window_id,
                semantic_vector=semantic_vector
            )

            # Log timestamp details for debugging
            logger.info(f"Creating memory with timestamps - ISO: {current_time_iso}, Unix: {current_time_unix}")

            # Store in Pinecone
            success = await self.pinecone_service.create_memory(
                memory_id=memory.id,
                vector=semantic_vector,
                metadata=metadata
            )

            if not success:
                raise MemoryOperationError("Failed to store memory in vector database")

            logger.info(f"Episodic memory stored successfully: {memory_id}")
            return memory

        except Exception as e:
            logger.error(f"Failed to store interaction: {e}", exc_info=True)
            raise MemoryOperationError("store_interaction", str(e))

    async def handle_temporal_query(self, query, window_id=None):
        """Dedicated handler for temporal queries."""
        # Detect temporal expressions with enhanced regex
        temporal_patterns = [
            r"(?:when|what time|how long ago) (?:did|have) (?:we|you|I) (?:talk|discuss|mention|say) (?:about)? (.+)"
        ]

    def _format_memory_time_context(self, memory: MemoryResponse) -> str:
        """Helper function to format time context for a memory, reused from memory_summarization_agent."""
        time_context = ""
        if hasattr(memory, 'metadata') and memory.metadata:

            if memory.metadata.get("is_very_recent", False):
                created_at = datetime.fromisoformat(memory.metadata.get("created_at_iso", "").rstrip('Z'))
                if datetime.now(timezone.utc) - created_at <= timedelta(minutes=10):
                    # Still very recent
                    minutes_ago = (datetime.now(timezone.utc) - created_at).total_seconds() / 60
                    return "just now" if minutes_ago < 2 else "a few minutes ago"
                else:
                    # No longer very recent, update flag
                    memory.metadata["is_very_recent"] = False
            # Get time metadata from memory
            time_of_day = memory.metadata.get('time_of_day', '')
            day_of_week = memory.metadata.get('day_of_week', '')
            date_str = memory.metadata.get('date_str', '')
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Parse the timestamp
            if isinstance(memory.created_at, str):
                created_at_dt = datetime.fromisoformat(memory.created_at.rstrip('Z'))
            else:
                created_at_dt = memory.created_at

            # Ensure timezone information
            if created_at_dt.tzinfo is None:
                created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            minutes_ago = (now - created_at_dt).total_seconds() / 60

            # Modified time_context formatting
            if created_at_dt + timedelta(minutes=10) > now:
                time_context = "just now" if minutes_ago < 2 else f"{int(minutes_ago)} minutes ago"
            elif date_str == today_str:
                time_context = f"earlier {time_of_day}"
            elif day_of_week and time_of_day:
                time_context = f"{day_of_week} {time_of_day}"
            elif time_of_day:
                time_context = f"the {time_of_day}"

        if not time_context:
            if memory.time_ago:
                time_context = memory.time_ago
            elif hasattr(memory, 'created_at'):
                # Check if created_at is a string or datetime
                if isinstance(memory.created_at, str):
                    created_at_dt = datetime.fromisoformat(memory.created_at.rstrip('Z'))
                else:
                    # Already a datetime object
                    created_at_dt = memory.created_at

                # Ensure timezone info
                if created_at_dt.tzinfo is None:
                    created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

                time_context = self._format_time_ago(created_at_dt) or "recently"
        return time_context

    async def _retrieve_immediate_previous_turn(self, window_id: Optional[str] = None) -> Optional[List[MemoryResponse]]:
        """Retrieves the single most recent episodic memory for the current window."""
        try:
            # Add window_id filter
            pinecone_filter = {
                    "memory_type": "EPISODIC",
                }
            if window_id:
              pinecone_filter["window_id"] = window_id

            query_request = QueryRequest(
                prompt="previous turn context", # Dummy prompt - we are sorting by time, not semantic relevance
                top_k=1, # Only need the most recent one
                memory_type=MemoryType.EPISODIC,
                request_metadata=RequestMetadata(operation_type=OperationType.QUERY)
            )
            query_response = await self._query_memory_type(request=query_request, pinecone_filter=pinecone_filter) # Use _query_memory_type for efficiency

            if query_response.matches:
                # Don't modify the objects, just return them as is
                # If we need to process timestamps, do it without adding new attributes
                for mem in query_response.matches:
                    if hasattr(mem, 'created_at'):
                        # Process the timestamp locally without modifying the object
                        if isinstance(mem.created_at, str):
                            local_created_at = datetime.fromisoformat(mem.created_at.rstrip('Z'))
                        else:
                            local_created_at = mem.created_at

                        # Ensure timezone info
                        if local_created_at.tzinfo is None:
                            local_created_at = local_created_at.replace(tzinfo=timezone.utc)

                        # Use local_created_at for any calculations here
                        # But don't add it back to the mem object

                return query_response.matches[:1]  # Return as a list (consistent with other memory vars)
            else:
                return None  # No previous turn memory found

        except Exception as e:
            logger.error(f"Error retrieving immediate previous turn memory: {e}")
            return None

    async def _retrieve_slightly_recent_episodic_memories(self, window_id: Optional[str] = None, time_window_minutes: int = 5) -> Optional[List[Tuple[MemoryResponse, float]]]:
        """Retrieves episodic memories from the last few minutes, sorted by recency and relevance."""
        try:
            # Add window_id filter
            pinecone_filter = {
              "memory_type": "EPISODIC",
            }
            if window_id:
              pinecone_filter["window_id"] = window_id

            current_time = datetime.now(timezone.utc)
            recent_past = current_time - timedelta(minutes=time_window_minutes)
            recent_past_timestamp = int(recent_past.timestamp())

            query_request = QueryRequest(
                prompt="recent conversation context",
                top_k=10,
                memory_type=MemoryType.EPISODIC,
                request_metadata=RequestMetadata(operation_type=OperationType.QUERY)
            )
            query_response = await self._query_memory_type(request=query_request, pinecone_filter=pinecone_filter) # Use _query_memory_type

            recent_memories_with_scores = []
            if query_response.matches:
                for i, memory in enumerate(query_response.matches):
                    created_at_unix = memory.metadata.get("created_at_unix")
                    if created_at_unix and created_at_unix >= recent_past_timestamp:
                        # Calculate a simple recency score (you can refine this)
                        created_at_dt = datetime.fromtimestamp(created_at_unix, tz=timezone.utc) # Make timezone aware
                        recency_score = 1.0 - ((current_time - created_at_dt).total_seconds() / (time_window_minutes * 60))
                        # Combine with the original similarity score (you might want to adjust weights)
                        final_score = (query_response.similarity_scores[i] * 0.7) + (recency_score * 0.3)
                        recent_memories_with_scores.append((memory, final_score))

            # Sort by final score (descending) - Most relevant and recent first
            recent_memories_with_scores.sort(key=lambda x: x[1], reverse=True)
            return recent_memories_with_scores[:5] # Return top 5 recent memories (adjust as needed)

        except Exception as e:
            logger.error(f"Error retrieving slightly recent episodic memories: {e}")
            return None

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
                    "created_at_unix": {"$gte": recent_timestamp}  # Changed from created_at to created_at_unix
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

    def _select_memories_for_prompt(
        self,
        memories: List[Tuple[MemoryResponse, float]],
        max_tokens: int = 1000,
        max_memories: int = 10,
        min_score: float = 0.1
    ) -> List[str]:
        """
        Select memories to include in prompt based on token budget and relevance.

        Args:
            memories: List of (memory, score) tuples
            max_tokens: Maximum token budget for all memories
            max_memories: Maximum number of memories to include
            min_score: Minimum score threshold

        Returns:
            List of formatted memory contents
        """
        if not memories:
            return []

        # Sort by score (descending)
        sorted_memories = sorted(memories, key=lambda x: x[1], reverse=True)

        # Filter by minimum score
        filtered_memories = [(m, s) for m, s in sorted_memories if s >= min_score]

        # Select memories within token budget and count limit
        selected = []
        total_tokens = 0

        for memory, score in filtered_memories[:max_memories]:
            # Estimate tokens (4 chars ≈ 1 token as rough estimate)
            est_tokens = len(memory.content) // 4

            # Skip if this would exceed our budget
            if total_tokens + est_tokens > max_tokens:
                break

            # Format the memory content
            formatted = memory.content

            # Add type-specific formatting if needed
            if memory.memory_type == MemoryType.EPISODIC:
                # Use the pre-computed time_ago field if available, otherwise calculate it
                if hasattr(memory, 'time_ago') and memory.time_ago:
                    time_ago = memory.time_ago
                else:
                    # Check the type of memory.created_at
                    if isinstance(memory.created_at, str):
                        created_at = datetime.fromisoformat(memory.created_at.rstrip('Z'))
                    else:
                        # Already a datetime object
                        created_at = memory.created_at

                    # Ensure timezone information
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)

                    time_ago = self._format_time_ago(created_at)
                formatted = f"{time_ago}: {formatted}"

            selected.append(formatted)
            total_tokens += est_tokens

        logger.info(f"Selected {len(selected)}/{len(memories)} memories for prompt " +
                    f"(~{total_tokens} tokens, {max_tokens} max)")

        return selected

    def _format_time_ago(self, timestamp: datetime) -> str:
            """Format a timestamp as a human-readable time ago string, potentially including AM/PM for recent times."""
            now = datetime.now(timezone.utc)
            diff = now - timestamp

            if diff.total_seconds() < 600:  # 30 minutes - Keep returning None for very recent
                minutes = max(1, int(diff.total_seconds() / 60))
                return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

            # For memories within the last few hours, consider showing AM/PM time
            if diff.total_seconds() < (6 * 3600): # Within 6 hours
                formatted_time = timestamp.strftime("%I:%M %p").lstrip('0') # e.g., "3:45 PM" (no leading zero for hour)
                return f"at {formatted_time} ({self._time_ago_string(diff)})" # e.g., "at 3:45 PM (2 hours ago)"

            # Existing logic for days, months, etc. (unchanged)
            if diff.days > 30:
                months = diff.days // 30
                return f"{months} months ago"
            elif diff.days > 0:
                return f"{diff.days} days ago"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                return f"{hours} hours ago"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                return f"{minutes} minutes ago"
            else:
                return "Just now"
    def _time_ago_string(self, time_difference: timedelta) -> str: # Helper for just the "X time units ago" part
        """Helper function to format just the 'time ago' part of the string."""
        diff = time_difference
        if diff.days > 30:
            months = diff.days // 30
            return f"{months} months ago"
        elif diff.days > 0:
            return f"{diff.days} days ago"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            return f"{hours} hours ago"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"

    async def store_interaction_enhanced(self, query: str, response: str, window_id: Optional[str] = None) -> Memory:
        """Stores a user interaction (query + response) as a new episodic memory with consistent timestamp formats."""
        try:
            logger.info(f"Storing interaction with query: '{query[:50]}...' and response: '{response[:50]}...'")

            # Combine query and response for embedding. Correct format.
            interaction_text = f"User: {query}\nAssistant: {response}"

            # Check for duplicates *before* creating the memory object
            if await self._check_recent_duplicate(interaction_text):
                logger.warning("Duplicate interaction detected. Skipping storage.")
                return None  # Or raise an exception, depending on desired behavior

            semantic_vector = await self.vector_operations.create_episodic_memory_vector(interaction_text)
            memory_id = f"mem_{uuid.uuid4().hex}"

            # Get current time for enhanced time metadata with proper UTC timezone
            current_time = datetime.now(timezone.utc)

            # Generate consistent timestamp formats that work reliably with filtering
            # 1. ISO 8601 format with explicit Z notation for UTC
            current_time_iso = current_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

            # 2. Unix timestamp (seconds since epoch) for range queries
            current_time_unix = int(current_time.timestamp())

            # Add more granular time_of_day periods
            hour = current_time.hour
            if 5 <= hour < 9:
                time_of_day = "early morning"
            elif 9 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 14:
                time_of_day = "noon"
            elif 14 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

            # Create a standardized metadata dictionary with consistent timestamp formats
            metadata = {
                "content": interaction_text,
                "memory_type": "EPISODIC",

                # IMPORTANT: Three different timestamp representations for different purposes
                "created_at_iso": current_time_iso,         # ISO string with Z notation
                "created_at_unix": current_time_unix,       # Unix timestamp for range filtering
                "created_at": current_time_iso,             # Backward compatibility

                "window_id": window_id,
                "source": "user_interaction",

                # Enhanced time metadata for better temporal queries
                "time_of_day": time_of_day,
                "day_of_week": current_time.strftime("%A").lower(),  # Use lowercase for consistent queries
                "date_str": current_time.strftime("%Y-%m-%d"),

                # Boolean flags for efficient filtering
                "is_morning": 5 <= hour < 12,
                "is_afternoon": 12 <= hour < 17,
                "is_evening": 17 <= hour < 24,
                "is_today": True,  # Will be useful for "today" queries
                "is_very_recent": True,

                # Add hour for more specific filtering
                "hour_of_day": current_time.hour,

                # Add additional useful time flags
                "is_weekday": 0 <= current_time.weekday() <= 4,
                "is_weekend": current_time.weekday() >= 5,
                "month": current_time.strftime("%B").lower(),
                "year": current_time.year,
                "week_of_year": current_time.isocalendar()[1]
            }

            # Create the Memory object
            # NOTE: We now use the ISO string format for created_at to maintain consistency
            memory = Memory(
                id=memory_id,
                content=interaction_text,
                memory_type=MemoryType.EPISODIC,
                created_at=current_time_iso,  # Use ISO string for consistent format
                metadata=metadata,
                window_id=window_id,
                semantic_vector=semantic_vector
            )

            # Log timestamp details for debugging
            logger.info(f"Creating memory with timestamps - ISO: {current_time_iso}, Unix: {current_time_unix}")

            # Store in Pinecone
            success = await self.pinecone_service.create_memory(
                memory_id=memory.id,
                vector=semantic_vector,
                metadata=metadata
            )

            if not success:
                raise MemoryOperationError("Failed to store memory in vector database")

            logger.info(f"Episodic memory stored successfully: {memory_id}")
            return memory

        except Exception as e:
            logger.error(f"Failed to store interaction: {e}", exc_info=True)
            raise MemoryOperationError("store_interaction", str(e))

    async def add_interaction(
        self,
        user_input: str,
        response: str,
        window_id: Optional[str] = None
    ) -> str:
        """Add an interaction as an episodic memory."""
        # Call store_interaction_enhanced but only return the ID
        memory = await self.store_interaction_enhanced(
            query=user_input,
            response=response,
            window_id=window_id
        )
        return memory.id if memory else None
    async def cleanup_old_episodic_memories(self, days: Optional[int] = None) -> int:
        """
        Delete episodic memories older than the specified number of days.

        Args:
            days: Number of days to keep memories (defaults to settings value)

        Returns:
            Number of memories deleted
        """
        try:
            # Use provided days or fall back to settings
            retention_days = days or self.settings.episodic_memory_ttl_days

            # Calculate cutoff timestamp
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            cutoff_timestamp = int(cutoff_date.timestamp())

            logger.info(f"Cleaning up episodic memories older than {retention_days} days (before {cutoff_date.isoformat()})")

            # Query for old episodic memories
            filter_dict = {
                "memory_type": "EPISODIC",
                "created_at_unix": {"$lt": cutoff_timestamp}  # Less than cutoff
            }

            # Generate a vector for the query (doesn't matter what vector since we're filtering by metadata)
            dummy_vector = [0] * 1536  # Use dimensionality of your vectors

            # Get memories to delete (limit batch size for performance)
            old_memories = await self.pinecone_service.query_memories(
                query_vector=dummy_vector,
                top_k=1000,  # Reasonable batch size
                filter=filter_dict,
                include_metadata=False  # Don't need full metadata for deletion
            )

            # Delete each memory
            deleted_count = 0
            for memory_data, _ in old_memories:
                memory_id = memory_data["id"]
                success = await self.delete_memory(memory_id)
                if success:
                    deleted_count += 1

            logger.info(f"Successfully deleted {deleted_count} old episodic memories")
            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old episodic memories: {e}", exc_info=True)
            return 0

    async def health_check(self):
        """Checks the health of the memory system."""
        return {"status": "healthy"}