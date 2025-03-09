from datetime import datetime, timezone, timedelta
import uuid
from typing import List, Dict, Optional, Any, Tuple, Union
import logging
import math
import asyncio
from pydantic import BaseModel, Field, field_validator
from enum import Enum

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

        # Only initialize keyword index if enabled in settings
        keyword_search_enabled = getattr(self.settings, 'keyword_search_enabled', False)
        if keyword_search_enabled:
            from api.core.memory.keyword_index import KeywordIndex
            self.keyword_index = KeywordIndex()
            logger.info("Keyword index initialized")
        else:
            logger.info("Keyword index disabled via settings")

    async def initialize(self) -> bool:
        """Initialize the memory system and its components."""
        try:
            # Verify vector operations initialization (if it has an initialize method)
            if hasattr(self.vector_operations, 'initialize'):
                await self.vector_operations.initialize()

            # Verify Pinecone initialization (if it has an initialize method)
            if hasattr(self.pinecone_service, 'initialize'):
                await self.pinecone_service.initialize()

            if not self._initialized and hasattr(self, 'keyword_index') and getattr(self.settings, 'keyword_search_enabled', False):
                try:
                    # Get a sample of memories to build the initial index
                    # Consider limiting this to avoid overloading during startup
                    memory_samples = await self.pinecone_service.sample_memories(
                        limit=1000,  # Adjust based on your system's capabilities
                        include_vector=False  # We don't need vectors for keyword index
                    )
                    if memory_samples:
                        logger.info(f"Building keyword index with {len(memory_samples)} initial memories")
                        await self.keyword_index.build_index(memory_samples)
                except Exception as e:
                    logger.warning(f"Initial keyword index build failed, will build incrementally: {e}")

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
            if hasattr(self, 'keyword_index') and getattr(self.settings, 'keyword_search_enabled', False):
                # Add to keyword index
                self.keyword_index.add_to_index(memory)
                logger.info(f"Added memory {memory_id} to keyword index")
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
        """Delete a memory by ID and remove it from keyword index."""
        try:
            logger.info(f"Deleting memory with ID: {memory_id}")
            
            # Remove from keyword index first
            if hasattr(self, 'keyword_index'):
                self.keyword_index._remove_from_index(memory_id)
                logger.info(f"Removed memory {memory_id} from keyword index")
                
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
            
            # Extract keywords for keyword search
            keywords = self._extract_keywords(request.prompt)
            logger.info(f"Extracted keywords: {keywords}")
            
            # Determine memory type filter
            memory_type_value = request.memory_type.value if hasattr(request, 'memory_type') and request.memory_type else None
            logger.info(f"Using memory type filter: {memory_type_value}")
            
            # Execute vector and keyword searches in parallel
            tasks = []
            
            # Add vector search task
            if not hasattr(request, 'memory_type') or request.memory_type is None:
                # Default case - query all types
                pinecone_filter = {}
                if request.window_id:
                    pinecone_filter["window_id"] = request.window_id
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
                seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
                seven_days_ago_timestamp = int(seven_days_ago.timestamp())
                
                pinecone_filter = {
                    "memory_type": "EPISODIC",
                    "created_at": {"$gte": seven_days_ago_timestamp}
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
                        "created_at": {"$gte": seven_days_ago_timestamp},
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
                if request.window_id:
                    pinecone_filter["window_id"] = request.window_id
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
                if request.window_id:
                    pinecone_filter["window_id"] = request.window_id
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
            
            # Add keyword search task if we have keywords
            if keywords and getattr(self.settings, 'keyword_search_enabled', False):
                keyword_task = asyncio.create_task(
                    self.keyword_index.search(
                        keywords=keywords,
                        limit=self.settings.keyword_search_top_k,
                        memory_type=memory_type_value,
                        window_id=request.window_id
                    )
                )
                tasks.append(("keyword", keyword_task))
            
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
            
            # Process keyword results if any
            if "keyword" in task_results and attempt_count < max_attempts:
                attempt_count += 1
                keyword_results = task_results["keyword"]
                logger.info(f"Received {len(keyword_results)} raw results from keyword search.")
            else:
                keyword_results = []

            
            # Combine both types of results
            combined_results = await self._combine_search_results(
                vector_results=vector_results,
                keyword_results=keyword_results
            )
            
            matches = []
            similarity_scores = []
            current_time = datetime.utcnow()

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
                        created_at = datetime.fromisoformat(normalize_timestamp(created_at_raw))
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
                    
                        if age_hours <= self.settings.episodic_recent_hours:  # Changed from recent_boost_hours
                            # Linear decrease from 1.0 to 0.7 during the boost period
                            recency_score = 1.0 - (age_hours / self.settings.episodic_recent_hours) * 0.3
                        else:
                            # Exponential decay for older memories
                            max_age_hours = self.settings.episodic_max_age_days * 24  # Changed from max_age_days
                            relative_age = (age_hours - self.settings.episodic_recent_hours) / (max_age_hours - self.settings.episodic_recent_hours)
                            # Exponential decay from 0.7 to 0.1
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
                        # For semantic memories, apply the semantic memory weight
                        final_score = similarity_score * self.settings.semantic_memory_weight
                        logger.info(f"Memory ID {memory_data['id']} (SEMANTIC): Raw={similarity_score:.3f}, Final={final_score:.3f}")
                    
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
                sorted_results = sorted(zip(matches, similarity_scores), key=lambda x: x[1], reverse=True)[:request.top_k]
                matches, similarity_scores = zip(*sorted_results)
            else:
                matches, similarity_scores = [], []

            return QueryResponse(
                matches=list(matches),
                similarity_scores=list(similarity_scores),
            )

        except Exception as e:
            logger.error(f"Error querying memories: {e}", exc_info=True)
            raise MemoryOperationError(f"Failed to query memories: {str(e)}")
        
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract significant keywords from the query"""
        # Tokenize and normalize
        tokens = [t.lower() for t in query.split()]
        
        # Remove stopwords
        stopwords = {
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
            'about', 'like', 'of', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'and',
            'or', 'if', 'then', 'else', 'when', 'what', 'how', 'where', 'why'
        }
        keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        logger.info(f"Extracted keywords from query: {keywords}")
        return keywords
    
    async def batch_query_memories(
        self, 
        query: str, 
        window_id: Optional[str] = None, 
        top_k_per_type: Union[int, Dict[MemoryType, int]] = 5,  # Accept either int or dict
        enable_keyword_search: bool = True,
        request_metadata: Optional[RequestMetadata] = None
    ) -> Dict[MemoryType, List[Tuple[MemoryResponse, float]]]:
        """
        Query all memory types in a single batched operation.
        """
        logger.info(f"Starting batch memory query: '{query[:50]}...'")
        
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
        keywords = self._extract_keywords(query) if enable_keyword_search else []
        
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
                enable_keyword_search=enable_keyword_search,
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
    
    async def _query_memory_type(
        self, 
        request: QueryRequest,
        pre_computed_vector: Optional[List[float]] = None
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
                seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
                seven_days_ago_timestamp = int(seven_days_ago.timestamp())
                pinecone_filter = {
                    "memory_type": "EPISODIC",
                    "created_at": {"$gte": seven_days_ago_timestamp}
                }
                if request.window_id:
                    pinecone_filter["window_id"] = request.window_id
            elif request.memory_type == MemoryType.LEARNED:
                pinecone_filter = {"memory_type": "LEARNED"}
                if request.window_id:
                    pinecone_filter["window_id"] = request.window_id
            else:
                pinecone_filter = {}
                if request.window_id:
                    pinecone_filter["window_id"] = request.window_id
            
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
                    
                    # Convert timestamp to datetime
                    if isinstance(created_at_raw, str):
                        created_at = datetime.fromisoformat(normalize_timestamp(created_at_raw))
                    else:
                        created_at = created_at_raw

                    # Get memory type and calculate final score based on memory type
                    memory_type = memory_data["metadata"].get("memory_type", "UNKNOWN")
                    final_score = similarity_score
                    
                    # Apply memory-type specific scoring
                    if memory_type == "EPISODIC":
                        # Adjust for timezone if needed
                        if current_time.tzinfo is None and created_at.tzinfo is not None:
                            current_time = current_time.replace(tzinfo=timezone.utc)
                        elif current_time.tzinfo is not None and created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        
                        # Calculate age and recency score
                        age_hours = (current_time - created_at).total_seconds() / (60*60)
                        
                        if age_hours <= self.settings.episodic_recent_hours:
                            recency_score = 1.0 - (age_hours / self.settings.episodic_recent_hours) * 0.3
                        else:
                            max_age_hours = self.settings.episodic_max_age_days * 24
                            relative_age = (age_hours - self.settings.episodic_recent_hours) / (max_age_hours - self.settings.episodic_recent_hours)
                            recency_score = 0.7 * (0.1/0.7) ** relative_age
                        
                        recency_score = max(0.0, min(1.0, recency_score))
                        
                        # Combine relevance and recency
                        relevance_weight = 1 - self.settings.episodic_recency_weight
                        combined_score = (
                            relevance_weight * similarity_score + 
                            self.settings.episodic_recency_weight * recency_score
                        )
                        
                        final_score = combined_score * self.settings.episodic_memory_weight
                        
                        logger.info(f"Memory ID {memory_data['id']} (EPISODIC): Raw={similarity_score:.3f}, " +
                            f"Age={age_hours:.1f}h, Recency={recency_score:.3f}, Final={final_score:.3f}")
                    
                    elif memory_type == "SEMANTIC":
                        final_score = similarity_score * self.settings.semantic_memory_weight
                        logger.info(f"Memory ID {memory_data['id']} (SEMANTIC): Raw={similarity_score:.3f}, Final={final_score:.3f}")
                    
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
                    matches.append(memory_response)
                    similarity_scores.append(final_score)

                except Exception as e:
                    logger.error(f"Error processing memory {memory_data.get('id', 'unknown')}: {e}")
                    continue

            # Sort results by final score
            if matches and similarity_scores:
                sorted_results = sorted(zip(matches, similarity_scores), key=lambda x: x[1], reverse=True)[:request.top_k]
                matches, similarity_scores = zip(*sorted_results)
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

    async def memory_summarization_agent(
        self,
        query: str,
        semantic_memories: List[Tuple[MemoryResponse, float]],
        episodic_memories: List[Tuple[MemoryResponse, float]],
        learned_memories: List[Tuple[MemoryResponse, float]]
    ) -> Dict[str, str]:
        """
        Use an LLM to process retrieved memories into more human-like summaries.
        
        Args:
            query: The user query
            semantic_memories: Semantic memories with scores
            episodic_memories: Episodic memories with scores
            learned_memories: Learned memories with scores
            
        Returns:
            Dictionary with summarized memories by type
        """
        logger.info(f"Processing memories with summarization agent for query: {query[:50]}...")
        
        # Format memories for summarization
        semantic_content = "\n".join([f"- {mem.content[:300]}..." if len(mem.content) > 300 
                                    else f"- {mem.content}" for mem, _ in semantic_memories])
        
        episodic_content = "\n".join([f"- ({self._format_time_ago(datetime.fromisoformat(mem.created_at.rstrip('Z')))}) "
                                    f"{mem.content[:300]}..." if len(mem.content) > 300 
                                    else f"- ({self._format_time_ago(datetime.fromisoformat(mem.created_at.rstrip('Z')))}) {mem.content}" 
                                    for mem, _ in episodic_memories])
        
        learned_content = "\n".join([f"- {mem.content[:300]}..." if len(mem.content) > 300 
                                    else f"- {mem.content}" for mem, _ in learned_memories])
        
        # Create separate prompts for each memory type
        semantic_prompt = f"""
        You are an AI memory processor helping a conversational AI recall knowledge.
        
        **User Query:** "{query}"
        
        **Retrieved Knowledge Fragments:**
        {semantic_content or "No relevant knowledge found."}
        
        **Task:**
        - Synthesize this knowledge like a human would recall facts and information.
        - Keep it concise but informative (max 150 words).
        - Prioritize details that are most relevant to the current query.
        - Connect related concepts naturally.
        - Make it feel like knowledge a person would have, not like search results.
        - If no knowledge is provided, respond with "No relevant background knowledge available."
        
        **Output just the synthesized knowledge:**
        """
        
        episodic_prompt = f"""
        You are an AI memory processor helping a conversational AI recall past conversations.
        
        **User Query:** "{query}"
        
        **Retrieved Conversation Fragments:**
        {episodic_content or "No relevant conversations found."}
        
        **Task:**
        - Summarize these past conversations like a human would recall them.
        - Keep it concise (max 100 words).
        - Prioritize conversations that are most relevant to the current query.
        - Make it feel like natural memories of past interactions, including time references.
        - Focus on what was discussed rather than listing timestamps.
        - If no conversations are provided, respond with "No relevant conversation history available."
        
        **Output just the summarized memory:**
        """
        
        learned_prompt = f"""
        You are an AI memory processor helping a conversational AI recall learned insights.
        
        **User Query:** "{query}"
        
        **Retrieved Insights:**
        {learned_content or "No relevant insights found."}
        
        **Task:**
        - Synthesize these insights like a human would recall their own conclusions and lessons.
        - Keep it concise and focused (max 80 words).
        - Prioritize insights that are most relevant to the current query.
        - Make it feel like a personal reflection rather than a data report.
        - If no insights are provided, respond with "No relevant insights available yet."
        
        **Output just the synthesized insights:**
        """
        
        # Process each memory type in parallel
        summaries = {}
        
        # Only summarize if we have content
        tasks = []
        if semantic_content:
            tasks.append(("semantic", asyncio.create_task(
                self.llm_service.generate_gpt_response_async(semantic_prompt, temperature=0.7)
            )))
        else:
            summaries["semantic"] = "No relevant background knowledge available."
            
        if episodic_content:
            tasks.append(("episodic", asyncio.create_task(
                self.llm_service.generate_gpt_response_async(episodic_prompt, temperature=0.7)
            )))
        else:
            summaries["episodic"] = "No relevant conversation history available."
            
        if learned_content:
            tasks.append(("learned", asyncio.create_task(
                self.llm_service.generate_gpt_response_async(learned_prompt, temperature=0.7)
            )))
        else:
            summaries["learned"] = "No relevant insights available yet."
        
        # Wait for all summarization tasks to complete
        for memory_type, task in tasks:
            try:
                result = await task
                summaries[memory_type] = result.strip()
                logger.info(f"Memory summarization for {memory_type} completed: {result[:100]}...")
            except Exception as e:
                logger.error(f"Memory summarization for {memory_type} failed: {e}")
                summaries[memory_type] = f"Error processing {memory_type} memories."
        
        return summaries

        

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
        
    def _select_memories_for_prompt(
        self, 
        memories: List[Tuple[MemoryResponse, float]], 
        max_tokens: int = 1500,
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
                # For episodic memories, extract creation time
                created_at = datetime.fromisoformat(memory.created_at.rstrip('Z'))
                time_ago = self._format_time_ago(created_at)
                formatted = f"{time_ago}: {formatted}"
                
            selected.append(formatted)
            total_tokens += est_tokens
        
        logger.info(f"Selected {len(selected)}/{len(memories)} memories for prompt " +
                    f"(~{total_tokens} tokens, {max_tokens} max)")
        
        return selected

    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format a timestamp as a human-readable time ago string."""
        now = datetime.now(timezone.utc)
        diff = now - timestamp
        
        # Format based on how long ago
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
        """Stores a user interaction (query + response) as a new episodic memory."""
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
            created_at = int(datetime.now(timezone.utc).timestamp())
            metadata = {
                "content": interaction_text,  # Store the COMBINED text
                "memory_type": "EPISODIC",
                "created_at": created_at, # Use consistent format here
                "window_id": window_id,
                "source": "user_interaction"  # Indicate the source of this memory
            }

            # Create the Memory object first
            memory = Memory(
                id=memory_id,
                content=interaction_text,
                memory_type=MemoryType.EPISODIC,
                created_at=datetime.fromtimestamp(created_at, tz=timezone.utc),  # Convert to datetime
                metadata=metadata,
                window_id=window_id,
                semantic_vector=semantic_vector,
            )

            # Store in Pinecone
            success = await self.pinecone_service.create_memory(
                memory_id=memory.id,
                vector=semantic_vector,
                metadata=metadata
            )
            
            if not success:
                raise MemoryOperationError("Failed to store memory in vector database")

            logger.info(f"Episodic memory stored successfully: {memory_id}")

            # Add to keyword index if available
            if hasattr(self, 'keyword_index') and getattr(self.settings, 'keyword_search_enabled', False):
                # Add to keyword index
                self.keyword_index.add_to_index(memory)
                logger.info(f"Added memory {memory_id} to keyword index")

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
                "created_at": {"$lt": cutoff_timestamp}  # Less than cutoff
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