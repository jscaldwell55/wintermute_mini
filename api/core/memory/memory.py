from datetime import datetime, timezone, timedelta
import uuid
from typing import List, Dict, Optional, Any, Tuple
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
from api.utils.utils import normalize_timestamp  # 

logging.basicConfig(level=logging.INFO)
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

        from api.core.memory.keyword_index import KeywordIndex
        self.keyword_index = KeywordIndex()
        logger.info("Keyword index initialized")

    async def initialize(self) -> bool:
        """Initialize the memory system and its components."""
        try:
            # Verify vector operations initialization (if it has an initialize method)
            if hasattr(self.vector_operations, 'initialize'):
                await self.vector_operations.initialize()

            # Verify Pinecone initialization (if it has an initialize method)
            if hasattr(self.pinecone_service, 'initialize'):
                await self.pinecone_service.initialize()

            if not self._initialized and hasattr(self, 'keyword_index'):
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
            if hasattr(self, 'keyword_index'):
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
            
            # Generate query vector
            query_vector = await self.vector_operations.create_semantic_vector(request.prompt)
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
            if keywords:
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
            keyword_results = []
            if "keyword" in task_results:
                keyword_results = task_results["keyword"]
                logger.info(f"Received {len(keyword_results)} raw results from keyword search.")
            
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

            # Combine both types of results
            combined_results = await self._combine_search_results(
            vector_results=vector_results,
            keyword_results=keyword_results
        )
        
            # Safety check to prevent iterating over None - ADD THIS RIGHT HERE
            if combined_results is None:
                logger.warning("Combined results is None, using empty list")
                combined_results = []
                
            matches = []
            similarity_scores = []
            current_time = datetime.utcnow()

            for memory_data, similarity_score in combined_results:
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
            
            # Process keyword results
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
                            "created_at": memory.created_at,  # This should be a datetime object
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
            
            # Log result counts
            logger.info(f"Combined results: {len(combined_results)} memories")
            
            return combined_results
        
        except Exception as e:
            logger.error(f"Error combining search results: {e}", exc_info=True)
            # Return an empty list if there's an error, not None
            return []

        

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
            if hasattr(self, 'keyword_index') and memory:
                # Add to keyword index
                self.keyword_index.add_to_index(memory)
                logger.info(f"Added interaction memory {memory_id} to keyword index")

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

    async def health_check(self):
        """Checks the health of the memory system."""
        return {"status": "healthy"}