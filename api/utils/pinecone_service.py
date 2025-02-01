import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pinecone import Pinecone
from api.core.memory.models import Memory, MemoryType
import asyncio
from functools import partial

logger = logging.getLogger(__name__)

class PineconeService:
    """Service for interacting with Pinecone vector database."""

    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.index = None
        self.pc = None

    async def initialize_index(self):
        """Initialize Pinecone index."""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine"
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Successfully initialized Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    async def upsert_memory(
        self,
        memory_id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ):
        """Upsert a memory vector to Pinecone."""
        try:
            if not self.index:
                await self.initialize_index()

            # Ensure memory_type is uppercase in metadata
            if "memory_type" in metadata:
                metadata["memory_type"] = metadata["memory_type"].upper()
            
            logger.debug(f"Upserting memory with ID: {memory_id}, metadata: {metadata}")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(
                    self.index.upsert,
                    vectors=[{
                        "id": memory_id,
                        "values": vector,
                        "metadata": metadata
                    }]
                )
            )
            
            logger.debug(f"Successfully upserted memory: {memory_id}")
            
        except Exception as e:
            logger.error(f"Error upserting memory: {e}")
            raise

    async def query_memory(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Memory, float]]:
        """Query memories by vector similarity."""
        try:
            if not self.index:
                await self.initialize_index()

            logger.debug(f"Querying with filter: {filter}")

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                partial(
                    self.index.query,
                    vector=query_vector,
                    top_k=top_k,
                    filter=filter,
                    include_metadata=True
                )
            )

            logger.debug(f"Query returned {len(results.matches)} matches")

            memories_with_scores = []
            for match in results.matches:
                try:
                    memory = self._create_memory_from_result(match)
                    memories_with_scores.append((memory, match.score))
                except Exception as e:
                    logger.error(f"Error processing match {match.id}: {e}")
                    continue

            return memories_with_scores

        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            raise

    def _create_memory_from_result(self, result: Any) -> Memory:
        """Create a Memory object from a Pinecone query result."""
        try:
            metadata = result.metadata
            
            # Log the incoming metadata for debugging
            logger.debug(f"Creating memory from result metadata: {metadata}")
            
            # Get memory type from metadata and ensure it's uppercase
            memory_type_str = metadata.get("memory_type", "EPISODIC").upper()
            
            # Create the memory object with explicit MemoryType enum
            memory = Memory(
                id=result.id,
                content=metadata.get("content", ""),
                memory_type=MemoryType[memory_type_str],
                created_at=metadata.get("created_at", datetime.now().isoformat()),
                semantic_vector=result.values,
                metadata=metadata
            )
            
            logger.debug(f"Successfully created memory object: {memory.id}")
            return memory
            
        except Exception as e:
            logger.error(f"Error creating memory from result: {e}")
            raise

    async def delete_memory(self, memory_id: str):
        """Delete a memory by ID."""
        try:
            if not self.index:
                await self.initialize_index()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(self.index.delete, ids=[memory_id])
            )
            
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise

    async def close_connections(self):
        """Close Pinecone connections."""
        self.index = None
        self.pc = None
        logger.info("Closed Pinecone connections")