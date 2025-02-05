# models.py
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from uuid import uuid4

class MemoryType(str, Enum):
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"

class CreateMemoryRequest(BaseModel):
    """Request model for creating a new memory"""
    content: str = Field(..., description="The content of the memory")
    memory_type: MemoryType = Field(..., description="Type of memory (EPISODIC or SEMANTIC)")
    metadata: Dict = Field(default_factory=dict, description="Optional metadata for the memory")
    window_id: Optional[str] = Field(None, description="Optional window ID for context grouping")

class Memory(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    created_at: str
    metadata: Dict[str, Any]
    window_id: Optional[str] = None
    semantic_vector: Optional[List[float]] = None

    def to_response(self) -> 'MemoryResponse':
        """Convert Memory to MemoryResponse"""
        return MemoryResponse(
            id=self.id,
            content=self.content,
            memory_type=self.memory_type,
            created_at=self.created_at,
            metadata=self.metadata,
            window_id=self.window_id,
            semantic_vector=self.semantic_vector
        )

class MemoryResponse(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    created_at: str
    metadata: Dict[str, Any]
    window_id: Optional[str] = None
    semantic_vector: Optional[List[float]] = None

    @classmethod
    def from_memory(cls, memory: Memory) -> 'MemoryResponse':
        """Create MemoryResponse from Memory object"""
        return cls(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type,
            created_at=memory.created_at,
            metadata=memory.metadata,
            window_id=memory.window_id,
            semantic_vector=memory.semantic_vector
        )

class QueryRequest(BaseModel):
    """Request model for querying memories"""
    query: str
    prompt: str = Field(..., description="The query prompt")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of memories to retrieve")
    window_id: Optional[str] = Field(None, description="Optional window ID to filter context")

class QueryResponse(BaseModel):
    """Response model for memory queries"""
    matches: List[MemoryResponse] = Field(..., description="List of matching memories")
    similarity_scores: List[float] = Field(..., description="Similarity scores for each match")
    response: Optional[str] = Field(None, description="Generated response from the LLM")