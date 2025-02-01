from enum import Enum
from typing import Dict, List, Optional
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
    """Core memory model"""
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the memory")
    content: str = Field(..., description="The content of the memory")
    memory_type: MemoryType = Field(..., description="Type of memory (EPISODIC or SEMANTIC)")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")
    semantic_vector: Optional[List[float]] = Field(None, description="Vector embedding of the memory content")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata for the memory")
    window_id: Optional[str] = Field(None, description="Optional window ID for context grouping")

class MemoryResponse(BaseModel):
    """Response model for memory operations"""
    id: str = Field(..., description="Unique identifier of the created/retrieved memory")
    content: str = Field(..., description="Content of the memory")
    memory_type: MemoryType = Field(..., description="Type of memory")
    created_at: str = Field(..., description="Creation timestamp")
    metadata: Dict = Field(..., description="Memory metadata")
    window_id: Optional[str] = Field(None, description="Window ID if specified")

class QueryRequest(BaseModel):
    """Request model for querying memories"""
    prompt: str = Field(..., description="The query prompt")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of memories to retrieve")
    window_id: Optional[str] = Field(None, description="Optional window ID to filter context")

class QueryResponse(BaseModel):
    """Response model for memory queries"""
    matches: List[MemoryResponse] = Field(..., description="List of matching memories")
    similarity_scores: List[float] = Field(..., description="Similarity scores for each match")
    response: Optional[str] = Field(None, description="Generated response from the LLM")