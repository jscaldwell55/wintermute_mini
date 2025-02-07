# memory/models.py
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, validator
from datetime import datetime
from uuid import uuid4
from api.utils.config import get_settings


class MemoryType(str, Enum):
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"

class OperationType(str, Enum):
    """Type of operation being performed"""
    QUERY = "QUERY"
    STORE = "STORE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    BATCH = "BATCH"

class RequestMetadata(BaseModel):
    """Enhanced metadata for tracking requests"""
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    operation_type: OperationType
    window_id: Optional[str] = None
    parent_trace_id: Optional[str] = None

class CreateMemoryRequest(BaseModel):
    """Request model for creating a new memory"""
    content: str = Field(..., description="The content of the memory")
    memory_type: MemoryType = Field(..., description="Type of memory (EPISODIC or SEMANTIC)")
    metadata: Dict = Field(default_factory=dict, description="Optional metadata for the memory")
    window_id: Optional[str] = Field(None, description="Optional window ID for context grouping")
    request_metadata: Optional[RequestMetadata] = None

    @validator('content')
    def validate_content_length(cls, v):
        if not v:
            raise ValueError("Content cannot be empty")
        if len(v) > 32000:
            raise ValueError("Content exceeds maximum length of 32000 characters")
        return v

class Memory(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    created_at: Union[str, datetime]
    metadata: Dict[str, Any]
    window_id: Optional[str] = None
    semantic_vector: Optional[List[float]] = None
    trace_id: Optional[str] = None

    def to_response(self) -> 'MemoryResponse':
        """Convert Memory to MemoryResponse"""
        return MemoryResponse(
            id=self.id,
            content=self.content,
            memory_type=self.memory_type,
            created_at=self.created_at,
            metadata=self.metadata,
            window_id=self.window_id,
            semantic_vector=self.semantic_vector,
            trace_id=self.trace_id
        )

class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str
    message: str
    trace_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    operation_type: Optional[OperationType] = None
    details: Optional[Dict[str, Any]] = None

class MemoryResponse(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    created_at: str
    metadata: Dict[str, Any]
    window_id: Optional[str] = None
    semantic_vector: Optional[List[float]] = None
    trace_id: Optional[str] = None
    error: Optional[ErrorDetail] = None

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
            semantic_vector=memory.semantic_vector,
            trace_id=memory.trace_id
        )

class QueryRequest(BaseModel):
    query: str
    prompt: str = Field(..., description="The query prompt")
    top_k: int = Field(
        default=5,  # Default to 5
        ge=1,
        le=20,  # Hard limit of 20
        description="Number of memories to retrieve (max 20)"
    )
    window_id: Optional[str] = Field(None, description="Optional window ID to filter context")
    request_metadata: Optional[RequestMetadata] = None

class QueryResponse(BaseModel):
    """Response model for memory queries"""
    matches: List[MemoryResponse] = Field(..., description="List of matching memories")
    similarity_scores: List[float] = Field(..., description="Similarity scores for each match")
    response: Optional[str] = Field(None, description="Generated response from the LLM")
    trace_id: Optional[str] = None
    error: Optional[ErrorDetail] = None

class BatchOperationResponse(BaseModel):
    """Response model for batch operations"""
    successful_operations: int
    failed_operations: int
    errors: List[ErrorDetail] = Field(default_factory=list)
    trace_id: str
    operation_type: OperationType
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())