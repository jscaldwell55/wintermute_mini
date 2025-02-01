from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class MemoryType(str, Enum):
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"

class Memory(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    semantic_vector: List[float]
    metadata: Dict = Field(default_factory=dict)
    window_id: Optional[str] = None