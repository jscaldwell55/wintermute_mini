from typing import Optional, Any, Dict
from pydantic import BaseModel
from datetime import datetime

class MemoryOperationError(Exception):
    """Exception raised for errors in memory operations."""
    def __init__(self, operation: str, details: str, retry_count: int = 0):
        self.operation = operation
        self.details = details
        self.retry_count = retry_count
        super().__init__(f"Memory operation failed: {operation} - {details}")

class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = str(datetime.utcnow())

def create_response(
    success: bool,
    message: str,
    data: Optional[Any] = None,
    error: Optional[str] = None
) -> Dict:
    """Create a standardized API response."""
    return APIResponse(
        success=success,
        message=message,
        data=data,
        error=error
    ).model_dump()

def success_response(message: str, data: Optional[Any] = None) -> Dict:
    """Create a success response."""
    return create_response(True, message, data)

def error_response(message: str, error: Optional[str] = None) -> Dict:
    """Create an error response."""
    return create_response(False, message, error=error)

def memory_error_response(error: MemoryOperationError) -> Dict:
    """Create an error response specifically for memory operation failures."""
    return create_response(
        success=False,
        message=f"Memory operation failed: {error.operation}",
        error=error.details
    )