from typing import Optional, Any, Dict
from pydantic import BaseModel
from datetime import datetime

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