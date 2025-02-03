class MemoryOperationError(Exception):
    """Base exception for memory operations."""
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}: {self.cause}"
        return super().__str__()

class PineconeError(MemoryOperationError):
    """Raised for errors related to Pinecone interactions."""
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(f"Pinecone error: {message}", cause)