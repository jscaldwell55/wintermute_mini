# api/core/memory/interfaces/memory_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class MemoryService(ABC):
    """
    Abstract Base Class (Interface) for Memory Services.

    This class defines the methods that any concrete memory service 
    implementation (e.g., Pinecone, Faiss, in-memory, etc.) must provide.
    """

    @abstractmethod
    async def create_memory(self, memory_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Creates a new memory.

        Args:
            memory_id: The unique identifier for the memory.
            vector: The vector representation of the memory content.
            metadata: A dictionary of metadata associated with the memory.

        Returns:
            True if the memory was created successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_memory_by_id(self, memory_id: str) -> Dict[str, Any]:
        """
        Retrieves a memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            A dictionary containing the memory data (vector and metadata), 
            or None if the memory is not found.
        """
        raise NotImplementedError

    @abstractmethod
    async def update_memory(self, memory_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Updates an existing memory.

        Args:
            memory_id: The ID of the memory to update.
            vector: The new vector representation.
            metadata: The new metadata.

        Returns:
            True if the memory was updated successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Deletes a memory by its ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if the memory was deleted successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def query_memories(self, query_vector: List[float], top_k: int = 10, filter: Dict = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries the memory service for memories similar to a given query vector.

        Args:
            query_vector: The vector to query against.
            top_k: The number of most similar memories to return.
            filter: An optional dictionary to filter results by metadata.

        Returns:
            A list of tuples, where each tuple contains:
                - A dictionary representing the memory data (including metadata).
                - The similarity score between the query vector and the memory vector.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Checks the health status of the memory service.
    
        Returns:
            Dict[str, Any]: A dictionary containing at minimum:
                - status: str ("healthy" or "unhealthy")
                - error: Optional[str] (error message if unhealthy)
        """
        raise NotImplementedError