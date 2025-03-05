# api/system_components.py
import logging
from api.core.vector.vector_operations import VectorOperationsImpl
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.core.memory.memory import MemorySystem
from api.utils.config import get_settings

logger = logging.getLogger(__name__)

logger.info("Initializing system components")
try:
    # Component initialization code
    logger.info("System components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize system components: {str(e)}")
    raise

class SystemComponents:
    def __init__(self):
        self.memory_system = None
        self.vector_operations = None
        self.pinecone_service = None
        self.llm_service = None
        self.consolidator = None
        self._initialized = False
        self.settings = get_settings()

    # Add the rest of your SystemComponents class methods here

# Create a single instance to be used throughout the app
components = SystemComponents()