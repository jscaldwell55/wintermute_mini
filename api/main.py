import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
from contextlib import asynccontextmanager

from api.core.memory.models import MemoryType, Memory
from api.core.memory.memory import MemorySystem
from api.core.vector.vector_operations import VectorOperations
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.utils.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# --- Pydantic Models ---
class Query(BaseModel):
    prompt: str
    top_k: int = 5

class AddMemoryRequest(BaseModel):
    content: str
    metadata: Optional[Dict] = {}
    memory_type: MemoryType = MemoryType.EPISODIC

class SystemComponents:
    def __init__(self):
        self.memory_system = None
        self.vector_operations = None
        self.pinecone_service = None
        self.llm_service = None
        self._initialized = False
        self.settings = get_settings()
    
    async def initialize(self):
        if not self._initialized:
            # Initialize vector operations
            self.vector_operations = VectorOperations()
            
            # Initialize Pinecone service
            self.pinecone_service = PineconeService(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment,
                index_name=self.settings.index_name
            )
            await self.pinecone_service.initialize_index()
            
            # Initialize LLM service
            self.llm_service = LLMService()
            
            # Initialize memory system
            self.memory_system = MemorySystem(
                pinecone_service=self.pinecone_service,
                vector_operations=self.vector_operations
            )
            
            self._initialized = True
            logger.info("System components initialized successfully")
    
    async def cleanup(self):
        if self.pinecone_service:
            await self.pinecone_service.close_connections()
        logger.info("System cleanup completed")

# Create global components instance
components = SystemComponents()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    await components.initialize()
    yield
    await components.cleanup()

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_memory(query: Query):
    """Process a query and return a response"""
    try:
        # Create query vector
        query_vector = await components.vector_operations.create_semantic_vector(query.prompt)
        
        # Query memories
        memories = await components.memory_system.query_memory(
            query_vector=query_vector,
            top_k=query.top_k
        )
        
        # Format memories for context
        context = "\n".join([
            f"- {memory.content}" 
            for memory, _ in memories
        ])
        
        # Format prompt
        prompt = f"""Context:
{context}

User Query:
{query.prompt}

Please provide a response based on the above context and query."""
        
        # Generate response using LLM
        response = await components.llm_service.generate_gpt_response_async(prompt)
        
        # Store interaction
        await components.memory_system.add_interaction(query.prompt, response)
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories")
async def add_memory(request: AddMemoryRequest):
    """Add a new memory"""
    try:
        memory_id = await components.memory_system.add_memory(
            content=request.content,
            memory_type=request.memory_type,
            metadata=request.metadata
        )
        return {"message": "Memory added successfully", "id": memory_id}
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": components.settings.environment
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)