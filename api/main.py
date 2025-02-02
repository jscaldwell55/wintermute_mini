from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
import asyncio

from api.core.memory.models import (
    CreateMemoryRequest,
    MemoryResponse,
    QueryRequest,
    QueryResponse
)
from api.core.memory.memory import MemorySystem, MemoryOperationError
from api.core.vector.vector_operations import VectorOperations
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.utils.config import get_settings
from api.core.consolidation.models import ConsolidationConfig
from api.core.consolidation.consolidator import MemoryConsolidator, run_consolidation

logger = logging.getLogger(__name__)

class SystemComponents:
    def __init__(self):
        self.memory_system = None
        self.vector_operations = None
        self.pinecone_service = None
        self.llm_service = None
        self.consolidator = None
        self.consolidation_task = None
        self._initialized = False
        self.settings = get_settings()
    
    async def initialize(self):
        if not self._initialized:
            try:
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
                
                # Initialize consolidator
                config = ConsolidationConfig()
                self.consolidator = MemoryConsolidator(
                    config=config,
                    pinecone_service=self.pinecone_service,
                    llm_service=self.llm_service
                )
                self.consolidation_task = asyncio.create_task(run_consolidation(self.consolidator))
                
                self._initialized = True
                logger.info("System components initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing system components: {e}")
                raise
    
    async def cleanup(self):
        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass
            logger.info("Memory consolidation task stopped")
            
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
app = FastAPI(
    title="Project Wintermute Memory System",
    description="An AI assistant with semantic memory capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", response_model=QueryResponse)
async def query_memory(query: QueryRequest):
    """Process a query and return relevant memories with a generated response"""
    try:
        # Query memories using the new query handler
        query_response = await components.memory_system.query_memories(query)
        
        # Format memories for context
        context = "\n".join([
            f"- {memory.content}" 
            for memory in query_response.matches
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
        await components.memory_system.add_interaction(
            user_input=query.prompt,
            response=response,
            window_id=query.window_id
        )
        
        # Add response to query response
        query_response.response = response
        return query_response
        
    except MemoryOperationError as e:
        logger.error(f"Memory operation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories", response_model=MemoryResponse)
async def add_memory(request: CreateMemoryRequest):
    """Add a new memory"""
    try:
        return await components.memory_system.create_memory_from_request(request)
    except MemoryOperationError as e:
        logger.error(f"Memory operation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    """Retrieve a specific memory by ID"""
    try:
        memory = await components.memory_system.get_memory_by_id(memory_id)
        return MemoryResponse.model_validate(memory)
    except MemoryOperationError as e:
        logger.error(f"Memory operation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory"""
    try:
        success = await components.memory_system.delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": components._initialized,
        "environment": components.settings.environment
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)