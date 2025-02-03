# main.py
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from contextlib import asynccontextmanager
import asyncio
import time
from typing import AsyncGenerator
import uvicorn
from httpx import AsyncClient, HTTPStatusError, ReadTimeout

from api.core.memory.models import (
    CreateMemoryRequest,
    MemoryResponse,
    QueryRequest,
    QueryResponse
)
from api.core.memory.memory import MemorySystem, MemoryOperationError
from api.core.vector.vector_operations import VectorOperationsImpl  # Import VectorOperationsImpl
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.utils.config import get_settings
from api.core.consolidation.models import ConsolidationConfig
from api.core.consolidation.consolidator import MemoryConsolidator, run_consolidation, AdaptiveConsolidator

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        rate_limit: int = 100,
        window: int = 60
    ):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.window = window
        self.requests = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = asyncio.create_task(self._cleanup_old_requests())

    async def _cleanup_old_requests(self):
        """Periodically clean up old requests"""
        while True:
            try:
                await asyncio.sleep(self.window)
                async with self._lock:
                    current_time = time.time()
                    self.requests = {
                        ip: [t for t in times if current_time - t < self.window]
                        for ip, times in self.requests.items()
                    }
                    # Remove empty entries
                    self.requests = {ip: times for ip, times in self.requests.items() if times}
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")

    async def dispatch(self, request: Request, call_next):
        """Handle incoming request and apply rate limiting"""
        client_ip = request.client.host
        current_time = time.time()

        async with self._lock:
            if client_ip in self.requests:
                recent_requests = [
                    t for t in self.requests[client_ip]
                    if current_time - t < self.window
                ]
                
                if len(recent_requests) >= self.rate_limit:
                    logger.warning(f"Rate limit exceeded for client: {client_ip}")
                    return Response(
                        status_code=429,
                        content="Rate limit exceeded. Please try again later.",
                        media_type="text/plain"
                    )
                
                self.requests[client_ip] = recent_requests + [current_time]
            else:
                self.requests[client_ip] = [current_time]

        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise

    async def shutdown(self):
        """Cleanup method for shutdown"""
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

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
        """Initialize all system components with proper error handling"""
        if not self._initialized:
            try:
                logger.info("🔧 Initializing system components...")

                # Initialize vector operations
                self.vector_operations = VectorOperationsImpl()
                logger.info("✅ Vector operations initialized")

                # Initialize Pinecone service
                logger.info("🔍 Attempting to initialize Pinecone Service...")
                self.pinecone_service = PineconeService(
                    api_key=self.settings.pinecone_api_key,
                    environment=self.settings.pinecone_environment,
                    index_name=self.settings.pinecone_index_name
                )

                # Verify Pinecone initialization worked
                try:
                    _ = self.pinecone_service.index  # This will trigger initialization if needed
                    logger.info("✅ Pinecone service initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Pinecone service: {e}")
                    raise

                logger.info(f"✅ Pinecone index '{self.pinecone_service.index_name}' initialized successfully!")

                # Initialize LLM service
                self.llm_service = LLMService()
                logger.info("✅ LLM service initialized")

                # Initialize memory system
                self.memory_system = MemorySystem(
                    pinecone_service=self.pinecone_service,
                    vector_operations=self.vector_operations,
                    settings=self.settings,
                )
                logger.info("✅ Memory system initialized")

                # Initialize memory consolidator
                config = ConsolidationConfig()
                self.consolidator = AdaptiveConsolidator(
                    config=config,
                    pinecone_service=self.pinecone_service,
                    llm_service=self.llm_service
                )
                self.consolidation_task = asyncio.create_task(run_consolidation(self.consolidator))
                logger.info("✅ Memory consolidator initialized")

                self._initialized = True
                logger.info("🎉 All system components initialized successfully")

            except Exception as e:
                logger.error(f"🚨 Error initializing system components: {e}")
                await self.cleanup()  # Ensure cleanup is available
                raise

    async def cleanup(self):
        """Cleanup all system components"""
        try:
            logger.info("🔧 Running system cleanup...")

            # Cancel consolidation task
            if self.consolidation_task:
                self.consolidation_task.cancel()
                try:
                    await self.consolidation_task
                except asyncio.CancelledError:
                    pass
                logger.info("✅ Memory consolidation task stopped")

            # Close Pinecone service
            if self.pinecone_service:
                self.pinecone_service = None  # Properly clear Pinecone reference
                logger.info("✅ Pinecone service connections closed")

            self._initialized = False
            logger.info("🎉 System cleanup completed")

        except Exception as e:
            logger.error(f"🚨 Error during system cleanup: {e}")
            raise


# Create global components instance
components = SystemComponents()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    try:
        await components.initialize()
        yield
    finally:
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

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    rate_limit=100,
    window=60
)

@app.post("/query", response_model=QueryResponse)
async def query_memory(query: QueryRequest):
    """Process a query and return relevant memories with a generated response"""
    try:
        if not components._initialized:
            raise HTTPException(
                status_code=503,
                detail="System initializing, please try again in a moment"
            )
        
        query_response = await components.memory_system.query_memories(query)
        
        context = "\n".join([
            f"- {memory.content}" 
            for memory in query_response.matches
        ])
        
        prompt = f"""Context:
{context}

User Query:
{query.prompt}

Please provide a response based on the above context and query."""
        
        response = await components.llm_service.generate_response_async(prompt)
        
        await components.memory_system.add_interaction(
            user_input=query.prompt,
            response=response,
            window_id=query.window_id
        )
        
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
        if not components._initialized:
            raise HTTPException(
                status_code=503,
                detail="System initializing, please try again in a moment"
            )
            
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
        if not components._initialized:
            raise HTTPException(
                status_code=503,
                detail="System initializing, please try again in a moment"
            )
            
        memory = await components.memory_system.get_memory_by_id(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
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
        if not components._initialized:
            raise HTTPException(
                status_code=503,
                detail="System initializing, please try again in a moment"
            )
            
        success = await components.memory_system.delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    status = {
        "status": "healthy" if components._initialized else "initializing",
        "initialized": components._initialized,
        "environment": components.settings.environment,
        "services": {}
    }
    
    if components._initialized:
        try:
            pinecone_health = await components.pinecone_service.health_check()
            status["services"]["pinecone"] = pinecone_health
        except Exception as e:
            status["services"]["pinecone"] = {"status": "unhealthy", "error": str(e)}
    
    return status

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)