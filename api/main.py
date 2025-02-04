# main.py
from fastapi import FastAPI, HTTPException, Depends, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
import asyncio
import time
import sys
import os
from typing import AsyncGenerator, List, Optional, Dict, Any
import uvicorn

from api.core.memory.models import (
    CreateMemoryRequest,
    MemoryResponse,
    QueryRequest,
    QueryResponse,
    Memory,
    MemoryType
)
from api.core.memory.memory import MemorySystem, MemoryOperationError
from api.core.vector.vector_operations import VectorOperationsImpl
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.utils.config import get_settings
from api.core.consolidation.models import ConsolidationConfig
from api.core.consolidation.consolidator import run_consolidation, AdaptiveConsolidator
from api.utils.prompt_templates import response_template
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.interfaces.vector_operations import VectorOperations

logger = logging.getLogger(__name__)
settings = get_settings()

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
        while True:
            try:
                await asyncio.sleep(self.window)
                async with self._lock:
                    current_time = time.time()
                    self.requests = {
                        ip: [t for t in times if current_time - t < self.window]
                        for ip, times in self.requests.items()
                    }
                    self.requests = {ip: times for ip, times in self.requests.items() if times}
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")

    async def dispatch(self, request: Request, call_next):
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
        if not self._initialized:
            try:
                logger.info("🔧 Initializing system components...")
                self.vector_operations = VectorOperationsImpl()
                logger.info("✅ Vector operations initialized")

                logger.info("🔍 Attempting to initialize Pinecone Service...")
                self.pinecone_service = PineconeService(
                    api_key=self.settings.pinecone_api_key,
                    environment=self.settings.pinecone_environment,
                    index_name=self.settings.pinecone_index_name
                )

                try:
                    _ = self.pinecone_service.index
                    logger.info("✅ Pinecone service initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Pinecone service: {e}")
                    raise

                logger.info(f"✅ Pinecone index '{self.pinecone_service.index_name}' initialized successfully!")

                self.llm_service = LLMService()
                logger.info("✅ LLM service initialized")

                self.memory_system = MemorySystem(
                    pinecone_service=self.pinecone_service,
                    vector_operations=self.vector_operations,
                    settings=self.settings,
                )
                logger.info("✅ Memory system initialized")

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
                await self.cleanup()
                raise

    async def cleanup(self):
        try:
            logger.info("🔧 Running system cleanup...")

            if self.consolidation_task:
                self.consolidation_task.cancel()
                try:
                    await self.consolidation_task
                except asyncio.CancelledError:
                    pass
                logger.info("✅ Memory consolidation task stopped")

            if self.pinecone_service:
                self.pinecone_service = None
                logger.info("✅ Pinecone service connections closed")

            self._initialized = False
            logger.info("🎉 System cleanup completed")

        except Exception as e:
            logger.error(f"🚨 Error during system cleanup: {e}")
            raise

# Create global components instance
components = SystemComponents()

# Define dependencies
async def get_memory_system() -> MemorySystem:
    if not components._initialized:
        raise HTTPException(
            status_code=503,
            detail="System initializing, please try again in a moment"
        )
    return components.memory_system

async def get_pinecone_service() -> MemoryService:
    if not components._initialized:
        raise HTTPException(
            status_code=503,
            detail="System initializing, please try again in a moment"
        )
    return components.pinecone_service

async def get_vector_operations() -> VectorOperations:
    if not components._initialized:
        raise HTTPException(
            status_code=503,
            detail="System initializing, please try again in a moment"
        )
    return components.vector_operations

def setup_static_files(app: FastAPI):
    """Configure static files serving with fallback for SPA routing"""
    try:
        # First try to serve from the production build directory
        static_dir = os.path.join(os.getcwd(), "frontend", "dist")
        if os.path.exists(static_dir):
            app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
            logger.info(f"Mounted static files from: {static_dir}")
        else:
            # Fallback to development directory
            static_dir = os.path.join(os.getcwd(), "frontend", "public")
            if os.path.exists(static_dir):
                app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
                logger.info(f"Mounted static files from fallback directory: {static_dir}")
            else:
                logger.warning("No static files directory found")
                
        # Add catch-all route for SPA routing
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            static_file = os.path.join(static_dir, "index.html")
            if os.path.exists(static_file):
                return FileResponse(static_file)
            raise HTTPException(status_code=404, detail="Not found")
                
    except Exception as e:
        logger.error(f"Error setting up static files: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await components.initialize()
        yield
    finally:
        await components.cleanup()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Project Wintermute Memory System",
        description="An AI assistant with semantic memory capabilities",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://wintermute-staging-x-49dd432d3500.herokuapp.com",
            "http://localhost:3000",
            "http://localhost:5173"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        RateLimitMiddleware,
        rate_limit=100,
        window=60
    )

    # Set up static files
    setup_static_files(app)

    # Register routes
    @app.post("/query", response_model=QueryResponse)
    async def query_memory(
        query: QueryRequest,
        memory_system: MemorySystem = Depends(get_memory_system),
        llm_service: LLMService = Depends(lambda: components.llm_service)
    ):
        try:
            query_response = await memory_system.query_memories(query)
            context = "\n".join([f"- {memory.content}" for memory in query_response.matches])
            prompt = response_template.format(context=context, query=query.prompt)
            response = await llm_service.generate_response_async(prompt)
            
            await memory_system.add_interaction(
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
    async def add_memory(
        request: CreateMemoryRequest,
        memory_system: MemorySystem = Depends(get_memory_system)
    ):
        try:
            memory_result = await memory_system.create_memory_from_request(request)
            if isinstance(memory_result, Memory):
                return memory_result.to_response()
            return memory_result
        except MemoryOperationError as e:
            logger.error(f"Memory operation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/memories", response_model=List[MemoryResponse])
    async def list_memories(
        limit: int = Query(default=10, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        window_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        memory_system: MemorySystem = Depends(get_memory_system)
    ):
        try:
            filter_dict = {}
            if window_id:
                filter_dict["window_id"] = window_id
            if memory_type:
                filter_dict["memory_type"] = memory_type.value

            query_vector = await memory_system.vector_operations.create_semantic_vector("")
            results = await memory_system.pinecone_service.query_memories(
                query_vector=query_vector,
                top_k=limit,
                filter=filter_dict if filter_dict else None
            )

            memories = []
            for memory_data, _ in results:
                try:
                    memory = Memory(
                        id=memory_data["id"],
                        content=memory_data["metadata"]["content"],
                        memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                        created_at=memory_data["metadata"]["created_at"],
                        metadata=memory_data["metadata"],
                        window_id=memory_data["metadata"].get("window_id"),
                        semantic_vector=memory_data.get("vector")
                    )
                    memories.append(memory.to_response())
                except Exception as e:
                    logger.error(f"Error converting memory {memory_data['id']}: {e}")
                    continue

            memories = memories[offset:]
            return memories

        except MemoryOperationError as e:
            logger.error(f"Memory operation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/memories/{memory_id}", response_model=MemoryResponse)
    async def get_memory(
        memory_id: str,
        memory_system: MemorySystem = Depends(get_memory_system)
    ):
        try:
            memory = await memory_system.get_memory_by_id(memory_id)
            if not memory:
                raise HTTPException(status_code=404, detail="Memory not found")

            if isinstance(memory, Memory):
                return memory.to_response()
            if isinstance(memory, MemoryResponse):
                return memory
                
            return MemoryResponse(
                id=memory.id,
                content=memory.content,
                memory_type=memory.memory_type,
                created_at=memory.created_at,
                metadata=memory.metadata,
                window_id=memory.window_id,
                semantic_vector=memory.semantic_vector
            )
        except MemoryOperationError as e:
            logger.error(f"Memory operation error: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/memories/{memory_id}")
    async def delete_memory(
        memory_id: str,
        memory_system: MemorySystem = Depends(get_memory_system)
    ):
        try:
            success = await memory_system.delete_memory(memory_id)
            if not success:
                raise HTTPException(status_code=404, detail="Memory not found")
            return {"message": "Memory deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        services_status = {}
        is_initialized = False
        error_message = None

        try:
            is_initialized = getattr(components, '_initialized', False)
            
            if is_initialized:
                try:
                    pinecone_health = await components.pinecone_service.health_check()
                    services_status["pinecone"] = pinecone_health
                except Exception as e:
                    services_status["pinecone"] = {"status": "unhealthy", "error": str(e)}
                    is_initialized = False
                    
                services_status.update({
                    "vector_operations": {
                        "status": "healthy",
                        "model": components.settings.embedding_model
                    },
                    "memory_service": {
                        "status": "healthy",
                        "cache_enabled": components.settings.enable_memory_cache
                    }
                })
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            error_message = str(e)
            is_initialized = False
            
        status = {
            "status": "ready" if is_initialized else "initializing",
            "initialization_complete": is_initialized,
            "environment": components.settings.environment,
            "services": services_status
        }

        if error_message:
            status.update({
                "status": "error",
                "error": error_message
            })

        return status

    return app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        reload = os.getenv("RELOAD", "true").lower() == "true"
        
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_config=log_config
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)