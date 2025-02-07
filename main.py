# 1. Imports
from fastapi import FastAPI, HTTPException, Depends, Request, Response, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timedelta, timezone
import asyncio
from datetime import datetime, timedelta, timezone
import sys
import os
from typing import AsyncGenerator, List, Optional, Dict, Any
import uvicorn
from pydantic import ValidationError
import uuid
import time
from starlette.routing import Mount

from api.core.memory.models import (
    CreateMemoryRequest,
    MemoryResponse,
    QueryRequest,
    QueryResponse,
    Memory,
    MemoryType,
    OperationType,
    RequestMetadata,
    ErrorDetail,
)
from api.core.memory.memory import MemorySystem, MemoryOperationError
from api.core.vector.vector_operations import VectorOperationsImpl
from api.utils.pinecone_service import PineconeService
from api.utils.llm_service import LLMService
from api.utils.config import get_settings, Settings
from api.core.consolidation.models import ConsolidationConfig
from api.core.consolidation.consolidator import AdaptiveConsolidator
from api.utils.prompt_templates import response_template
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.interfaces.vector_operations import VectorOperations

# 2. Class Definitions
logger = logging.getLogger(__name__)
logger.info("Main module loading")
logging.basicConfig(level=logging.INFO)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, rate_limit: int = 100, window: int = 60):
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
                    self.requests = {
                        ip: times for ip, times in self.requests.items() if times
                    }
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
                        media_type="text/plain",
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
        if hasattr(self, "_cleanup_task"):
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
        self._initialized = False
        self.settings = get_settings()  # Get settings *once* here.

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
                    index_name=self.settings.pinecone_index_name,
                )

                try:
                    _ = self.pinecone_service.index
                    logger.info("✅ Pinecone service initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Pinecone service: {e}")
                    raise

                logger.info(
                    f"✅ Pinecone index '{self.pinecone_service.index_name}' initialized successfully!"
                )

                self.llm_service = LLMService()
                logger.info("✅ LLM service initialized")

                self.memory_system = MemorySystem(
                    pinecone_service=self.pinecone_service,
                    vector_operations=self.vector_operations,
                    settings=self.settings,
                )
                logger.info("✅ Memory system initialized")

                self._initialized = True
                logger.info("🎉 All system components initialized successfully")

            except Exception as e:
                logger.error(f"🚨 Error initializing system components: {e}")
                await self.cleanup()
                raise

    async def cleanup(self):
        try:
            logger.info("🔧 Running system cleanup...")

            if self.pinecone_service:
                await self.pinecone_service.close_connections()
                self.pinecone_service = None
                logger.info("✅ Pinecone service connections closed")

            self._initialized = False
            logger.info("🎉 System cleanup completed")

        except Exception as e:
            logger.error(f"🚨 Error during system cleanup: {e}")
            raise


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"Request {request.method} {request.url.path} completed in {duration:.2f}s")
        return response

# 3. Global Instance & Dependencies
components = SystemComponents()

async def get_memory_system() -> MemorySystem:
    if not components._initialized:
        raise HTTPException(
            status_code=503,
            detail="System initializing, please try again in a moment",
        )
    return components.memory_system

async def get_pinecone_service() -> MemoryService:
    if not components._initialized:
        raise HTTPException(
            status_code=503,
            detail="System initializing, please try again in a moment",
        )
    return components.pinecone_service

async def get_vector_operations() -> VectorOperations:
    if not components._initialized:
        raise HTTPException(
            status_code=503,
            detail="System initializing, please try again in a moment",
        )
    return components.vector_operations

# 4. Static File Setup Function Definition (but don't call it yet)
def setup_static_files(app: FastAPI):
    """Configure static files serving with fallback for SPA routing"""
    try:
        # First try to serve from the production build directory
        static_dir = os.path.join(os.getcwd(), "frontend", "dist")
        if os.path.exists(static_dir):
            logger.info(f"Mounting static files from: {static_dir}")
            # Mount specific directories first
            assets_dir = os.path.join(static_dir, "assets")
            if os.path.exists(assets_dir):
                app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
                logger.info(f"Mounted assets directory: {assets_dir}")

            # Add catch-all route for SPA routing
            @app.get("/{full_path:path}")
            async def serve_spa(full_path: str, request: Request):
                logger.info(f"Received request for path: {full_path}")

                # Always pass through API endpoints.  Use a tuple for startswith.
                if full_path.startswith(("memories/", "health", "ping", "consolidate", "query", "api/")):
                    logger.info(f"Passing through API request: {full_path}")
                    raise HTTPException(status_code=404, detail="Not found")

                logger.info(f"Serving static file for path: {full_path}")
                static_file = os.path.join(static_dir, "index.html")
                if os.path.exists(static_file):
                    return FileResponse(static_file)
                raise HTTPException(status_code=404, detail="Not found")


            @app.get("/")
            async def serve_root():
                static_file = os.path.join(static_dir, "index.html")
                if os.path.exists(static_file):
                    logger.info(f"Serving root: {static_file}")
                    return FileResponse(static_file)
                raise HTTPException(status_code=404, detail="Not found")
        else:
            logger.warning("No static files directory found")

    except Exception as e:
        logger.error(f"Error setting up static files: {e}")
        raise

# 5. Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await components.initialize()
        yield
    finally:
        await components.cleanup()

# 6. Create FastAPI App
app = FastAPI(
    title="Project Wintermute Memory System",
    description="An AI assistant with semantic memory capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

# --- API Router Setup ---
api_router = APIRouter(prefix="/api/v1")

# 7. Add Middleware (ONCE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, rate_limit=100, window=60)
app.add_middleware(LoggingMiddleware)


# 8. Define ALL API Routes using api_router
@api_router.post("/consolidate")
async def consolidate_now(settings: Settings = Depends(lambda: components.settings)):
    try:
        config = ConsolidationConfig(
            consolidation_interval_hours=24,
            max_age_days=settings.memory_max_age_days,
            min_cluster_size=settings.min_cluster_size,
            eps=settings.eps if hasattr(settings, "eps") else 0.5,
        )
        consolidator = AdaptiveConsolidator(
            config, components.pinecone_service, components.llm_service
        )
        await consolidator.consolidate_memories()
        logger.info("Manual consolidation completed successfully")
        return {"message": "Consolidation triggered"}

    except Exception as e:
        logger.error(f"Manual consolidation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Consolidation failed: {str(e)}"
        )

@api_router.get("/ping")
async def ping():
    return {"message": "pong"}

@api_router.get("/health")
async def health_check():
    services_status = {}
    is_initialized = False
    error_message = None
    try:
        is_initialized = getattr(components, "_initialized", False)
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
                    "status": "healthy"
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

@api_router.get("/memories")
async def list_memories(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    window_id: Optional[str] = None,
    memory_type: Optional[MemoryType] = None,
    memory_system: MemorySystem = Depends(get_memory_system),
):
    try:
        filter_dict = {}
        if window_id:
            filter_dict["window_id"] = window_id
        if memory_type:
            filter_dict["memory_type"] = memory_type.value
        query_text = "Retrieve all memories"
        query_vector = await memory_system.vector_operations.create_semantic_vector(
            query_text
        )
        results = await memory_system.pinecone_service.query_memories(
            query_vector=query_vector,
            top_k=limit,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
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
                    semantic_vector=memory_data.get("vector"),
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

@api_router.post("/memories", response_model=MemoryResponse)
async def add_memory(
    request: CreateMemoryRequest, memory_system: MemorySystem = Depends(get_memory_system)
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



@api_router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, memory_system: MemorySystem = Depends(get_memory_system)):
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
            semantic_vector=memory.semantic_vector,
        )
    except MemoryOperationError as e:
        logger.error(f"Memory operation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str, memory_system: MemorySystem = Depends(get_memory_system)
):
    try:
        success = await memory_system.delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@api_router.options("/query")
async def query_options():
    return {"message": "Options request successful"}

@api_router.post("/test")
async def test_post(request: Request):
    """Test endpoint to verify POST requests are working"""
    try:
        body = await request.json()
        return {
            "message": "POST request successful",
            "received_data": body,
            "method": request.method,
            "headers": dict(request.headers),
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return {"error": str(e)}

@api_router.get("/routes")
async def list_routes():
    """List all registered routes"""
    try:
        routes = []
        for route in app.routes:
            route_info = {"path": route.path, "name": route.name}
            # Check if the route has methods (mounted routes don't)
            if hasattr(route, 'methods'):
                route_info["methods"] = list(route.methods)
            else:
                route_info["type"] = "mount" if isinstance(route, Mount) else "other"
            routes.append(route_info)

        logger.info(f"Successfully retrieved {len(routes)} routes")

        # Remove duplicates based on 'path' and 'type'
        unique_routes = []
        seen_routes = set()
        for route in routes:
            route_key = (route['path'], route.get('type'))  # Use get for 'type' which might be missing
            if route_key not in seen_routes:
                unique_routes.append(route)
                seen_routes.add(route_key)


        return {"routes": unique_routes}
    except Exception as e:
        logger.error(f"Error listing routes: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing routes: {str(e)}"
        )

@api_router.post("/debug-query")
async def debug_query(request: Request):
    """Debug endpoint to echo back request data"""
    body = await request.json()
    return {
        "received": {
            "method": request.method,
            "headers": dict(request.headers),
            "body": body,
        },
    }



@api_router.post("/query", response_model=QueryResponse)
async def query_memory(
    request: Request,
    query: QueryRequest,
    memory_system: MemorySystem = Depends(get_memory_system),
    llm_service: LLMService = Depends(lambda: components.llm_service)
):
    trace_id = f"query_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    query.request_metadata = RequestMetadata(
        operation_type=OperationType.QUERY,
        window_id=query.window_id,
        trace_id=trace_id
    )
    try:
        logger.info(f"[{trace_id}] Received query request: {query.prompt[:100]}...")
        user_query_embedding = await memory_system.vector_operations.create_semantic_vector(
            query.prompt
        )

        # --- Semantic Query ---
        semantic_results = await memory_system.pinecone_service.query_memories(
            query_vector=user_query_embedding,
            top_k=3,
            filter={"memory_type": "SEMANTIC"},
            include_metadata=True,
            include_values=False  # We don't need the vectors themselves
        )
        semantic_memories = [
            match[0]["metadata"]["content"] for match in semantic_results
        ]
        # We don't need to store the scores separately if we aren't returning them

        # --- Episodic Query ---
        cutoff_time = datetime.utcnow() - timedelta(hours=3)
        cutoff_time_str = cutoff_time.isoformat()
        episodic_results = await memory_system.pinecone_service.query_memories(
            query_vector=user_query_embedding,
            top_k=5,
            filter={
                "memory_type": "EPISODIC",
                "created_at": {"$gte": cutoff_time_str},
            },
            include_metadata=True,
            include_values=False  # We don't need the vectors
        )
        episodic_memories = []
        for match in episodic_results:
            memory_data, _ = match  # We only care about memory_data, not the score
            created_at = datetime.fromisoformat(
                memory_data["metadata"]["created_at"].replace("Z", "+00:00")
            )
            time_ago = (
                datetime.utcnow().replace(tzinfo=timezone.utc) - created_at
            ).total_seconds()
            if time_ago < 60:
                time_str = f"{int(time_ago)} seconds ago"
            elif time_ago < 3600:
                time_str = f"{int(time_ago / 60)} minutes ago"
            else:
                time_str = f"{int(time_ago / 3600)} hours ago"
            episodic_memories.append(f"[{time_str}] {memory_data['metadata']['content']}")

        prompt = response_template.format(
            query=query.prompt,
            semantic_memories=chr(10).join(semantic_memories),
            episodic_memories=chr(10).join(episodic_memories),
        )
        response = await llm_service.generate_response_async(prompt)
        logger.info(f"[{trace_id}] Generated response successfully")
        try:
            await memory_system.store_interaction(
                query=query.prompt,
                response=response,
                window_id=query.window_id,
            )
            logger.info(f"[{trace_id}] Interaction stored successfully")
        except Exception as e:
            logger.error(
                f"[{trace_id}] Failed to store interaction: {str(e)}", exc_info=True
            )
        return QueryResponse( # Always return a QueryResponse
            response=response, matches=[], trace_id=trace_id, similarity_scores=[], error=None, #Set error to None
            metadata={"success": True}
        )
    except ValidationError as e:
        logger.error(f"[{trace_id}] Validation error: {str(e)}", exc_info=True)
        return QueryResponse( # Return QueryResponse for errors
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="422", message="Validation Error", details={"error" : str(e)}, trace_id=trace_id),
            metadata={"success": False}
        )
    except MemoryOperationError as e:
        logger.error(f"[{trace_id}] Memory operation error: {str(e)}", exc_info=True)
        return QueryResponse( # Return QueryResponse for errors
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="400", message="Memory Operation Error", details={"error" : str(e)}, trace_id=trace_id),
            metadata={"success": False}
        )
    except Exception as e:
        logger.error(f"[{trace_id}] Unexpected error: {str(e)}", exc_info=True)
        return QueryResponse( # Return QueryResponse for errors
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="500", message="Internal Server Error", details={"error" : str(e)}, trace_id=trace_id),
            metadata={"success": False}
        )


# --- Include Router and Setup Static Files ---
app.include_router(api_router)
# 9. Setup Static Files (LAST, ONCE)
setup_static_files(app)

# 10. Main Block
if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        reload = os.getenv("RELOAD", "true").lower() == "true"
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config["formatters"]["default"][
            "fmt"
        ] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        uvicorn.run("main:app", host=host, port=port, reload=reload, log_config=log_config)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)