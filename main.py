# main.py 

from fastapi import FastAPI, HTTPException, Depends, Request, Response, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timezone
import asyncio
import sys
import os
from typing import AsyncGenerator, List, Optional, Dict, Any
import uvicorn
from pydantic import ValidationError
import uuid
import time
from starlette.routing import Mount
from functools import lru_cache
import random

# Corrected import: Use the instance, case_response_template
from api.utils.prompt_templates import case_response_template  # USE THE CASE TEMPLATE
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
from api.core.consolidation.config import ConsolidationConfig
from api.core.consolidation.consolidator import MemoryConsolidator, get_consolidation_config
# from api.utils.prompt_templates import response_template  <- REMOVE THIS
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.interfaces.vector_operations import VectorOperations

# Keep only ONE router definition here:
api_router = APIRouter(prefix="/api/v1")

logger = logging.getLogger(__name__)
logger.info("Main module loading")
logging.basicConfig(level=logging.INFO)

# ... (RateLimitMiddleware, SystemComponents - No changes needed) ...
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
                    _ = self.pinecone_service.index  # Force initialization
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
        static_dir = os.path.join(os.getcwd(), "dist")
        if os.path.exists(static_dir):
            logger.info(f"Mounting static files from: {static_dir}")
            # Mount specific directories first
            app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

            # No need for a separate /assets mount, Vite handles that
            # No need for catch-all routes, StaticFiles(..., html=True) handles SPA routing

        else:
            logger.warning("No static files directory found at: %s", static_dir) # More specific warning

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
async def consolidate_now(config: ConsolidationConfig = Depends(get_consolidation_config)):
    try:
        consolidator = MemoryConsolidator(
            config=config,
            pinecone_service=components.pinecone_service,
            llm_service=components.llm_service
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
            filter_dict["memory_type"] = memory_type.value.upper()  # Use uppercase for consistency
        query_text = "Retrieve all memories"  # Placeholder for vector creation
        query_vector = await memory_system.vector_operations.create_semantic_vector(
            query_text
        )
        logger.info(f"Querying Pinecone with filter: {filter_dict}")  # Log the filter

        results = await memory_system.pinecone_service.query_memories(
            query_vector=query_vector,
            top_k=limit,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        logger.info(f"Pinecone query returned {len(results)} results.")  # Log results count

        memories = []
        for memory_data, _ in results:
            logger.info(f"memory data from result: {memory_data}")
            try:
                created_at = memory_data["metadata"].get("created_at") # Already a datetime object

                # No more handling different types
                if created_at is None:
                    created_at = datetime.now(timezone.utc)
                    logger.warning(f"Memory {memory_data['id']} is missing created_at, using current time.")

                # Convert to ISO string with ZULU time *here*, consistently.
                created_at_str = created_at.isoformat() + "Z"

                memory = Memory(
                    id=memory_data["id"],
                    content=memory_data["metadata"]["content"],
                    memory_type=MemoryType(memory_data["metadata"]["memory_type"]),
                    created_at= created_at,  # Use datetime object
                    metadata=memory_data["metadata"],
                    window_id=memory_data["metadata"].get("window_id"),
                    semantic_vector=memory_data.get("vector"),
                )
                memories.append(memory.to_response())
            except Exception as e:
                logger.error(f"Error converting memory {memory_data['id']}: {e}")
                continue
        memories = memories[offset:]  # Apply offset *after* processing all
        logger.info(f"Returning {len(memories)} memories.")  # Log returned count
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
        # Add created_at to the metadata *before* creating the memory
        if request.metadata is None:
            request.metadata = {}
        request.metadata["created_at"] = datetime.now(timezone.utc).isoformat() + "Z"

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
# Removed duplicate import of batty_response_template

@api_router.post("/query", response_model=QueryResponse)  # Keep only ONE /query route
async def query_memory(
    request: Request,
    query: QueryRequest,
    memory_system: MemorySystem = Depends(get_memory_system),
    llm_service: LLMService = Depends(lambda: components.llm_service)
) -> QueryResponse:

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
            top_k=memory_system.settings.semantic_top_k,  # Use setting
            filter={"memory_type": "SEMANTIC"},
            include_metadata=True,
        )
        # Semantic Memory Filtering:
        semantic_memories = []
        for match, _ in semantic_results: # Don't need score
            content = match["metadata"]["content"]
            if len(content.split()) >= 5:  # Keep only memories with 5+ words
                semantic_memories.append(content)
        logger.info(f"[{trace_id}] Semantic memories retrieved: {semantic_memories}")


        # --- Episodic Query ---
        episodic_results = await memory_system.pinecone_service.query_memories(
            query_vector=user_query_embedding,
            top_k=memory_system.settings.episodic_top_k, # Use setting
            filter={"memory_type": "EPISODIC"},  # ONLY filter by type
            include_metadata=True,
        )
        logger.info(f"[{trace_id}] Episodic memories retrieved: {len(episodic_results)}")

        episodic_memories = []
        seen_contents = set() # For basic deduplication
        for match in episodic_results:
            memory_data, score = match  # Unpack score here
            created_at = memory_data["metadata"].get("created_at") # Now a datetime object!
            logger.info(f"[{trace_id}] created_at: {created_at}, type: {type(created_at)}")

            try:
                # No more string parsing needed! We get a datetime object directly.
                time_ago = (datetime.now(timezone.utc) - created_at).total_seconds()
                if time_ago < 60:
                    time_str = f"{int(time_ago)} seconds ago"
                elif time_ago < 3600:
                    time_str = f"{int(time_ago / 60)} minutes ago"
                else:
                    time_str = f"{int(time_ago / 3600)} hours ago"

                # Keep only the combined interaction text, prepended with time.
                content = memory_data['metadata']['content']
                 # Simple deduplication (exact match)
                content_hash = hash(content) # More efficient than string comparisons
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    episodic_memories.append(f"{time_str}: {content[:200]}")  # Limit to 200 chars each

            except Exception as e:
                logger.error(f"[{trace_id}] Error processing episodic memory {memory_data['id']}: {e}")
                continue  # Continue to the next memory


        logger.info(f"[{trace_id}] Processed episodic memories: {episodic_memories}")

        # --- Construct the Prompt ---
        prompt = case_response_template.format(
            query=query.prompt,
            semantic_memories=semantic_memories,  # Pass the limited list
            episodic_memories=episodic_memories,  # Pass the limited list
        )
        logger.info(f"[{trace_id}] Sending prompt to LLM: {prompt[:200]}...")

                # --- Generate Response ---
        # ADD RANDOM TEMPERATURE HERE
        temperature = round(random.uniform(0.6, 0.9), 2)  # Random temp between 0.6 and 0.9
        logger.info(f"[{trace_id}] Using temperature: {temperature}") #Log the temperature
        await asyncio.sleep(1) # Add the 1 second delay.
        response = await llm_service.generate_response_async(
            prompt,
            max_tokens=500,  # Increased max_tokens for response
            temperature=temperature  # Pass the random temperature
        )
        logger.info(f"[{trace_id}] Generated response successfully")

        # --- Store Interaction (Episodic Memory) ---
        try:
            await memory_system.store_interaction_enhanced(  # Await the interaction storage, use ENHANCED
                query=query.prompt,
                response=response,
                window_id=query.window_id,
            )
            logger.info(f"[{trace_id}] Interaction stored successfully")
        except Exception as e:
            logger.error(
                f"[{trace_id}] Failed to store interaction: {str(e)}", exc_info=True
            )
        # Always return a QueryResponse, even in error cases
        return QueryResponse(
            response=response, matches=[], trace_id=trace_id, similarity_scores=[], error=None,
            metadata={"success": True}
        )

    except ValidationError as e:
        logger.error(f"[{trace_id}] Validation error: {str(e)}", exc_info=True)
        return QueryResponse(  # Return QueryResponse for errors
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="422", message="Validation Error", details={"error" : str(e)}, trace_id=trace_id),
            metadata={"success": False}
        )
    except MemoryOperationError as e:
        logger.error(f"[{trace_id}] Memory operation error: {str(e)}", exc_info=True)
        return QueryResponse(  # Return QueryResponse for errors
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="400", message="Memory Operation Error", details={"error" : str(e)}, trace_id=trace_id),
            metadata={"success": False}
        )
    except Exception as e:  # Correctly handle other exceptions here
        logger.error(f"[{trace_id}] Unexpected error: {str(e)}", exc_info=True)
        return QueryResponse(
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="500", message="Internal Server Error", details={"error": str(e)}, trace_id=trace_id), # Pass a dict
            metadata={"success": False}
        )
# --- Include Router and Setup Static Files ---
app.include_router(api_router)
setup_static_files(app)

# Shutdown handler for rate limiter
@app.on_event("shutdown")
async def shutdown_event():
    await components.cleanup() # Ensure all components are cleaned up.
    for middleware in app.middleware_stack:
        if isinstance(middleware, RateLimitMiddleware):
            await middleware.shutdown()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Get port from environment variable, default to 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")