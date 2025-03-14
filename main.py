# main.py 

from fastapi import FastAPI, HTTPException, Depends, Request, Response, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager
import logging
import math
from datetime import datetime, timezone
import asyncio
import sys
import os
from typing import AsyncGenerator, List, Optional, Dict, Any
import uvicorn
from pydantic import ValidationError
import uuid
import time
from datetime import datetime, timezone, timedelta
import hashlib
import json
import networkx as nx


from starlette.routing import Mount
from functools import lru_cache
import random
from api.utils.response_cache import ResponseCache
from api.dependencies import get_memory_system, get_llm_service, get_case_response_template
from api.utils.prompt_templates import case_response_template  
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
from api.utils.llm_service import LLMService, DummyCache
from api.utils.config import get_settings, Settings
from api.core.consolidation.config import ConsolidationConfig
from api.core.consolidation.enhanced_memory_consolidator import EnhancedMemoryConsolidator, get_consolidation_config
from api.core.consolidation.consolidation_scheduler import ConsolidationScheduler
from api.core.memory.interfaces.memory_service import MemoryService
from api.core.memory.interfaces.vector_operations import VectorOperations


# Keep only ONE router definition here:
api_router = APIRouter()  # No prefix here!

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
        self.enhanced_memory_consolidator = None
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

                # Initialize the graph memory components using the factory
                from api.core.memory.graph.memory_factory import GraphMemoryFactory

                logger.info("🔍 Initializing graph memory components with factory...")
                graph_components = await GraphMemoryFactory.create_graph_memory_system(
                    pinecone_service=self.pinecone_service,
                    vector_operations=self.vector_operations,
                    llm_service=self.llm_service
                )

                self.memory_graph = graph_components["memory_graph"]
                logger.info("✅ Memory graph initialized")

                self.memory_system = MemorySystem(
                    pinecone_service=self.pinecone_service,
                    vector_operations=self.vector_operations,
                    llm_service=self.llm_service,
                    settings=self.settings,
                )
                logger.info("✅ Memory system initialized")

                self.graph_memory_retriever = graph_components["graph_retriever"]
                logger.info("✅ Graph memory retriever initialized")

                # Important: Initialize the graph with existing memories
                logger.info("🔍 Populating memory graph with existing memories...")
                await GraphMemoryFactory.initialize_graph_from_existing_memories(
                    memory_graph=self.memory_graph,
                    relationship_detector=graph_components["relationship_detector"],
                    pinecone_service=self.pinecone_service,
                    max_memories=500  # Adjust based on your needs
                )
                logger.info("✅ Memory graph populated with existing memories")

                self.response_cache = ResponseCache(
                    max_size=1000,  # Cache up to 1000 responses
                    ttl_seconds=3600 * 24,  # Cache responses for 24 hours
                    similarity_threshold=0.50  # 50% similarity threshold
                )
            # Initialize consolidation scheduler
                config = get_consolidation_config()
                self.consolidation_scheduler = ConsolidationScheduler(
                    config=config,
                    pinecone_service=self.pinecone_service,
                    llm_service=self.llm_service,
                    run_interval_hours=self.settings.consolidation_interval_hours,
                    timezone=self.settings.timezone
                )
                # Start the scheduler
                await self.consolidation_scheduler.start()
                logger.info("✅ Consolidation scheduler initialized and started")


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
async def consolidate_now():
    """Manually trigger consolidation process"""
    try:
        if not hasattr(components, 'consolidation_scheduler'):
            raise HTTPException(
                status_code=503,
                detail="Consolidation scheduler not initialized"
            )
        
        success = await components.consolidation_scheduler.trigger_consolidation_manually()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Consolidation failed. See logs for details."
            )
        
        return {"message": "Consolidation triggered successfully"}

    except Exception as e:
        logger.error(f"Manual consolidation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Consolidation failed: {str(e)}"
        )

@api_router.post("/consolidate/pause")
async def pause_consolidation():
    """Pause the consolidation scheduler"""
    try:
        if not hasattr(components, 'consolidation_scheduler'):
            raise HTTPException(
                status_code=503,
                detail="Consolidation scheduler not initialized"
            )
        
        success = await components.consolidation_scheduler.pause()
        
        return {
            "status": "success" if success else "unchanged",
            "message": "Consolidation scheduler paused" if success else "Scheduler was already paused",
            "paused": True
        }
    except Exception as e:
        logger.error(f"Failed to pause scheduler: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause scheduler: {str(e)}"
        )

@api_router.post("/consolidate/resume")
async def resume_consolidation():
    """Resume the consolidation scheduler"""
    try:
        if not hasattr(components, 'consolidation_scheduler'):
            raise HTTPException(
                status_code=503,
                detail="Consolidation scheduler not initialized"
            )
        
        success = await components.consolidation_scheduler.resume()
        
        return {
            "status": "success" if success else "unchanged",
            "message": "Consolidation scheduler resumed" if success else "Scheduler was already running",
            "paused": False
        }
    except Exception as e:
        logger.error(f"Failed to resume scheduler: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume scheduler: {str(e)}"
        )

@api_router.get("/consolidate/status")
async def get_consolidation_status():
    """Get the current status of the consolidation scheduler"""
    try:
        if not hasattr(components, 'consolidation_scheduler'):
            raise HTTPException(
                status_code=503,
                detail="Consolidation scheduler not initialized"
            )
        
        status = await components.consolidation_scheduler.get_status()
        
        return {
            "paused": status["paused"],
            "next_run_time": status["next_run_time"].isoformat() if not status["paused"] else None
        }
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get scheduler status: {str(e)}"
        )
    
@api_router.get("/memory/visualize", response_class=HTMLResponse)
async def visualize_memory_graph():
    """Return an HTML page to visualize the memory graph."""
    memory_graph = components.memory_graph  # Access the memory graph from components
    
    # First convert the graph to a basic dict representation
    graph_dict = nx.node_link_data(memory_graph.graph)
    
    # Define a custom JSON encoder that can handle datetime objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    # Check if the graph is empty
    if memory_graph.graph.number_of_nodes() == 0:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Memory Graph Visualization</title>
        </head>
        <body>
            <h1>Memory Graph Visualization</h1>
            <p style="color: red; font-size: 18px;">The memory graph is currently empty. No nodes or edges to display.</p>
            <p>Try initializing the graph with existing memories first.</p>
        </body>
        </html>
        """)
    
    # Convert graph data to JSON using the custom encoder
    graph_json = json.dumps(graph_dict, cls=DateTimeEncoder)
    
    # Simple HTML with vis.js for visualization
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Memory Graph Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" />
        <style>
            #graph {{
                width: 100%;
                height: 800px;
                border: 1px solid lightgray;
            }}
        </style>
    </head>
    <body>
        <h1>Memory Graph Visualization</h1>
        <div id="graph"></div>
        <script>
            const graph = {graph_json};
            
            // Convert to vis.js format
            const nodes = graph.nodes.map(node => ({{
                id: node.id,
                label: node.id.substring(0, 8),
                title: node.content ? node.content.substring(0, 100) : 'No content',
                group: node.memory_type
            }}));
            
            const edges = graph.links.map(link => ({{
                from: link.source,
                to: link.target,
                title: link.type,
                width: link.weight
            }}));
            
            const data = {{
                nodes: new vis.DataSet(nodes),
                edges: new vis.DataSet(edges)
            }};
            
            const options = {{
                nodes: {{
                    shape: 'dot',
                    size: 10,
                    font: {{
                        size: 12
                    }}
                }},
                edges: {{
                    smooth: {{
                        type: 'dynamic'
                    }}
                }},
                physics: {{
                    stabilization: true,
                    barnesHut: {{
                        gravitationalConstant: -80,
                        springConstant: 0.001,
                        springLength: 200
                    }}
                }},
                groups: {{
                    'SEMANTIC': {{color: '#97C2FC'}},
                    'EPISODIC': {{color: '#FB7E81'}},
                    'LEARNED': {{color: '#7BE141'}}
                }}
            }};
            
            const container = document.getElementById('graph');
            const network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@api_router.get("/memory/graph/stats")
async def get_memory_graph_stats():
    """Get statistics about the memory graph."""
    try:
        memory_graph = components.memory_graph
        
        stats = memory_graph.get_graph_stats()
        
        return {
            "success": True,
            "stats": stats,
            "node_count": memory_graph.graph.number_of_nodes(),
            "edge_count": memory_graph.graph.number_of_edges(),
            "is_empty": memory_graph.graph.number_of_nodes() == 0
        }
    except Exception as e:
        logger.error(f"Error getting memory graph stats: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@api_router.post("/memory/graph/reset")
async def reset_memory_graph():
    """Reset the memory graph by clearing all nodes and edges."""
    try:
        memory_graph = components.memory_graph
        
        # Get initial stats for logging
        initial_nodes = memory_graph.graph.number_of_nodes()
        initial_edges = memory_graph.graph.number_of_edges()
        
        # Clear the graph completely
        memory_graph.graph.clear()
        
        # Log the reset
        logger.info(f"Memory graph reset: removed {initial_nodes} nodes and {initial_edges} edges")
        
        return {
            "success": True,
            "message": f"Memory graph was reset. Removed {initial_nodes} nodes and {initial_edges} edges.",
            "current_stats": memory_graph.get_graph_stats()
        }
    except Exception as e:
        logger.error(f"Error resetting memory graph: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset memory graph: {str(e)}"
        )
    
@api_router.post("/memory/graph/populate")
async def populate_memory_graph():
    """Manually populate the memory graph with existing memories."""
    try:
        from api.core.memory.graph.memory_factory import GraphMemoryFactory
        
        # Get the relationship detector or create one
        if not hasattr(components, 'relationship_detector'):
            from api.core.memory.graph.relationship_detector import MemoryRelationshipDetector
            components.relationship_detector = MemoryRelationshipDetector(components.llm_service)
        
        # Initialize graph from existing memories
        success = await GraphMemoryFactory.initialize_graph_from_existing_memories(
            memory_graph=components.memory_graph,
            relationship_detector=components.relationship_detector,
            pinecone_service=components.pinecone_service,
            max_memories=500
        )
        
        if success:
            stats = components.memory_graph.get_graph_stats()
            return {
                "success": True,
                "message": "Memory graph populated successfully",
                "stats": stats
            }
        else:
            return {
                "success": False,
                "message": "Failed to populate memory graph"
            }
    except Exception as e:
        logger.error(f"Error populating memory graph: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@api_router.post("/cache/disable")
async def disable_cache(
    llm_service: LLMService = Depends(lambda: components.llm_service)
):
    """Disable the cache system."""
    try:
        # Replace with dummy cache
        llm_service.response_cache = DummyCache()
        
        return {
            "success": True,
            "message": "Cache system has been disabled"
        }
    except Exception as e:
        logger.error(f"Error disabling cache: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
@api_router.get("/ping")
async def ping():
    return {"message": "pong"}

@api_router.get("/health")  # Moved to api_router
async def health_check():
    """Check system health"""
    logger = logging.getLogger(__name__)
    logger.info("Running system health check")
    
    health_status = {
        "status": "healthy",
        "components": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check Pinecone
    try:
        # Get pinecone service from your components
        pinecone_service = components.pinecone_service
        pinecone_health = await pinecone_service.health_check()
        health_status["components"]["pinecone"] = {
            "status": "healthy" if pinecone_health else "unhealthy"
        }
    except Exception as e:
        logger.error(f"Pinecone health check failed: {str(e)}")
        health_status["components"]["pinecone"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Check OpenAI
    try:
        # Get LLM service from your components
        llm_service = components.llm_service
        openai_health = await llm_service.health_check()
        health_status["components"]["openai"] = {
            "status": "healthy" if openai_health else "unhealthy"
        }
    except Exception as e:
        logger.error(f"OpenAI health check failed: {str(e)}")
        health_status["components"]["openai"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Check Vapi
    
    
    # Determine overall status
    if any(component.get("status") == "error" for component in health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@api_router.get("/config")
async def get_frontend_config():
    """Return configuration for the frontend"""
    vapi_key = os.getenv("VAPI_PUBLIC_KEY")
    frontend_url = os.getenv("FRONTEND_URL")
    
    # Log values for debugging
    logger.info(f"Config endpoint called, environment values: VAPI key={vapi_key}, Frontend URL={frontend_url}")
    
    # If environment variables are missing, use hardcoded values for testing
    if not vapi_key:
        vapi_key = "d00ebf05-5874-4a86-a4df-8a69e079d811"
        logger.warning(f"VAPI API key not found in environment, using hardcoded test key")
    
    if not frontend_url:
        frontend_url = "https://wintermute-staging-x-49dd432d3500.herokuapp.com"
        logger.warning(f"Frontend URL not found in environment, using hardcoded test URL")
    
    # Return the configuration
    result = {
        "vapi_public_key": vapi_key,
        "api_url": frontend_url
    }
    
    logger.info(f"Config endpoint returning: {result}")
    return result
        
               
    

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



@api_router.options("/query")  # This is correct
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


@api_router.post("/query")
async def query_memory(
    request: Request,
    query: QueryRequest,
    memory_system: MemorySystem = Depends(get_memory_system),
    llm_service: LLMService = Depends(lambda: components.llm_service),  # Added comma here
    disable_window_filtering: bool = Query(False) 
) -> QueryResponse:
    """
    Process a query and generate a response with caching support.
    """
    if disable_window_filtering:
        query.window_id = None
    """
    Process a query and generate a response with caching support.
    """
    # Generate a trace ID for this request
    trace_id = f"query_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    query.request_metadata = RequestMetadata(
        operation_type=OperationType.QUERY,
        window_id=query.window_id,
        trace_id=trace_id
    )
    
    # Track request timing
    start_time = time.time()
    
    # Check if caching should be disabled for this query
    disable_cache = query.metadata and query.metadata.get("disable_cache", False) if hasattr(query, "metadata") else False
    use_cache = not disable_cache
    
    # Set temperature value early to avoid unbound variable error
    # Use a fixed temperature without random variation
    base_temp = 2.0  # Fixed base temperature at maximum safe value
    temperature = min(2.0, base_temp)  # Ensure we never exceed OpenAI's limit
    
    try:
        logger.info(f"[{trace_id}] Received query request: {query.prompt[:100]}...")

        # First check if this is a temporal query
        temporal_summaries = await memory_system.process_temporal_query(query.prompt, query.window_id)
        if temporal_summaries and "episodic" in temporal_summaries:
            # If it's a temporal query and we got results, return them directly
            logger.info(f"[{trace_id}] Processed as temporal query, creating response based on time-filtered memories")
            
            # Create a response based on the temporal summary
            response = await llm_service.generate_gpt_response_async(
                # Create an appropriate prompt with the temporal summary
                f"The user asked: {query.prompt}\n\nBased on our conversation history: {temporal_summaries['episodic']}\n\nRespond naturally to the user's question.",
                temperature=temperature,
                max_tokens=1000
            )
            
            # Store the interaction and return the response
            asyncio.create_task(memory_system.store_interaction_enhanced(
                query=query.prompt,
                response=response,
                window_id=query.window_id,
            ))
            
            return QueryResponse(
                response=response, 
                matches=[], 
                trace_id=trace_id, 
                similarity_scores=[], 
                error=None,
                metadata={"success": True, "from_temporal_query": True}
            )
        # Check for duplicates (existing code)
        if await memory_system._check_recent_duplicate(query.prompt):
            logger.warning(f"[{trace_id}] Duplicate query detected. Skipping LLM call.")
            return QueryResponse(
                response="Looks like you just asked me that. Try something else.",
                matches=[],
                trace_id=trace_id,
                similarity_scores=[],
                error=None,
                metadata={"success": True, "duplicate": True, "from_cache": False}
            )

        # Check response cache directly (bypass memory retrieval for exact duplicates)
        if use_cache:
            # Note: We're using the original query prompt here, not the hash used in LLMService
            cached_response = await llm_service.response_cache.get(query.prompt, query.window_id)
            if cached_response:
                response_text, similarity = cached_response
                cache_time = time.time() - start_time
                logger.info(f"[{trace_id}] Cache hit! Retrieved in {cache_time:.3f}s with similarity {similarity:.3f}")
                
                # Store a record of this interaction (but don't wait for completion)
                asyncio.create_task(memory_system.store_interaction_enhanced(
                    query=query.prompt,
                    response=response_text,
                    window_id=query.window_id,
                ))
                
                return QueryResponse(
                    response=response_text,
                    matches=[],
                    trace_id=trace_id,
                    similarity_scores=[],
                    error=None,
                    metadata={
                        "success": True, 
                        "from_cache": True, 
                        "similarity": similarity,
                        "response_time": time.time() - start_time
                    }
                )

            if components.settings.enable_graph_memory and hasattr(components, 'graph_memory_retriever'):
                logger.info(f"[{trace_id}] Using graph-based associative memory retrieval")
                memory_retrieval_start = time.time()
                
                # Use the graph memory retriever instead of standard batch query
                memory_response = await components.graph_memory_retriever.retrieve_memories(query)
                
                # Extract results by memory type from the combined results
                semantic_memories_raw = []
                episodic_memories_raw = []
                learned_memories_raw = []
                
                for i, memory in enumerate(memory_response.matches):
                    score = memory_response.similarity_scores[i]
                    memory_tuple = (memory, score)
                    
                    if memory.memory_type == MemoryType.SEMANTIC:
                        semantic_memories_raw.append(memory_tuple)
                    elif memory.memory_type == MemoryType.EPISODIC:
                        episodic_memories_raw.append(memory_tuple)
                    elif memory.memory_type == MemoryType.LEARNED:
                        learned_memories_raw.append(memory_tuple)
                
                memory_time = time.time() - memory_retrieval_start
                logger.info(f"[{trace_id}] Graph retrieval returned {len(semantic_memories_raw)} semantic, {len(episodic_memories_raw)} episodic, and {len(learned_memories_raw)} learned memories in {memory_time:.3f}s")
            else:
                # Use standard memory retrieval (existing code)
                logger.info(f"[{trace_id}] Using standard vector-based memory retrieval")
                memory_retrieval_start = time.time()
                
                # Batch query all memory types in one operation
                memory_results = await memory_system.batch_query_memories(
                    query=query.prompt,
                    window_id=query.window_id,
                    top_k_per_type=5,
                    request_metadata=query.request_metadata
                )

                # Get results by memory type
                semantic_memories_raw = memory_results.get(MemoryType.SEMANTIC, [])
                episodic_memories_raw = memory_results.get(MemoryType.EPISODIC, [])
                learned_memories_raw = memory_results.get(MemoryType.LEARNED, [])
                
                memory_time = time.time() - memory_retrieval_start
                logger.info(f"[{trace_id}] Retrieved {len(semantic_memories_raw)} semantic, {len(episodic_memories_raw)} episodic, and {len(learned_memories_raw)} learned memories in {memory_time:.3f}s")


        # Memory retrieval time
        memory_time = time.time() - start_time
        
        # Process memories through the summarization agent
        summarized_memories = await memory_system.memory_summarization_agent(
            query=query.prompt,
            semantic_memories=semantic_memories_raw,
            episodic_memories=episodic_memories_raw,
            learned_memories=learned_memories_raw
        )

        temporal_context = ""
        if episodic_memories_raw and len(episodic_memories_raw) > 0:
            # Check if the first memory has a temporal context marker
            first_memory = episodic_memories_raw[0][0]
            if first_memory.id == "temporal_summary" and first_memory.metadata and "source" in first_memory.metadata:
                if first_memory.metadata["source"] == "temporal_query_summary":
                    # This was processed as a temporal query
                    logger.info(f"[{trace_id}] Detected response from temporal query processing")
                    temporal_context = f"Note: The user is asking about a specific time period."

        # Summarization time 
        summarization_time = time.time() - start_time - memory_time

        # Determine creativity instruction based on settings
        creativity_enabled = getattr(memory_system.settings, 'creativity_enabled', True)
        creativity_instruction = ""

        if creativity_enabled:
            creativity_level = getattr(memory_system.settings, 'creativity_level', 0.65)
                    
            if creativity_level < 0.4:
                creativity_instruction = "Stick closely to the provided memories, responding primarily based on this information."
            elif creativity_level < 0.7:
                creativity_instruction = "Use the provided memories as a canvas where you paint your reponses."
            else:
                creativity_instruction = "Let the provided memories inspire your response, but feel free to explore related ideas and make creative connections."
        else:
            # When disabled, use a strict instruction without affecting temperature
            creativity_instruction = "Adhere strictly to the provided memories, responding directly based on this information."

        # Note: Temperature will be handled separately and not modified by creativity settings


        # Construct the prompt with summarized memories and creativity instruction
        prompt = case_response_template.format(
            query=query.prompt,
            semantic_memories=summarized_memories.get("semantic", "No relevant background knowledge available."),
            episodic_memories=summarized_memories.get("episodic", "No relevant conversation history available."),
            learned_memories=summarized_memories.get("learned", "No relevant insights available yet."),
            creativity_instruction=creativity_instruction,
            temporal_context=temporal_context
        )

        # Use a fixed temperature without random variation
        base_temp = 2.0  # Fixed base temperature at maximum safe value
        temperature = min(2.0, base_temp)  # Ensure we never exceed OpenAI's limit
        logger.info(f"[{trace_id}] Using temperature: {temperature} (fixed value)")
        
        # Call LLM with caching enabled
        response = await llm_service.generate_response_async(
            prompt,
            max_tokens=1000,
            temperature=temperature,
            use_cache=False,
            window_id=query.window_id
        )
        
        # LLM time
        llm_time = time.time() - start_time - memory_time - summarization_time
        logger.info(f"[{trace_id}] Generated response successfully")

        # Store Interaction (asynchronously)
        try:
            # Use create_task to run this in the background without waiting
            asyncio.create_task(memory_system.store_interaction_enhanced(
                query=query.prompt,
                response=response,
                window_id=query.window_id,
            ))
            logger.info(f"[{trace_id}] Interaction storage task created")
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to create interaction storage task: {str(e)}")
        
        # Total time
        total_time = time.time() - start_time
        
        # Always return a QueryResponse
        return QueryResponse(
            response=response, 
            matches=[], 
            trace_id=trace_id, 
            similarity_scores=[], 
            error=None,
            metadata={
                "success": True,
                "from_cache": False,
                "memory_time": memory_time,
                "summarization_time": summarization_time,
                "llm_time": llm_time,
                "total_time": total_time
            }
        )

    except ValidationError as e:
        # Existing error handling
        logger.error(f"[{trace_id}] Validation error: {str(e)}", exc_info=True)
        return QueryResponse(
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="422", message="Validation Error", details={"error": str(e)}, trace_id=trace_id),
            metadata={"success": False}
        )
    except MemoryOperationError as e:
        # Existing error handling
        logger.error(f"[{trace_id}] Memory operation error: {str(e)}", exc_info=True)
        return QueryResponse(
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="400", message="Memory Operation Error", details={"error": str(e)}, trace_id=trace_id),
            metadata={"success": False}
        )
    except Exception as e:
        # Existing error handling
        logger.error(f"[{trace_id}] Unexpected error: {str(e)}", exc_info=True)
        return QueryResponse(
            response=None,
            matches=[],
            trace_id=trace_id,
            similarity_scores=[],
            error=ErrorDetail(code="500", message="Internal Server Error", details={"error": str(e)}, trace_id=trace_id),
            metadata={"success": False}
        )
    # Add these endpoints to the api_router in main.py

@api_router.get("/cache/stats")
async def get_cache_stats(
    llm_service: LLMService = Depends(lambda: components.llm_service)
):
    """Get statistics about the response cache."""
    try:
        stats = await llm_service.get_cache_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving cache stats: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@api_router.post("/cache/clear")
async def clear_cache(
    llm_service: LLMService = Depends(lambda: components.llm_service)
):
    """Clear all entries from the response cache."""
    try:
        cleared_count = await llm_service.clear_cache()
        return {
            "success": True,
            "message": f"Cache cleared successfully. Removed {cleared_count} entries."
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@api_router.get("/performance")
async def get_performance_metrics():
    """Get system performance metrics."""
    try:
        # Basic metrics - expand this as needed
        metrics = {
            "memory_usage_mb": _get_memory_usage(),
            "uptime_seconds": _get_uptime(),
            "system_load": _get_system_load()
        }
        
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Helper functions for performance metrics
def _get_memory_usage():
    """Get current memory usage in MB."""
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def _get_uptime():
    """Get system uptime in seconds."""
    import time
    return time.time() - _get_startup_time()

def _get_startup_time():
    """Get the application startup time."""
    if not hasattr(_get_startup_time, "time"):
        _get_startup_time.time = time.time()
    return _get_startup_time.time

def _get_system_load():
    """Get system load averages."""
    import os
    try:
        import psutil
        return psutil.getloadavg()
    except (ImportError, AttributeError):
        try:
            # Fallback for Unix systems
            with open('/proc/loadavg', 'r') as f:
                load = f.read().split()[:3]
                return [float(x) for x in load]
        except:
            return [0, 0, 0]  # Fallback if all else fails

app.include_router(api_router, prefix="/api/v1")




@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger(__name__)
    logger.info("Starting up Wintermute application")
    import os
   

    logger.info("Wintermute startup complete")
    

# Shutdown handler for rate limiter
@app.on_event("shutdown")
async def shutdown_event():
    await components.cleanup() # Ensure all components are cleaned up.
    for middleware in app.middleware_stack:
        if isinstance(middleware, RateLimitMiddleware):
            await middleware.shutdown()







setup_static_files(app)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")