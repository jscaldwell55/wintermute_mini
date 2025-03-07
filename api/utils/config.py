from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from functools import lru_cache
import logging
from api.core.consolidation.config import ConsolidationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("config.py is being imported")

class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    openai_api_key: str
    pinecone_api_key: str

    # Pinecone Settings
    pinecone_environment: str
    pinecone_index_name: str

    # LLM Settings
    llm_model_id: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500

    # Vector and Embedding Settings
    embedding_model: str = "text-embedding-3-small"
    vector_dimension: int = 1536

    # Environment
    environment: Literal["dev", "test", "production"] = "dev"
    debug: bool = False

    # API Settings
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1

    # Frontend Settings
    frontend_url: str = "https://wintermute-staging-x-49dd432d3500.herokuapp.com"
    cors_origins: list[str] = [
        "https://wintermute-staging-x-49dd432d3500.herokuapp.com"
    ]

    # Session Settings
    session_secret_key: str
    session_expiry: int = 86400  # 24 hours in seconds

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour in seconds

 # Memory Retrieval Settings
    max_memories_per_query: int = 20
    default_memories_per_query: int = 5
    min_similarity_threshold: float = 0.5
    
    # Memory Type Weights
    semantic_memory_weight: float = 0.4  # Weight for pre-populated knowledge
    episodic_memory_weight: float = 0.3  # Weight for recent interactions
    learned_memory_weight: float = 0.3   # Weight for consolidated insights
    
    # Memory Type Limits - NEW
    semantic_top_k: int = 5   # Retrieve up to 5 semantic memories
    episodic_top_k: int = 5   # Retrieve up to 5 episodic memories
    learned_top_k: int = 3    # Retrieve up to 3 learned memories
    
    # Episodic Memory Settings - NEW
    episodic_max_age_days: int = 7      # Only consider past 7 days
    episodic_recency_weight: float = 0.3  # Weight for recency in scoring
    episodic_recent_hours: int = 48     # Hours considered "recent" (higher priority)
    episodic_decay_factor: float = 120  # Controls exponential decay rate for older memories
    
    # Semantic Memory Settings - NEW
    semantic_min_words: int = 5  # Minimum word count for semantic memories
    
    # Learned Memory Settings - NEW
    learned_confidence_weight: float = 0.2  # Weight for confidence in scoring learned memories

    # Consolidation Settings
    consolidation_hour: int = 2
    consolidation_minute: int = 0
    timezone: str = "UTC"
    consolidation_batch_size: int = 1000
    min_cluster_size: int = 3
    consolidation_interval_hours: int = 24
    consolidation_output_type: str = "LEARNED"  # NEW - memories produced are LEARNED type
    
    # Memory Enhancement Settings
    enable_enhanced_relevance: bool = True
    enable_deduplication: bool = True
    similarity_weight: float = 0.7
    content_length_weight: float = 0.15
    unique_ratio_weight: float = 0.15
    deduplication_window_minutes: int = 60
    duplicate_similarity_threshold: float = 0.98

    # Graph Memory Settings
    enable_graph_memory: bool = False  # Toggle for enabling/disabling graph memory (for A/B testing)
    graph_memory_weight: float = 0.3   # Weight for graph-based retrievals in combined results
    vector_memory_weight: float = 0.7  # Weight for vector-based retrievals in combined results
    
    # Graph Traversal Settings
    max_graph_depth: int = 2           # Maximum hops in graph traversal
    max_memories_per_hop: int = 3      # Maximum memories to retrieve per hop
    association_score_decay: float = 0.7  # Score decay per hop in graph
    min_association_score: float = 0.3    # Minimum score to include an associated memory
    
    # Relationship Detection Settings
    semantic_similarity_threshold: float = 0.75  # Threshold for semantic relationships
    temporal_proximity_minutes: int = 30         # Time window for temporal relationships
    max_relationships_per_memory: int = 10       # Max number of relationships per memory
    
    # Prompt Template Settings
    template_type: str = "standard"    # "standard" or "graph_enhanced"
    
    # Evaluation Settings
    enable_memory_evaluation: bool = False  # Toggle for evaluation framework
    evaluation_sample_rate: float = 0.1     # Percentage of queries to evaluate

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


@lru_cache()
def get_settings() -> Settings:
    """Create and cache settings instance."""
    logger.info("Creating Settings instance")
    settings = Settings()
    logger.info(f"Pinecone API Key: {settings.pinecone_api_key}")
    logger.info(f"Pinecone Environment: {settings.pinecone_environment}")
    logger.info(f"Pinecone Index Name: {settings.pinecone_index_name}")
    # Removed VAPI logging
    return settings

@lru_cache()
def get_consolidation_config() -> ConsolidationConfig:
    settings = get_settings()
    return ConsolidationConfig(
        min_cluster_size=settings.min_cluster_size,
        consolidation_interval_hours=settings.consolidation_interval_hours,
    )