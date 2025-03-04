# api/utils/config.py
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
    min_similarity_threshold: float = 0.6 # No longer used now that we enhance relevance.
    time_weight_ratio: float = 0.2 # No longer used
    initial_fetch_multiplier: float = 2.0 #No longer used

    # Consolidation Settings
    consolidation_hour: int = 2
    consolidation_minute: int = 0
    timezone: str = "UTC"
    consolidation_batch_size: int = 1000
    min_cluster_size: int = 3
    consolidation_interval_hours: int = 24

    # Memory Enhancement Settings
    enable_enhanced_relevance: bool = True
    enable_deduplication: bool = True
    semantic_top_k: int = 5  # NO LONGER USED, REMOVE
    episodic_top_k: int = 5 # NO LONGER USED, REMOVE
    min_similarity_threshold: float = 0.4  # NO LONGER USED, REMOVE, handled by enhanced relevance
    similarity_weight: float = 0.7 # Added. Used by enhanced relevance
    content_length_weight: float = 0.15 # Added.  Used by enhanced relevance
    unique_ratio_weight: float = 0.15 # Added.  Used by enhanced relevance
    deduplication_window_minutes: int = 60 # Added.
    duplicate_similarity_threshold: float = 0.98 # Added. Very high threshold.

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
    return settings

@lru_cache()
def get_consolidation_config() -> ConsolidationConfig:
    settings = get_settings()
    return ConsolidationConfig(
        min_cluster_size=settings.min_cluster_size,
        consolidation_interval_hours=settings.consolidation_interval_hours,
    )