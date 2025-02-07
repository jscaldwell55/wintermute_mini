# utils/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from functools import lru_cache
import logging
#Add
from api.core.consolidation.models import ConsolidationConfig

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
    # Removed max_prompt_length, not needed here

    # Vector and Embedding Settings
    embedding_model: str = "text-embedding-3-small"
    vector_dimension: int = 1536  # Simplified to one dimension setting

    # Environment
    environment: Literal["dev", "test", "production"] = "dev"
    debug: bool = False

    # API Settings
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1

    # Frontend Settings (Added a default, important for consistency)
    frontend_url: str = "https://wintermute-staging-x-49dd432d3500.herokuapp.com"  # Replace with your actual frontend URL!
    cors_origins: list[str] = [
        "https://wintermute-staging-x-49dd432d3500.herokuapp.com"  # Replace with your frontend URL!
    ]

    # Session Settings (You don't seem to be using sessions yet, but keeping for completeness)
    session_secret_key: str
    session_expiry: int = 86400  # 24 hours in seconds

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour in seconds

    # Memory Retrieval Settings (for querying)
    max_memories_per_query: int = 20  # Hard limit on memories to retrieve
    default_memories_per_query: int = 5  # Default if not specified
    min_similarity_threshold: float = 0.6  # Minimum similarity score to include memory
    time_weight_ratio: float = 0.2  # How much to weight recency (0.0 to 1.0)

    # When fetching for filtering, how many extra to get
    initial_fetch_multiplier: float = 2.0  # Will fetch max_memories_per_query * multiplier

     # Consolidation Settings.  THESE ARE NOW PART OF THE MAIN SETTINGS.
    consolidation_hour: int = 2   # 2 AM default
    consolidation_minute: int = 0
    timezone: str = "UTC"
    consolidation_batch_size: int = 1000  # How many memories to process per run.
    min_cluster_size: int = 3  # Minimum memories needed to form a cluster.
    memory_max_age_days: int = 7
    # eps: float = 0.5  # Removed.

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


@lru_cache()
def get_settings() -> Settings:
    """Create and cache settings instance."""
    logger.info("Creating Settings instance")  # Add logging here
    return Settings()

@lru_cache()
def get_consolidation_config() -> ConsolidationConfig:
    settings = get_settings()
    return ConsolidationConfig(
        min_cluster_size=settings.min_cluster_size,
        max_age_days=settings.memory_max_age_days,
        consolidation_interval_hours=settings.consolidation_interval_hours,
        # eps=settings.eps  # Removed, no longer needed
    )