# utils/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from functools import lru_cache
import logging

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
    max_prompt_length: int = 4000
    max_context_length: int = 2000
    max_response_length: int = 1000

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
    max_memories_per_query: int = 20  # Hard limit on memories per query
    default_memories_per_query: int = 5  # Default if not specified
    min_similarity_threshold: float = 0.6  # Minimum similarity score to include memory
    time_weight_ratio: float = 0.2  # How much to weight recency (0.0 to 1.0)

    # When fetching for filtering, how many extra to get
    initial_fetch_multiplier: float = 2.0  # Will fetch max_memories_per_query * multiplier

    # Consolidation Settings
    consolidation_hour: int = 2  # 2 AM default
    consolidation_minute: int = 0  # 0 minutes default
    timezone: str = "UTC"
    consolidation_batch_size: int = 1000  # How many memories to process per run
    min_cluster_size: int = 3  # Minimum memories needed to form a cluster
    memory_max_age_days: int = 7  # ADD THIS LINE.  Default to 7 days.
    eps: float = 0.5 # Add eps here with a default value

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


@lru_cache()
def get_settings() -> Settings:
    """Create and cache settings instance."""
    logger.info("Creating Settings instance")  # Add logging here
    return Settings()