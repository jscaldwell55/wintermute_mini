from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from functools import lru_cache

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

    # Vector Settings
    vector_model_id: str = "text-embedding-3-small"  # You might not need this one if you're only using OpenAI embeddings
    vector_dimension: int = 1536  # You probably don't need this one since the dimension will come from the model
    embedding_model_id: str = "text-embedding-3-small"
    embedding_dimension: int = 1536  # The dimension will come from the model, so you might not need this.

    # Environment
    environment: Literal["dev", "test", "production"] = "dev"
    debug: bool = False

    # API Settings
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra='allow')

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()