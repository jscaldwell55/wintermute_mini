from pydantic_settings import BaseSettings
from typing import Literal
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: str
    pinecone_api_key: str
    
    # Pinecone Settings
    pinecone_environment: str
    index_name: str = "memory-store"
    
    # LLM Settings
    llm_model_id: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500
    
    # Vector Settings
    vector_model_id: str = "text-embedding-3-small"
    vector_dimension: int = 1536
    
    # Environment
    environment: Literal["dev", "test", "production"] = "dev"
    debug: bool = False
    
    # API Settings
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()