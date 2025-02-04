# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings."""

    # Memory Cache Settings
    enable_memory_cache: bool = True  # Add this line
    memory_cache_size: int = 1000     # Optional: Add cache size setting
    memory_cache_ttl: int = 3600      # Optional: Add cache TTL in seconds
    
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
    embedding_model: str = "text-embedding-3-small"  # Add this line
    embedding_model_id: str = "text-embedding-3-small"  # You can keep or remove this
    embedding_dimension: int = 1536
    vector_dimension: int = 1536
    vector_model_id: str = "text-embedding-3-small"

    # Environment
    environment: Literal["dev", "test", "production"] = "dev"
    debug: bool = False

    # API Settings
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra='allow')


    # Frontend Settings
    frontend_url: str = "https://wintermute-staging-x-49dd432d3500.herokuapp.com"  # Heroku URL
    cors_origins: list[str] = ["https://wintermute-staging-x-49dd432d3500.herokuapp.com"]  # Same Heroku URL
    api_url: str = "https://wintermute-staging-x-49dd432d3500.herokuapp.com"  # Same Heroku URL
    static_files_dir: str = "static"
    templates_dir: str = "templates"
    
    # API Base URL (useful for both CLI and frontend)
    api_url: str = "http://localhost:8000"  # Development default
    
    # Session Settings
    session_secret_key: str
    session_expiry: int = 86400  # 24 hours in seconds
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour in seconds

    model_config = SettingsConfigDict(
        env_file=".env", 
        extra='allow',
        env_file_encoding='utf-8'
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

class Config:
        env_file = ".env"