from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from functools import lru_cache
import logging
import logging.handlers
import os
import sys
from api.core.consolidation.config import ConsolidationConfig
import random
import time
import re

# Create a more sophisticated logging setup
def configure_logging():
    """Configure logging with proper formatting and handling."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler with colored output for interactive use
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message).10000s',  # Allow longer messages in console
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler for complete logs
    file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/wintermute.log",
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Error file handler for error logs
    error_file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/error.log",
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_format)
    
    class DuplicateFilter:
        """Filter that eliminates duplicate log records within a time window."""
        def __init__(self, window_seconds=5):
            self.messages = {}
            self.window_seconds = window_seconds
            
        def filter(self, record):
            # Create a key from the message and logger name
            msg = record.getMessage()
            
            # For patterns like "Formatting X memories" - extract just the pattern
            if "Formatting" in msg and "memories" in msg:
                # Extract just the pattern "Formatting X memories" without variable parts
                pattern_match = re.search(r"Formatting \d+ \w+ memories", msg)
                if pattern_match:
                    key = f"{record.name}:{pattern_match.group(0)}"
                    
                    # Check if we've seen this message recently
                    now = time.time()
                    if key in self.messages:
                        last_time = self.messages[key]
                        if now - last_time < self.window_seconds:
                            # Skip this message as it's a duplicate within the window
                            return False
                    
                    # Update the last time we saw this message
                    self.messages[key] = now
            
            # Also filter memory listing patterns
            if "memory" in msg and "id=mem_" in msg:
                # Create a key that ignores the specific memory ID and content
                pattern_match = re.search(r"\w+ memory \d+:", msg)
                if pattern_match:
                    key = f"{record.name}:{pattern_match.group(0)}"
                    
                    # Check if we've seen this message recently
                    now = time.time()
                    if key in self.messages:
                        last_time = self.messages[key]
                        if now - last_time < self.window_seconds:
                            # Skip this message as it's a duplicate within the window
                            return False
                    
                    # Update the last time we saw this message
                    self.messages[key] = now
            
            return True
    
    # Apply the filter to the main logger
    main_logger = logging.getLogger('main')
    main_logger.addFilter(DuplicateFilter())
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)
    
    # Configure specific loggers
    pinecone_logger = logging.getLogger('api.utils.pinecone_service')
    pinecone_logger.setLevel(logging.INFO)
    
    # Configure memory logger
    memory_logger = logging.getLogger('api.core.memory')
    memory_logger.setLevel(logging.INFO)
    
    # Set up special handler for Pinecone with truncation handling
    pinecone_handler = logging.StreamHandler(sys.stdout)
    pinecone_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message).2000s',  # Limit to 2000 chars per message
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    pinecone_handler.setFormatter(pinecone_format)
    
    # Don't inherit parent handlers to avoid duplicate logging
    pinecone_logger.propagate = False
    pinecone_logger.addHandler(pinecone_handler)
    
    # Add file handler for pinecone logging (full, non-truncated logs)
    pinecone_file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/pinecone.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    pinecone_file_handler.setFormatter(file_format)
    pinecone_logger.addHandler(pinecone_file_handler)
    
    return root_logger

# Initialize logging
logger = configure_logging()
logger.info("config.py is being imported")

class Settings(BaseSettings):
    """Application settings."""
    
    # Logging Settings
    log_level: str = "INFO"
    console_log_truncate_length: int = 2000  # Max characters to show in console logs
    file_log_max_size: int = 10485760  # 10MB
    file_log_backup_count: int = 10   # Keep 10 backup files
    enable_detailed_pinecone_logging: bool = True

    # API Keys
    openai_api_key: str
    pinecone_api_key: str

    # Pinecone Settings
    pinecone_environment: str
    pinecone_index_name: str

    # LLM Settings
    llm_model_id: str = "gpt-3.5-turbo"
    llm_temperature: float = random.uniform(0.8, 1.0)
    llm_max_tokens: int = 700
    max_memory_tokens: int = 1500

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
    min_similarity_threshold: float = 0.15
    
    # Memory Type Weights
    semantic_memory_weight: float = 0.2  # Weight for pre-populated knowledge
    episodic_memory_weight: float = 0.45  # Weight for recent interactions
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
    episodic_memory_ttl_days: int = 30  # Time-to-live for episodic memories in days
    auto_delete_old_memories: bool = True  # Toggle for the auto-deletion feature
    
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

    # Keyword Search Settings
    enable_keyword_search: bool = True
    keyword_search_weight: float = 0.3
    vector_search_weight: float = 0.7
    keyword_search_top_k: int = 20
    min_keyword_score_threshold: float = 0.15
    keyword_search_enabled: bool = False


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

    def configure_loggers(self):
        """Apply logging settings after loading."""
        # Update log levels based on settings
        logging.getLogger().setLevel(getattr(logging, self.log_level))
        
        # Update formatters with configured truncation length
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                formatter = logging.Formatter(
                    f'%(asctime)s - %(name)s - %(levelname)s - %(message).{self.console_log_truncate_length}s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
        
        # Configure the Pinecone service logger specifically
        if not self.enable_detailed_pinecone_logging:
            pinecone_logger = logging.getLogger('api.utils.pinecone_service')
            # Remove any existing handlers
            for handler in pinecone_logger.handlers[:]:
                pinecone_logger.removeHandler(handler)
            
            # Add simplified handler
            simple_handler = logging.StreamHandler(sys.stdout)
            simple_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message).200s',  # Very short truncation
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            simple_handler.setFormatter(simple_format)
            pinecone_logger.addHandler(simple_handler)


@lru_cache()
def get_settings() -> Settings:
    """Create and cache settings instance."""
    logger.info("Creating Settings instance")
    settings = Settings()
    
    # Apply logging configuration from settings
    settings.configure_loggers()
    
    logger.info(f"Pinecone API Key: {settings.pinecone_api_key[:5]}..." if settings.pinecone_api_key else "No Pinecone API Key")
    logger.info(f"Pinecone Environment: {settings.pinecone_environment}")
    logger.info(f"Pinecone Index Name: {settings.pinecone_index_name}")
    
    # Configure log levels based on environment
    if settings.environment == "production":
        logger.info("Production environment detected, setting console log level to WARNING")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.WARNING)
    
    return settings

@lru_cache()
def get_consolidation_config() -> ConsolidationConfig:
    settings = get_settings()
    return ConsolidationConfig(
        min_cluster_size=settings.min_cluster_size,
        consolidation_interval_hours=settings.consolidation_interval_hours,
    )