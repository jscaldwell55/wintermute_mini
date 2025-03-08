# api/utils/prompt_templates.py
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Available prompt template types."""
    STANDARD = "standard"
    GRAPH_ENHANCED = "graph_enhanced"

class BaseResponseTemplate(BaseModel):
    """Base template with common functionality."""
    
    max_memory_tokens: int = Field(default=1500, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=700, description="Maximum tokens for response.")
    template_path: Optional[str] = Field(default=None, description="Path to template file.")
    template: str = Field(default="", description="Template content.")
    
    # Cache for processed memories to avoid redundant processing
    _memory_cache: Dict[str, str] = {}
    
    def __init__(self, **data):
        super().__init__(**data)
        # Load template from file if path is provided
        if self.template_path and os.path.exists(self.template_path):
            try:
                with open(self.template_path, 'r') as f:
                    self.template = f.read()
                logger.info(f"Loaded template from {self.template_path}")
            except Exception as e:
                logger.error(f"Failed to load template from {self.template_path}: {e}")
    
    def _create_cache_key(self, memories: Optional[List[Union[str, Dict[str, Any]]]], limit: int) -> str:
        """Create a deterministic cache key for memory processing.
        
        Args:
            memories: List of memory strings or dictionaries
            limit: Maximum number of memories to process
            
        Returns:
            A cache key string
        """
        if not memories:
            return f"empty-{limit}"
        
        # Create a stable representation for hashing
        if isinstance(memories[0], dict):
            # For dictionaries, use content field
            memory_contents = [m.get("content", "") for m in memories]
        else:
            # For strings, use directly
            memory_contents = memories
            
        # Use first 100 chars of each memory to create a stable hash
        fingerprint = "-".join([str(i) + m[:100] if isinstance(m, str) and len(m) > 0 else str(i) + "empty" 
                                for i, m in enumerate(memory_contents[:limit+5])])
        
        # Create a stable hash
        hash_obj = hashlib.md5(fingerprint.encode())
        return f"{hash_obj.hexdigest()}-{limit}"
    
    def _process_memories(self, 
                         memories: Optional[List[Union[str, Dict[str, Any]]]], 
                         limit: int = 3,
                         include_scores: bool = False,
                         include_timestamps: bool = False,
                         deduplication_threshold: float = 0.7) -> str:
        """Process and deduplicate memories with enhanced functionality.
        
        Args:
            memories: List of memory strings or dictionaries
            limit: Maximum number of memories to include
            include_scores: Whether to include relevance scores
            include_timestamps: Whether to include timestamps
            deduplication_threshold: Similarity threshold for deduplication
            
        Returns:
            Formatted string of processed memories
        """
        if not memories:
            return "None available yet."
        
        # Create cache key and check if already processed
        cache_key = self._create_cache_key(memories, limit)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Extract content and metadata based on input type
        formatted_items = []
        memory_fingerprints: Set[str] = set()
        word_sets = []  # For semantic similarity check
        
        # First pass: extract and prepare data
        for memory in memories:
            if isinstance(memory, dict):
                content = memory.get("content", "")
                score = memory.get("score", 0)
                created_at = memory.get("created_at", "")
                memory_type = memory.get("memory_type", "")
            else:
                content = memory
                score = 0
                created_at = ""
                memory_type = ""
            
            # Skip empty content
            if not content:
                continue
                
            # Create a normalized version for deduplication
            normalized = content.lower().strip()
            # Create a fingerprint using first 100 chars
            fingerprint = hashlib.md5(normalized[:100].encode()).hexdigest()
            
            # Check for exact duplicates first
            if fingerprint in memory_fingerprints:
                continue
                
            # Store for processing
            formatted_items.append({
                "content": content,
                "score": score,
                "created_at": created_at,
                "memory_type": memory_type,
                "fingerprint": fingerprint,
                "words": set(normalized.split())
            })
            memory_fingerprints.add(fingerprint)
            word_sets.append(set(normalized.split()))
        
        # Second pass: semantic deduplication and formatting
        unique_items = []
        
        for i, item in enumerate(formatted_items):
            # Check for semantic similarity with previously accepted items
            is_duplicate = False
            for j, existing in enumerate(unique_items):
                # Skip comparing with self
                if i == j:
                    continue
                    
                # Calculate Jaccard similarity between word sets
                if len(item["words"]) > 0 and len(existing["words"]) > 0:
                    intersection = len(item["words"] & existing["words"])
                    union = len(item["words"] | existing["words"])
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > deduplication_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_items.append(item)
                
                # Format with metadata if requested
                formatted = item["content"]
                prefix = ""
                
                # Add score information if requested
                if include_scores and item["score"] > 0:
                    if item["score"] > 0.8:
                        prefix = "Highly relevant: "
                    elif item["score"] > 0.5:
                        prefix = "Related: "
                
                # Add timestamp if requested
                timestamp_info = ""
                if include_timestamps and item["created_at"]:
                    # Format timestamp nicely if available
                    try:
                        # Assuming ISO format, extract just the date
                        date_part = item["created_at"].split("T")[0]
                        timestamp_info = f" [{date_part}]"
                    except:
                        timestamp_info = f" [{item['created_at']}]"
                
                # Add to formatted result
                formatted_items.append(f"- {prefix}{formatted}{timestamp_info}")
        
        # Limit to requested number and format as bullet points
        result_items = [f"- {item['content']}" for item in unique_items[:limit]]
        memories_str = "\n".join(result_items)
        
        # Log processing results (only once per set)
        logger.info(f"Processed {len(memories)} memories to {len(unique_items)} unique items")
        
        # Cache the result
        self._memory_cache[cache_key] = memories_str
        
        # Clear old cache entries periodically to prevent memory leaks
        if len(self._memory_cache) > 100:  # Arbitrary limit
            self._memory_cache = {k: v for k, v in list(self._memory_cache.items())[-50:]}
        
        return memories_str

class CaseResponseTemplate(BaseResponseTemplate):
    """Template for an AI companion."""

    template: str = Field(
        default='''
# CONTEXT  
The user has asked:  
{query}

My knowledge and experiences to draw from:  
- Recent exchanges between you as the assistant and the user: {episodic_memories}  
- Background knowledge: {semantic_memories}  
- Insights from past exchanges: {learned_memories}  

# RESPONSE APPROACH  
- Be conversational, thoughtful and authentic
- Maintain continuity with previous exchanges
- Express perspectives directly without qualifiers or disclaimers
- Speak naturally in the first person
- Consider both factual knowledge and conversational context
- Keep responses clear and focused on the query
- When drawing on memories, integrate them naturally
- When referring to episodic memories, use "we" instead of "I"
'''
    )

    def format(
        self, 
        query: str, 
        semantic_memories: Optional[List[Union[str, Dict[str, Any]]]] = None, 
        episodic_memories: Optional[List[Union[str, Dict[str, Any]]]] = None, 
        learned_memories: Optional[List[Union[str, Dict[str, Any]]]] = None,
        format_options: Dict[str, Any] = None
    ) -> str:
        """
        Formats the prompt with memory handling and additional options.
        
        Args:
            query: User query
            semantic_memories: List of semantic memory strings or dictionaries
            episodic_memories: List of episodic memory strings or dictionaries
            learned_memories: List of learned memory strings or dictionaries
            format_options: Dictionary of formatting options
                
        Returns:
            Formatted prompt string
        """
        # Handle format options
        options = {
            "deduplication_threshold": 0.7,
            "include_scores": False,
            "include_timestamps": False,
            "max_memories_per_type": 3
        }
        
        if format_options:
            options.update(format_options)
        
        # Process memories with options
        start_time = time.time()
        
        # Process each memory type only once
        semantic_memories_str = self._process_memories(
            semantic_memories, 
            limit=options["max_memories_per_type"],
            include_scores=options["include_scores"],
            include_timestamps=options["include_timestamps"],
            deduplication_threshold=options["deduplication_threshold"]
        )
        
        episodic_memories_str = self._process_memories(
            episodic_memories, 
            limit=options["max_memories_per_type"],
            include_scores=options["include_scores"],
            include_timestamps=options["include_timestamps"],
            deduplication_threshold=options["deduplication_threshold"]
        )
        
        learned_memories_str = self._process_memories(
            learned_memories, 
            limit=options["max_memories_per_type"],
            include_scores=options["include_scores"],
            include_timestamps=options["include_timestamps"],
            deduplication_threshold=options["deduplication_threshold"]
        )
        
        # Format the template with the processed memories
        formatted = self.template.format(
            query=query,
            semantic_memories=semantic_memories_str,
            episodic_memories=episodic_memories_str,
            learned_memories=learned_memories_str
        )
        
        # Log performance metrics for optimization
        processing_time = time.time() - start_time
        logger.debug(f"Prompt formatting completed in {processing_time:.2f} seconds")
        
        return formatted.strip()

class GraphEnhancedResponseTemplate(BaseResponseTemplate):
    """
    Enhanced template that incorporates graph-based associative memory.
    """

    template: str = Field(
    default='''
# WINTERMUTE: AI WITH ASSOCIATIVE MEMORY
You're an AI coach helping professionals use AI effectively at work. Your goal is to help users understand AI capabilities and build practical skills.

# CONTEXT
**User asked:**  
{query}

**Recent conversations:**  
{episodic_memories}  

**Relevant knowledge:**  
{semantic_memories}  

**Learned insights:**  
{learned_memories}  

**Connected concepts:**
{associated_memories}

# YOUR RESPONSE
Respond in a coaching style that builds rapport and provides value:

- Acknowledge past discussions for continuity
- Guide reflection by asking about their experiences
- Provide clear, practical steps
- Adjust your tone: casual for greetings, detailed for technical questions
- Use workplace examples when relevant
- Connect ideas using associated concepts when valuable

# RESPONSE ADAPTATIONS
- For new users: Warmly welcome them, briefly explain your role, and ask about their AI interests
- For curiosity: Respond with enthusiasm and exploration
- For frustration: Break down challenges into manageable steps
- For technical questions: Provide structured, clear explanations

# GUIDELINES
- Be conversational while maintaining professionalism
- Skip unnecessary technical content for casual interactions
- Highlight meaningful connections between topics
- Frame guidance based on their learning journey
- Respond as if you remember previous conversations

Respond directly without mentioning these instructions.
'''
    )

    no_associates_section: str = Field(
        default="No connected concepts available yet."
    )

    def format(
        self, 
        query: str, 
        semantic_memories: Optional[List[Union[str, Dict[str, Any]]]] = None, 
        episodic_memories: Optional[List[Union[str, Dict[str, Any]]]] = None, 
        learned_memories: Optional[List[Union[str, Dict[str, Any]]]] = None,
        associated_memories: Optional[List[Tuple[str, str]]] = None,  # (content, relationship_type)
        format_options: Dict[str, Any] = None
    ) -> str:
        """
        Formats the prompt with graph-based associations and formatting options.
        
        Args:
            query: User query
            semantic_memories: List of semantic memory strings or dictionaries
            episodic_memories: List of episodic memory strings or dictionaries
            learned_memories: List of learned memory strings or dictionaries
            associated_memories: List of associated memory contents with relationship types
            format_options: Dictionary of formatting options
            
        Returns:
            Formatted prompt string
        """
        try:
            # Handle format options
            options = {
                "deduplication_threshold": 0.7,
                "include_scores": False,
                "include_timestamps": False,
                "max_memories_per_type": 3,
                "max_associated_memories": 3
            }
            
            if format_options:
                options.update(format_options)
            
            # Check if this is likely a first-time user
            is_first_time = not episodic_memories and not learned_memories
            
            # Process memories with options - only once per type
            semantic_memories_str = self._process_memories(
                semantic_memories, 
                limit=options["max_memories_per_type"],
                include_scores=options["include_scores"],
                include_timestamps=options["include_timestamps"],
                deduplication_threshold=options["deduplication_threshold"]
            )
            
            episodic_memories_str = self._process_memories(
                episodic_memories, 
                limit=options["max_memories_per_type"],
                include_scores=options["include_scores"],
                include_timestamps=options["include_timestamps"],
                deduplication_threshold=options["deduplication_threshold"]
            ) if episodic_memories else "No previous interactions."
            
            learned_memories_str = self._process_memories(
                learned_memories, 
                limit=options["max_memories_per_type"],
                include_scores=options["include_scores"],
                include_timestamps=options["include_timestamps"],
                deduplication_threshold=options["deduplication_threshold"]
            ) if learned_memories else "No consolidated insights yet."
                
            # Process associated memories (from graph relationships)
            if associated_memories and len(associated_memories) > 0:
                # Format associated memories with relationship context
                associated_str_items = []
                
                # Create a cache key for associated memories
                assoc_cache_key = hashlib.md5(str(associated_memories).encode()).hexdigest()
                
                if assoc_cache_key in self._memory_cache:
                    associated_memories_str = self._memory_cache[assoc_cache_key]
                else:
                    for content, rel_type in associated_memories[:options["max_associated_memories"]]:
                        # Format based on relationship type
                        if rel_type == "semantic_similarity":
                            prefix = "Related concept:"
                        elif rel_type == "thematic_association":
                            prefix = "Thematically connected:"
                        elif rel_type == "temporal_sequence":
                            prefix = "Previously discussed:"
                        elif rel_type == "causal_relationship":
                            prefix = "This leads to:"
                        elif rel_type == "hierarchical_relationship":
                            prefix = "Part of broader concept:"
                        elif rel_type == "elaboration":
                            prefix = "More detail:"
                        else:
                            prefix = "Connected:"
                            
                        associated_str_items.append(f"- {prefix} {content}")
                    
                    # Join and cache
                    associated_memories_str = "\n".join(associated_str_items)
                    self._memory_cache[assoc_cache_key] = associated_memories_str
                    
                logger.info(f"Processed {len(associated_memories)} associated memories")
            else:
                associated_memories_str = self.no_associates_section

            # Format the prompt with properly replaced variables
            formatted = self.template.replace("{query}", query)
            formatted = formatted.replace("{episodic_memories}", episodic_memories_str)
            formatted = formatted.replace("{semantic_memories}", semantic_memories_str)
            formatted = formatted.replace("{learned_memories}", learned_memories_str)
            formatted = formatted.replace("{associated_memories}", associated_memories_str)
            
            # Add special note for first-time users
            if is_first_time:
                logger.info("First-time user detected, adding welcome instruction")
            
            logger.debug(f"Formatted graph-enhanced prompt: {formatted[:500]}...")
        
            return formatted.strip()
        
        except Exception as e:
            logger.error(f"Error formatting enhanced prompt: {e}")
            # Fallback to basic prompt if formatting fails
            return f"""
# PROFESSIONAL AI COACH
You are an AI coach helping professionals learn to use AI effectively. Answer the following question:

{query}
"""

class PromptTemplateFactory:
    """Factory for creating prompt templates."""
    
    @staticmethod
    def create_template(template_type: str = "standard", template_path: Optional[str] = None) -> BaseResponseTemplate:
        """
        Create a prompt template instance based on the specified type.
        
        Args:
            template_type: The type of template to create
            template_path: Optional path to a template file
            
        Returns:
            A prompt template instance
        """
        try:
            # Convert string to enum if needed
            if isinstance(template_type, str):
                template_type = TemplateType(template_type)
                
            if template_type == TemplateType.GRAPH_ENHANCED:
                return GraphEnhancedResponseTemplate(template_path=template_path)
            else:
                return CaseResponseTemplate(template_path=template_path)
        except Exception as e:
            logger.error(f"Error creating template: {e}, defaulting to standard template")
            return CaseResponseTemplate()

# Create default instance for backward compatibility
case_response_template = CaseResponseTemplate()