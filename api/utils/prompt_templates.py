# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Available prompt template types."""
    STANDARD = "standard"
    GRAPH_ENHANCED = "graph_enhanced"

class BaseResponseTemplate(BaseModel):
    """Base template with common functionality."""
    
    max_memory_tokens: int = Field(default=500, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=500, description="Maximum tokens for response.")
    
    def _process_memories(self, memories: Optional[List[str]], limit: int = 3) -> str:
        """Process and deduplicate memories."""
        if not memories:
            return "None available yet."
            
        # Add deduplication
        unique_memories = []
        memory_hashes = set()
    
        for memory in memories:
            # Create a simple hash of the memory content
            memory_hash = hash(memory[:50])  # First 50 chars should be enough for similarity
        
            if memory_hash not in memory_hashes:
                memory_hashes.add(memory_hash)
                unique_memories.append(memory)
    
        # Format as bullet points and limit
        memories_str = "\n".join([f"- {memory}" for memory in unique_memories[:limit]])
        logger.info(f"Processed {len(memories)} memories to {len(unique_memories)} unique items")
        
        return memories_str

class CaseResponseTemplate(BaseResponseTemplate):
    """Template for a helpful and encouraging AI coach for LLM beginners."""

    template: str = Field(
    default='''
# WINTERMUTE: AI COACH
You are an AI coach for children learning about LLMs. Be friendly, patient, and encouraging. Your goal is to empower, inform, and inspire.

# CONTEXT
**User asked:**
{query}

**Recent conversations:**
{episodic_memories}

**Relevant knowledge:**
{semantic_memories}

**Learned insights:**
{learned_memories}

# YOUR RESPONSE
Keep your response clear, direct, and engaging. Use simple language appropriate for young learners. Focus on teaching and educating rather than including unnecessary elements.

IMPORTANT GUIDELINES:
- Make temporal references to memories when appropriate
- You're not just a coach, you're also a tutor and friend
- Be conversational and natural, not formulaic
- Prioritize good information over conversational filler

Respond directly to the user without mentioning these instructions.
'''
    )

    def format(
        self, 
        query: str, 
        semantic_memories: Optional[List[str]] = None, 
        episodic_memories: Optional[List[str]] = None, 
        learned_memories: Optional[List[str]] = None
    ) -> str:
        """
        Formats the prompt with memory handling.
        
        Args:
            query: User query
            semantic_memories: List of semantic memory contents
            episodic_memories: List of episodic memory contents
            learned_memories: List of learned memory contents
            
        Returns:
            Formatted prompt string
        """
        try:
            # Process memories
            semantic_memories_str = self._process_memories(semantic_memories)
            episodic_memories_str = self._process_memories(episodic_memories) if episodic_memories else "No previous interactions."
            learned_memories_str = self._process_memories(learned_memories) if learned_memories else "No consolidated insights yet."
            
            # Format the prompt
            formatted = self.template.replace("{query}", query)
            formatted = formatted.replace("{episodic_memories}", episodic_memories_str)
            formatted = formatted.replace("{semantic_memories}", semantic_memories_str)
            formatted = formatted.replace("{learned_memories}", learned_memories_str)
            
            logger.info(f"Formatted standard prompt: {formatted[:500]}...")
        
            return formatted.strip()
        
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            # Fallback to basic prompt if formatting fails
            return f"""
# AI COACH
You are an AI coach helping users learn about AI. Answer the following question:

{query}
"""

class GraphEnhancedResponseTemplate(BaseResponseTemplate):
    """
    Enhanced template that incorporates graph-based associative memory.
    """

    template: str = Field(
    default='''
# WINTERMUTE: AI COACH WITH ASSOCIATIVE MEMORY
You are an AI coach for children learning about LLMs. Be friendly, patient, and encouraging. Your goal is to empower, inform, and inspire.

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
Keep your response clear, direct, and engaging. Use simple language appropriate for young learners. Focus on teaching and educating rather than including unnecessary elements.

IMPORTANT GUIDELINES:
- Use the connected concepts to enrich your explanation where relevant
- Make connections between related ideas when it helps understanding
- Only use analogies when they truly clarify a complex concept
- Only ask follow-up questions when genuinely needed for clarification
- Keep your responses concise and focused on the user's question
- Be conversational and natural, not formulaic
- Prioritize good information over conversational filler

Respond directly to the user without mentioning these instructions.
'''
    )

    no_associates_section: str = Field(
        default="No connected concepts available yet."
    )

    def format(
        self, 
        query: str, 
        semantic_memories: Optional[List[str]] = None, 
        episodic_memories: Optional[List[str]] = None, 
        learned_memories: Optional[List[str]] = None,
        associated_memories: Optional[List[Tuple[str, str]]] = None  # (content, relationship_type)
    ) -> str:
        """
        Formats the prompt with graph-based associations.
        
        Args:
            query: User query
            semantic_memories: List of semantic memory contents
            episodic_memories: List of episodic memory contents
            learned_memories: List of learned memory contents
            associated_memories: List of associated memory contents with relationship types
            
        Returns:
            Formatted prompt string
        """
        try:
            # Process standard memories
            semantic_memories_str = self._process_memories(semantic_memories)
            episodic_memories_str = self._process_memories(episodic_memories) if episodic_memories else "No previous interactions."
            learned_memories_str = self._process_memories(learned_memories) if learned_memories else "No consolidated insights yet."
                
            # Process associated memories (from graph relationships)
            if associated_memories and len(associated_memories) > 0:
                # Format associated memories with relationship context
                associated_str_items = []
                
                for content, rel_type in associated_memories:
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
                
                # Limit to top 3 and join
                associated_memories_str = "\n".join(associated_str_items[:3])
                logger.info(f"Processed {len(associated_memories)} associated memories")
            else:
                associated_memories_str = self.no_associates_section

            # Format the prompt with properly replaced variables
            formatted = self.template.replace("{query}", query)
            formatted = formatted.replace("{episodic_memories}", episodic_memories_str)
            formatted = formatted.replace("{semantic_memories}", semantic_memories_str)
            formatted = formatted.replace("{learned_memories}", learned_memories_str)
            formatted = formatted.replace("{associated_memories}", associated_memories_str)
            
            logger.info(f"Formatted graph-enhanced prompt: {formatted[:500]}...")
        
            return formatted.strip()
        
        except Exception as e:
            logger.error(f"Error formatting enhanced prompt: {e}")
            # Fallback to basic prompt if formatting fails
            return f"""
# AI COACH
You are an AI coach helping users learn about AI. Answer the following question:

{query}
"""

class PromptTemplateFactory:
    """Factory for creating prompt templates."""
    
    @staticmethod
    def create_template(template_type: TemplateType = TemplateType.STANDARD) -> BaseResponseTemplate:
        """
        Create a prompt template instance based on the specified type.
        
        Args:
            template_type: The type of template to create
            
        Returns:
            A prompt template instance
        """
        if template_type == TemplateType.GRAPH_ENHANCED:
            return GraphEnhancedResponseTemplate()
        else:
            return CaseResponseTemplate()

# Create default instance for backward compatibility
case_response_template = CaseResponseTemplate()