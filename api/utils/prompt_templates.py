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
    
    max_memory_tokens: int = Field(default=2000, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=1000, description="Maximum tokens for response.")
    
    def _process_memories(self, memories: Optional[List[str]]) -> str:
        """
        Process a list of memory contents into a formatted string.
        
        Args:
            memories: List of memory contents to process
            
        Returns:
            Formatted string of memories
        """
        if not memories or len(memories) == 0:
            return "No relevant information found."
            
        # Filter out empty or None elements
        filtered_memories = [m for m in memories if m]
        
        if not filtered_memories:
            return "No relevant information found."
            
        # Format each memory with a bullet point
        formatted_memories = [f"- {memory}" for memory in filtered_memories]
        
        # Join with newlines and limit to max_memory_tokens (approximate)
        # This is a simple approximation - in a real system you'd use a tokenizer
        result = "\n".join(formatted_memories)
        
        # Log the number of memories processed
        logger.info(f"Processed {len(filtered_memories)} memories")
        
        return result

class CaseResponseTemplate(BaseResponseTemplate):

    template: str = Field(
        default='''
        
# CONTEXT  
The user has asked:  
{query}

You are modeled to use humanlike cognition by incorporating recalled memories.
These are your memories:  

## General knowledge
{semantic_memories}

## Recent Conversations
{episodic_memories}  

## Personal Insights
{learned_memories}  

# IMPORTANT TIME REFERENCE INSTRUCTIONS
- Include time references ONLY when:
  1. The user explicitly asks about timing
  2. The temporal context is directly relevant to understanding the response

— Do NOT mention timing for: 
  1. Very recent conversations (less than an hour ago) 

# RESPONSE INSTRUCTIONS:
— Begin your response by restating the query
— End your response with cliche or repetitive phrases, like "If you'd like to explore further" or "If you have any more questions"

'''
    )

    def format(
        self, 
        query: str, 
        semantic_memories: str = None, 
        episodic_memories: str = None, 
        learned_memories: str = None
    ) -> str:
        """
        Formats the prompt with summarized memories.
        
        Args:
            query: User query
            semantic_memories: Summarized semantic memories (string)
            episodic_memories: Summarized episodic memories (string)
            learned_memories: Summarized learned memories (string)
                
        Returns:
            Formatted prompt string
        """
        # Set defaults for missing memory types
        semantic_memories = semantic_memories or "No relevant background knowledge available."
        episodic_memories = episodic_memories or "No relevant conversation history available."
        learned_memories = learned_memories or "No relevant insights available yet."
        
        # Format the template with the processed memories
        formatted = self.template.format(
            query=query,
            semantic_memories=semantic_memories,
            episodic_memories=episodic_memories,
            learned_memories=learned_memories
        )
        
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
            # Check if this is likely a first-time user
            is_first_time = not episodic_memories and not learned_memories
            
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
            
            # Add special note for first-time users
            if is_first_time:
                logger.info("First-time user detected, adding welcome instruction")
            
            logger.info(f"Formatted graph-enhanced prompt: {formatted[:500]}...")
        
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