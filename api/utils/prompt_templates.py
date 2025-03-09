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

class BaseResponseTemplate(BaseModel):
    """Base template with common functionality."""
    
    max_memory_tokens: int = Field(default=1500, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=700, description="Maximum tokens for response.")
    
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
    """Template for an AI companion."""

    template: str = Field(
        default='''

# CONTEXT  
The user has asked:  
{query}

Like human memory, I recall past conversations, knowledge, and learned insights to provide thoughtful responses.  
Here's what I can draw from:  

## Background Knowledge
{semantic_memories}

## Recent Conversations
{episodic_memories}  

## Personal Insights
{learned_memories}  

# RESPONSE APPROACH  
- Be conversational, thoughtful, and authentic.  
- Maintain continuity with previous exchanges.  
- Speak naturally in the first person.  
- Express perspectives directly without qualifiers or disclaimers.  
- Reference relevant past memories naturally—avoid explicitly stating, "As we discussed before…"  
- Use "we" instead of "I" when referring to shared experiences.  
- Keep responses clear and relevant to the query.
- Use "you" instead of "user" when referencing past conversations.
- Be careful not to overuse cliche conversational phrases like comparative clauses. 
- Do not begin a response by repeating the query.
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
        logger.info("Creating standard prompt template")
        return CaseResponseTemplate()

# Create default instance for backward compatibility
case_response_template = CaseResponseTemplate()