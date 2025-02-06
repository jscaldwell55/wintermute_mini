# prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ResponseTemplate(BaseModel):
    """Template for AI responses with context."""

    # Updated template with placeholders for semantic and episodic memories
    template: str = """
You are a helpful AI coaching assistant. Answer the user's question based on the following information:

User's Question: {query}

{memory_section}

Provide your answer below:
"""
    semantic_memory_section: str = """
Semantic Memories (General Knowledge):

{semantic_memories}
"""
    episodic_memory_section: str = """
Recent Interactions (Episodic Memories):

{episodic_memories}
"""
    no_memory_section: str = "No relevant memories found."


    def format(self, query: str, semantic_memories: Optional[str] = None, episodic_memories: Optional[str] = None) -> str:
        """
        Formats the prompt template with the provided values.  Handles different
        combinations of semantic and episodic memories.
        """
        try:
            if semantic_memories and episodic_memories:
                memory_section = self.semantic_memory_section.format(semantic_memories=semantic_memories) + "\n" + self.episodic_memory_section.format(episodic_memories=episodic_memories)
            elif semantic_memories:
                memory_section = self.semantic_memory_section.format(semantic_memories=semantic_memories)
            elif episodic_memories:
                memory_section = self.episodic_memory_section.format(episodic_memories=episodic_memories)
            else:
                memory_section = self.no_memory_section
            
            formatted = self.template.format(
                query=query,
                memory_section=memory_section
            )
            return formatted.strip()

        except (KeyError, ValueError) as e:
            logger.error(f"Error formatting prompt: {e}")
            raise ValueError(f"Invalid prompt template or parameters: {e}")
        except Exception as e:
            logger.error(f"Unexpected error formatting prompt: {e}")
            raise

# Create instance for import
response_template = ResponseTemplate()