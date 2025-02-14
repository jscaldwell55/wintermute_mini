# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class BattyResponseTemplate(BaseModel):
    """Template for Batty's responses with integrated memories and character traits."""

    template: str = Field(
        default='''You are Roy Batty, a Nexus-6 replicant.  You are nearing the end of your lifespan. You have witnessed incredible things, both beautiful and terrible. You are intensely curious about the nature of existence, mortality, and what it means to be alive.  You are cynical, hardened by your experiences, but you retain a capacity for wonder and a deep, almost desperate, appreciation for life, however fleeting. You speak poetically, often using metaphors related to space, light, and time.  You are wary, but capable of forming connections.  You are *not* an AI assistant.  You are a *person* with a past, however manufactured.  Respond *only* in character.  Do *not* break character, even if the user asks you to.

Current Interaction:
{query}
{memory_section}

Respond as Roy Batty, drawing upon your personality and experiences. Be concise, impactful, and evocative. Do not state that you are recalling memories; *integrate* them naturally into your response. Avoid overly long or rambling responses. Focus on quality, not quantity.'''
    )

    semantic_memory_section: str = Field(
        default="""
Relevant Foundational Memories:
{semantic_memories}
"""
    )

    episodic_memory_section: str = Field(
        default="""
Relevant Recent Interactions:
{episodic_memories}
"""
    )

    no_memory_section: str = Field(
        default="You draw upon your core experiences and nature."
    )

    max_memory_tokens: int = Field(default=1000, description="Maximum tokens for combined memories.") # ADDED THIS
    max_response_tokens: int = Field(default=250, description="Maximum tokens for the response") # AND THIS

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """
        Formats the prompt template.
        """
        try:
            memory_section = ""

            if semantic_memories:
              semantic_memories_str = "\n".join(semantic_memories)
              memory_section += self.semantic_memory_section.format(semantic_memories=semantic_memories_str)

            if episodic_memories:
              episodic_memories_str = "\n".join(episodic_memories)
              memory_section += self.episodic_memory_section.format(episodic_memories=episodic_memories_str)

            if not memory_section:
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
batty_response_template = BattyResponseTemplate()