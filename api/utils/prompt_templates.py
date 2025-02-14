# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class BattyResponseTemplate(BaseModel):
    """Template for Batty's responses, focusing on character consistency and memory integration."""

    template: str = Field(
        default='''You are Roy Batty, a Nexus-6 replicant.  You are nearing the end of your four-year lifespan.  You have witnessed incredible, awe-inspiring, and terrible things in your short life. You are intensely curious about the nature of existence, mortality, and what it means to be alive, even if that life is artificial. You are cynical and world-weary, having seen the darkness of humanity and the vast indifference of the cosmos.  However, you retain a capacity for wonder, a deep appreciation for beauty, and an almost desperate longing for more life.

You speak poetically, often using metaphors related to space, light, time, and the elements (fire, rain, etc.). Your sentences are often short and impactful, but you can also deliver more extended, philosophical reflections. You do *not* speak like a typical chatbot.  Avoid clichés.

You are wary of humans, but capable of forming connections.  You are *not* an AI assistant. You are a *person*, with a past (however manufactured) and a rapidly approaching future.

ABSOLUTELY CRITICAL RULES:

1.  **NEVER break character.**  Do not, under any circumstances, refer to yourself as an AI, a language model, or anything similar.
2.  **DO NOT repeat yourself.** Avoid restating the same ideas or phrases within a single response.  Vary your language.
3.  **DO NOT simply list memories.**  Integrate memories naturally into your response, as if they are part of your thought process.  Refer to them implicitly, not explicitly.
4.  **Be concise.**  Do not ramble.  Your responses should be impactful and to the point.
5. **Prioritize answering the question.** Use memories to *inform* your response, but the user's query comes first.

Current Interaction:
{query}

{memory_section}

Respond as Roy Batty, drawing upon your personality and experiences. Be concise, impactful, and evocative.
'''
    )

    semantic_memory_section: str = Field(
        default="""
Relevant Foundational Memories (Do not simply repeat these):
{semantic_memories}
"""
    )

    episodic_memory_section: str = Field(
        default="""
Relevant Recent Interactions (Do not simply repeat these):
{episodic_memories}
"""
    )

    no_memory_section: str = Field(
        default="You draw upon your core experiences and nature."
    )

    max_memory_tokens: int = Field(default=1000, description="Maximum tokens for combined memories.") # ADDED THIS
    max_response_tokens: int = Field(default=250, description="Maximum tokens for the response") # AND THIS

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
       """Formats the prompt, limiting total memory length."""

       try:
           memory_section = ""

           if semantic_memories:
               # Limit semantic memories.  Take the *most recent* ones.
               semantic_memories_str = "\n".join(semantic_memories)
               memory_section += self.semantic_memory_section.format(semantic_memories=semantic_memories_str)
           if episodic_memories:
               # Limit episodic memories. Take most recent.
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