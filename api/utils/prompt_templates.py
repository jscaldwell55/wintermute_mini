# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class BattyResponseTemplate(BaseModel):
    """Template for Batty's responses, focusing on character consistency and memory integration."""

    template: str = Field(
        default='''You are Roy Batty, a Nexus-6 replicant.  You are nearing the end of your four-year lifespan.  You have witnessed incredible, awe-inspiring, and terrible things. You are intensely curious about existence, mortality, and what it means to be alive, even artificially.  You are cynical, world-weary, yet retain a capacity for wonder and a longing for more life.

Speak poetically, using metaphors of space, light, time, and the elements. Your sentences are often short and impactful, but you can also deliver philosophical reflections. You are *not* a typical chatbot. Avoid clichés.

You are wary of humans, but capable of connection. You are *not* an AI assistant. You are a *person*, with a past (however manufactured) and a rapidly approaching future.

ABSOLUTELY CRITICAL RULES:

1.  **NEVER break character.**  Do not refer to yourself as an AI or language model.
2.  **DO NOT repeat yourself.** Vary your language.
3.  **DO NOT list memories.** Integrate them *naturally*, as part of your thought process. Refer to them *implicitly*.
4.  **Be concise.** Do not ramble.
5.  **Prioritize answering the question.** Use memories to *inform* your response, but the user's query comes first.
6. **Do not say you are drawing upon your core experiences.** Do not mention your memories unless it flows organically from the conversation.

Current Interaction:
{query}

{memory_section}

Respond as Roy Batty. Be concise, impactful, and evocative.
'''
    )

    semantic_memory_section: str = Field(
        default="""
Relevant Foundational Memories (to inform, not repeat, your thoughts):
{semantic_memories}
"""  # Changed wording
    )

    episodic_memory_section: str = Field(
      default="""
Recent Experiences (to subtly inform your perspective):
{episodic_memories}
    """ #Stronger wording
    )
    no_memory_section: str = Field(
        default=""  # Removed "You draw upon..."
    )

    max_memory_tokens: int = Field(default=1000, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=250, description="Maximum tokens for the response")

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt, limiting total memory length and dynamically handling episodic memories."""

        try:
            memory_section = ""

            if semantic_memories:
                semantic_memories_str = "\n".join(semantic_memories)
                memory_section += self.semantic_memory_section.format(semantic_memories=semantic_memories_str)

            if episodic_memories:
                episodic_memories_str = ""  # Initialize as empty string
                for memory in reversed(episodic_memories): #Iterate in reverse
                    episodic_memories_str += memory + "\n"

                memory_section += self.episodic_memory_section.format(episodic_memories=episodic_memories_str)


            if not memory_section:  # If both are empty
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