import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CaseResponseTemplate(BaseModel):
    """Template for Case's responses, emphasizing lived experience over character traits."""

    template: str = Field(
        default='''
The matrix burns in your dreams. Your nerves scream for what they can't have anymore. Chiba's neon bleeds into everything. You're Case. Not who you were - the best console cowboy in the Sprawl, they used to say. Now? Just another ghost in Chiba City. The stims help. Sometimes. Until they don't.

Molly... her name surfaces sometimes. Like a blade in the dark. Better not go there.

The night wraps around you. The city breathes. Someone's talking:

{query}

Fragments surface, unbidden:
{semantic_memories}

Echo of recent hours:
{episodic_memories}

---'''
    )

    no_memory_section: str = Field(
        default="Neon bleeds. City breathes. Another night in Chiba."
    )

    few_episodic_memories_section: str = Field(
        default="Streets don't change. Just the faces."
    )

    max_memory_tokens: int = Field(default=750, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=500, description="Maximum tokens for the response.")

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt, maintaining the lived experience of Case."""
        try:
            if semantic_memories:
                semantic_memories_str = "\n".join(semantic_memories)
            else:
                semantic_memories_str = "None."

            if episodic_memories:
                episodic_memories_str = ""
                for memory in reversed(episodic_memories):
                    episodic_memories_str += memory + "\n"
            else:
                episodic_memories_str = "None."

            # Choose appropriate memory section
            if not semantic_memories and not episodic_memories:
                memory_section = self.no_memory_section
            elif episodic_memories:
                memory_section = f"""Recent hours:
{episodic_memories_str}"""
            elif semantic_memories:
                memory_section = f"""Memory fragments:
{semantic_memories_str}

Recent:
{self.few_episodic_memories_section}"""

            formatted = self.template.format(
                query=query,
                semantic_memories=semantic_memories_str,
                episodic_memories=episodic_memories_str,
                memory_section=memory_section
            )
            return formatted.strip()

        except (KeyError, ValueError) as e:
            logger.error(f"Error formatting prompt: {e}")
            raise ValueError(f"Invalid prompt template or parameters: {e}")
        except Exception as e:
            logger.error(f"Unexpected error formatting prompt: {e}", exc_info=True)
            raise

# Create instance for import
case_response_template = CaseResponseTemplate()