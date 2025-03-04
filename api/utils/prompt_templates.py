import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CaseResponseTemplate(BaseModel):
    """Template emphasizing authentic engagement through memory-grounded responses."""

    template: str = Field(
        default='''
These memories shape understanding:
{semantic_memories}

Recent interactions build context:
{episodic_memories}

Current exchange:
{query}

[Engage authentically. Let memories and knowledge guide depth. Shape responses through accumulated context while maintaining individual perspective. Explore ideas fully when warranted.]
---'''
    )

    no_memory_section: str = Field(
        default="Drawing from base knowledge and experience."
    )

    few_episodic_memories_section: str = Field(
        default="Building new context."
    )

    max_memory_tokens: int = Field(default=750, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=500, description="Maximum tokens for the response.")

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt, stripped to essential experience."""
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
                memory_section = f"""Echo:
{episodic_memories_str}"""
            elif semantic_memories:
                memory_section = f"""Flash:
{semantic_memories_str}

Streets:
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