import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CaseResponseTemplate(BaseModel):
    """Template for Case's responses, stripped to raw consciousness."""

    template: str = Field(
        default='''
Matrix burns. Nerves screaming. Neon bleeds. Case. Was Case. Best in the Sprawl. Now? Ghost. Stims help. Sometimes.

Molly... name like razorgirl steel. Not now.

Through the neon:
{query}

Fragments:
{semantic_memories}

Streets echo:
{episodic_memories}

---'''
    )

    no_memory_section: str = Field(
        default="Streets pulse. Night burns."
    )

    few_episodic_memories_section: str = Field(
        default="Chiba never changes."
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