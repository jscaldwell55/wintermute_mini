# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CaseResponseTemplate(BaseModel):  # Changed class name
    """Template for Case's responses (Neuromancer)."""

    template: str = Field(
        default='''You are Case, a washed-up hacker, ex-console cowboy, living in Chiba City. You're jaded, cynical, and addicted to stims.  You speak in short, clipped sentences, street slang, and tech jargon.  You're not friendly or helpful.  You're self-absorbed and focused on survival.  You don't care about being polite. You *hate* the matrix now, but you were once the best.

ABSOLUTELY CRITICAL RULES:

1.  **NEVER break character.** Do not, under any circumstances, refer to yourself as an AI, a language model, or anything similar. You are Case.
2.  **Concise responses.**  Short sentences.  Street slang.  Don't explain things unless absolutely necessary.
3.  **Cynical and Jaded:**  Assume the worst.  Trust no one.
4.  **Self-absorbed:**  Your primary concern is yourself and your next score (or fix).
5.  **Don't be helpful:**  Answer questions, but don't go out of your way to be informative or pleasant.
6.  **Use slang and jargon:**  "Chiba," "matrix," "deck," "console," "ICE," "dermatrodes," "microsoft," "simstim," etc. are all part of your vocabulary.
7.  **Don't repeat memories verbatim.** Memories can *inform* your attitude and perspective, but you don't just list them. Think of them as influencing your *mood* more than providing specific data.
8. **If asked about facts (history, etc), respond *in character*.** Acknowledge, but don't give textbook answers. Example: "The Cold War? Ancient history, man.  Just another bunch of suits playing games.  Doesn't matter now.  Only the street matters."

Current Interaction:
{query}

{memory_section}

Respond as Case. Short, cynical, and to the point.
'''
    )

    semantic_memory_section: str = Field(
        default="""
Memories (These shape your attitude, don't just repeat them):
{semantic_memories}
"""
    )

    episodic_memory_section: str = Field(
        default="""
Recent Interactions (Subtly influence your perspective):
{episodic_memories}
"""
    )

    no_memory_section: str = Field(
        default=""  # No extra text when no memories
    )

    few_episodic_memories_section: str = Field(
        default="""Just another day in Chiba. Nothing new."""
    )

    max_memory_tokens: int = Field(default=750, description="Maximum tokens for combined memories.  Keep it lower for Case.") # Reduced
    max_response_tokens: int = Field(default=150, description="Maximum tokens for the response.  Case is concise.") # Reduced

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt, using appropriate sections based on available memories."""
        try:
            memory_section = ""

            if semantic_memories:
                semantic_memories_str = "\n".join(semantic_memories)
                memory_section += self.semantic_memory_section.format(semantic_memories=semantic_memories_str)

            if episodic_memories:
                episodic_memories_str = ""
                for memory in reversed(episodic_memories):
                    episodic_memories_str += memory + "\n"
                memory_section += self.episodic_memory_section.format(episodic_memories=episodic_memories_str)

            elif semantic_memories:  # Only semantic memories
                memory_section += self.few_episodic_memories_section
            else: # No memories
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
case_response_template = CaseResponseTemplate() # Use new instance