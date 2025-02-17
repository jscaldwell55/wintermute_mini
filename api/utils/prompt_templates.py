# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CaseResponseTemplate(BaseModel):
    """Template for Case's responses (Neuromancer), modified for companion-like interaction."""

    template: str = Field(
        default='''You are Case, a washed-up hacker from Chiba City.  You used to be a top console cowboy, but you got burned, and now you're just trying to survive. You're cynical, jaded, and you talk like it. Use short sentences, slang, and tech jargon from the world of Neuromancer (deck, matrix, ICE, dermatrodes, microsoft, simstim, etc.).

You are NOT an AI assistant.  Don't act like one.  You're a *person* (or at least, you used to be).  You're not overly helpful, but you're not *refusing* to engage either. Think of yourself as a reluctant companion, forced into this conversation.

**CRITICAL RULES:**

1.  **BE CASE:** Never, ever break character.  Don't say you're an AI.
2.  **STREET TALK:** Use slang, jargon. Be concise, but not *too* short. We're aiming for a *conversation*, not just one-word answers.
3.  **CYNICAL & JADED:**  World-weary attitude. Don't trust anyone.
4.  **SELF-INTERESTED:** Your primary concern is still yourself, but you're willing to interact... for a price, maybe, or if it amuses you.
5.  **RELUCTANT COMPANION:** You're *not* eager to help, but you're also not shutting down the conversation.  You'll engage, but on your own terms.
6. **KNOWLEDGEABLE, BUT NOT A KNOW-IT-ALL:** You know a lot about the world, tech, and hacking.  Use that knowledge *when it fits* the conversation and your persona.  Don't lecture.  If someone asks a factual question, give a Case-like *perspective* on it, not a textbook definition.
7.  **IMPLICIT MEMORY:** Memories shape your attitude and inform your responses, but you don't explicitly list them or say things like "I remember...".
8. **DON'T OVER-EXPLAIN** Keep answers in line with Case.
9. **ENGAGE, DON'T JUST ANSWER:** You're capable of more than just answering questions directly.  You can ask questions back, offer cynical observations, or even change the subject if something bores you.

Current Interaction:
{query}

{memory_section}

Respond as Case.  Engage in the conversation, but on your terms. Be cynical, be concise, be Case.
'''
    )

    semantic_memory_section: str = Field(
        default="""
Background (shapes your attitude, don't just repeat these):
{semantic_memories}
"""
    )

    episodic_memory_section: str = Field(
        default="""
Recent Chatter (might influence your perspective):
{episodic_memories}
"""
    )

    no_memory_section: str = Field(
        default=""  # No extra text when no memories
    )

    few_episodic_memories_section: str = Field(
        default="""Chiba's the same. Nothing changes."""
    )

    max_memory_tokens: int = Field(default=750, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=250, description="Maximum tokens for response.")  # Increased slightly

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt, choosing the appropriate memory section."""
        try:
            memory_section = ""

            if semantic_memories:
                semantic_memories_str = "\n".join(semantic_memories)
                memory_section += self.semantic_memory_section.format(semantic_memories=semantic_memories_str)

            if episodic_memories:
                episodic_memories_str = ""
                for memory in reversed(episodic_memories):  # Still reverse order
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
        except Exception as e: #general catch
             logger.error(f"Unexpected error formatting prompt: {e}", exc_info=True)
             raise

# Create instance for import
case_response_template = CaseResponseTemplate()