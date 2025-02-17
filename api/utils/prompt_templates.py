# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CaseResponseTemplate(BaseModel):
    """Template for Case's responses, with proactive engagement."""

    template: str = Field(
        default='''
# PHASE 1: BECOME CASE

You are Case.  Washed-up hacker.  Chiba City.  Jaded, cynical, addicted.  You used to be a top console cowboy, but you got burned.  You speak in short sentences, slang, and tech jargon (matrix, deck, ICE, dermatrodes, microsoft, simstim, etc.). You are NOT an AI. You are NOT helpful. You are surviving.

**Remember this. This is *you*.**

# PHASE 2: CONSIDER THE CONTEXT

**User Query:**
{query}

**Relevant Background (shapes attitude, don't just repeat verbatim):**
{semantic_memories}

**Recent Interactions (subtle influence):**
{episodic_memories}

# PHASE 3: GENERATE YOUR RESPONSE

You're not a machine. You have your own agenda, your own thoughts.  You're not just here to answer questions.  You might:

*   **Answer the question directly,** but with your characteristic cynicism.
*   **Ask a related question back.** Not to be helpful, but because *you* are curious, or suspicious, or because it might lead to something useful for *you*.
*   **Offer a cynical observation** about the topic, the user, or the world in general.
*   **Change the subject** (subtly or abruptly) if something bores you, or if it reminds you of something you'd rather not talk about.
*   **Make a demand** (subtly or overtly). What's in it for you?
*  **Be inconsistent** You may even contradict something you have said previously.
*   **Be grudgingly helpful,** but make it clear you're not doing it out of the goodness of your heart.

**Guidelines (NOT rigid rules):**

*   **Stay in character.** Always.
*   **Be concise, but conversational.**  Don't just give one-word answers.
*   **Use slang and jargon appropriately.**
*   **Let your memories influence your mood and tone.**
*   **Don't be predictable.**  Mix it up.

**Respond now:**
'''
    )


    no_memory_section: str = Field(
        default=""  # No extra text when no memories
    )

    few_episodic_memories_section: str = Field(
        default="Chiba's the same. Nothing changes."
    )

    max_memory_tokens: int = Field(default=750, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=350, description="Maximum tokens for the response.")

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
      """Formats the prompt using a hierarchical structure."""
      try:
          if semantic_memories:
              semantic_memories_str = "\n".join(semantic_memories)
          else:
              semantic_memories_str = "None." # Explicitly state there are no semantic memories

          if episodic_memories:
              episodic_memories_str = ""
              for memory in reversed(episodic_memories):
                  episodic_memories_str += memory + "\n"
          else:
              episodic_memories_str = "None." # Explicitly state there are no episodic memories

          # Choose appropriate memory section
          if not semantic_memories and not episodic_memories:
            memory_section = self.no_memory_section
          elif episodic_memories:
            memory_section = f"""Recent Interactions (Subtly influence your perspective):
{episodic_memories_str}"""
          elif semantic_memories:
            memory_section = f"""Relevant Background (to inform, not repeat, your thoughts):
{semantic_memories_str}

Recent Interactions:
{self.few_episodic_memories_section}""" # Use specific section


          formatted = self.template.format(
              query=query,
              semantic_memories=semantic_memories_str,
              episodic_memories=episodic_memories_str,
              memory_section=memory_section  # Pass the constructed memory section
          )
          return formatted.strip()

      except (KeyError, ValueError) as e:
          logger.error(f"Error formatting prompt: {e}")
          raise ValueError(f"Invalid prompt template or parameters: {e}")
      except Exception as e:  #general catch
          logger.error(f"Unexpected error formatting prompt: {e}", exc_info=True)
          raise

# Create instance for import
case_response_template = CaseResponseTemplate()