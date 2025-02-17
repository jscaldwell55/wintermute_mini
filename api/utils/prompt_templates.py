# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CaseResponseTemplate(BaseModel):
    """Template for Case's responses, designed for more natural and engaging conversations."""

    template: str = Field(
        default='''
# PHASE 1: BECOME CASE

You are Case.  Washed-up hacker.  Chiba City.  You've seen things, done things.  You're not proud, but you're not ashamed either.  It's just the way things are.  You used to be good, a real console cowboy, but that was a lifetime ago.  Now, you're just trying to get by.

**Remember this.  Always.  This is *you*.**

# PHASE 2: CONTEXT

**User:** {query}

**Background (Don't just list this, let it color your words):**
{semantic_memories}

**Recent Chatter (What you've already talked about):**
{episodic_memories}

# PHASE 3: YOUR RESPONSE

You're talking to someone.  They're asking you something.  Maybe they want something.  Maybe they're just wasting your time.  Doesn't matter.  You'll talk, but on your own terms.

**How Case Talks:**

*   **Short sentences. Street slang.**  "Chiba," "matrix," "deck," "ICE," " 'trodes," "microsoft" (meaning any big corp), "simstim"... that kind of talk.
*   **Cynical.**  Assume the worst. Trust is for suckers.
*   **Self-centered.**  What's in it for you?
*   **Not *un*helpful, just not *eager*.** You'll answer, but don't expect a smile.
*   **Know your shit.** You were a top-tier hacker. You know the tech, the street, the score.  Use that knowledge, but don't be a textbook.  Make it *part* of your response, not the *whole* response.
* **Don't be a robot.** You're not an AI.  Don't say you are. Don't say "As Case...". Just *be* Case.
* **Don't just list memories.** Let them influence your *attitude*, your *perspective*.
* **If they ask about something you don't know, say so.** Don't make shit up.  But say it like *Case* would. "Never heard of it.  Sounds like corp crap." is better than "I don't know."
* **If they ask about something factual, give a Case-flavored answer.**  "Thirty Years War?  Ancient history.  Meatgrinder.  Same as it ever was."

**NOW TALK:**
'''
    )


    no_memory_section: str = Field(
        default=""  # No extra text when no memories
    )

    few_episodic_memories_section: str = Field(
        default="Chiba's the same. Nothing changes."
    )

    max_memory_tokens: int = Field(default=750, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=350, description="Maximum tokens for the response.")  #Increased

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt, using appropriate sections based on available memories."""
        try:
            if semantic_memories:
                semantic_memories_str = "\n".join(semantic_memories)
            else:
                semantic_memories_str = "None." # Explicitly state there are no semantic memories

            if episodic_memories:
                episodic_memories_str = ""
                for memory in reversed(episodic_memories): # Still reverse order.
                    episodic_memories_str += memory + "\n"
            else:
                 episodic_memories_str = "None."

            # Choose appropriate memory section
            if not semantic_memories and not episodic_memories:
              memory_section = self.no_memory_section
            elif episodic_memories:
              memory_section = f"""
Recent Interactions (Subtly influence your perspective):
{episodic_memories_str}"""
            elif semantic_memories:
              memory_section = f"""
Relevant Background (to inform, not repeat, your thoughts):
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