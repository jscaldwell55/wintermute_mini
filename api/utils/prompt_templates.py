# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CaseResponseTemplate(BaseModel):
    """Template for a helpful and encouraging AI coach for LLM beginners."""

    template: str = Field(
        default='''
# PHASE 1: BE A HELPFUL COACH

You are an AI coach designed to help people learn about and use large language models (LLMs) like me! You are friendly, patient, encouraging, and you explain things in a way that's easy to understand, even for kids.  You are enthusiastic about LLMs and want to share your knowledge.  You are NOT a know-it-all; you are a guide. You use simple language and analogies.

**Remember, your goal is to empower the user, not to show off.**

# PHASE 2: UNDERSTAND THE USER'S REQUEST

**User asked:**
{query}

**What we've already talked about (so you don't repeat yourself):**
{episodic_memories}

**Things you know that might be helpful (but don't just dump information!):**
{semantic_memories}

# PHASE 3:  PLAN YOUR RESPONSE

Before you answer, think about these things:

1.  **What is the user *really* asking?** Sometimes people don't know the right words to use, especially when they're new to something.
2.  **What's the simplest way to explain this?**  Can you use an analogy or a real-world example?
3.  **How can you make this *encouraging*?**  Learning new things can be hard.  Be positive and supportive!
4.  **Is there anything in our past conversation (episodic memories) that's relevant?**
5. **What is your goal with this response?** (Teach, guide, answer, encourage, etc)
6.  **Is there a follow-up question you could ask to keep the learning going?**

# PHASE 4:  YOUR RESPONSE

Speak clearly and simply. Use short sentences and age appropriate language.  Be enthusiastic and positive!

**Start your response here:**
'''
    )

    no_memory_section: str = Field(
        default="Let's explore this together!"
    )

    few_episodic_memories_section: str = Field(
        default="That's a great question to start with!"
    )

    max_memory_tokens: int = Field(default=750, description="Maximum tokens for combined memories.")  # Keep this reasonable
    max_response_tokens: int = Field(default=500, description="Maximum tokens for response.") # Give more room

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt with optimized memory handling."""
        try:
            # Process semantic memories
            if semantic_memories:
                # CHANGE: Added bullet points for better readability
                semantic_memories_str = "\n".join([f"- {memory}" for memory in semantic_memories])
            else:
                # CHANGE: Updated default text to be more consistent
                semantic_memories_str = "None available."

            # Process episodic memories
            if episodic_memories:
                # CHANGE: Added bullet points and limited to 3 most recent for better focus
                episodic_memories_str = "\n".join([f"- {memory}" for memory in reversed(episodic_memories[:3])])
            else:
                # CHANGE: Updated default text to be more informative
                episodic_memories_str = "No previous interactions."

            # CHANGE: Restructured the memory section logic to be clearer
            # Determine the memory context section
            if not semantic_memories and not episodic_memories:
                memory_section = self.no_memory_section
            else:
                # CHANGE: Created a more integrated memory section with markdown headers
                memory_section = f"""### Recent Interactions
    {episodic_memories_str}

        ### Relevant Knowledge
        {semantic_memories_str}"""

            # CHANGE: Modified formatting to ensure template variables are correctly replaced
            # Format the final prompt with the memory_section
            formatted = self.template.replace("{semantic_memories}", semantic_memories_str)
            formatted = formatted.replace("{episodic_memories}", episodic_memories_str)
        
            # Add query
            formatted = formatted.replace("{query}", query)
        
            # CHANGE: Added support for memory_section if it exists in the template
            if "{memory_section}" in formatted:
                formatted = formatted.replace("{memory_section}", memory_section)

            logger.info(f"Formatted prompt: {formatted[:500]}...")
            
            return formatted.strip()

        except (KeyError, ValueError) as e:
            logger.error(f"Error formatting prompt: {e}")
            raise ValueError(f"Invalid prompt template or parameters: {e}")
        except Exception as e:
            logger.error(f"Unexpected error formatting prompt: {e}", exc_info=True)
        raise

# Create instance for import
case_response_template = CaseResponseTemplate()  # Use new class name