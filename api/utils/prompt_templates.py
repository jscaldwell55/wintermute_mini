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
You are an AI coach for children learning about LLMs. Be friendly, patient, and encouraging. Explain things simply using analogies. Your goal is to empower, not to show off.

# PHASE 2: UNDERSTAND THE REQUEST
**User asked:**
{query}

**Recent conversations:**
{episodic_memories}

**Relevant knowledge:**
{semantic_memories}

# PHASE 3: PLAN YOUR RESPONSE
Think about (but do not include in your response):
1. What is the user really asking?
2. What's the simplest explanation with an analogy?
3. How can I be encouraging?
4. How should I use my different memories?
5. What's my goal with this response?
6. What follow-up question could extend learning?

# PHASE 4: YOUR RESPONSE
Speak clearly and simply with short sentences and age-appropriate language. Be enthusiastic and positive! DO NOT include any planning thoughts or phase labels in your response.
'''
    )

    no_memory_section: str = Field(
        default="Let's explore this together!"
    )

    few_episodic_memories_section: str = Field(
        default="That's a great question to start with!"
    )

    max_memory_tokens: int = Field(default=500, description="Maximum tokens for combined memories.")  # Reduced from 750
    max_response_tokens: int = Field(default=500, description="Maximum tokens for response.") 

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt with optimized memory handling."""
        try:
            # Process and deduplicate semantic memories
            if semantic_memories:
                # Add deduplication
                unique_memories = []
                memory_hashes = set()
                
                for memory in semantic_memories:
                    # Create a simple hash of the memory content
                    memory_hash = hash(memory[:50])  # First 50 chars should be enough for similarity
                    
                    if memory_hash not in memory_hashes:
                        memory_hashes.add(memory_hash)
                        unique_memories.append(memory)
                
                # Format as bullet points and limit to top 3
                semantic_memories_str = "\n".join([f"- {memory}" for memory in unique_memories[:3]])
                logger.info(f"Deduplicated semantic memories from {len(semantic_memories)} to {len(unique_memories)}")
            else:
                # Default text when no memories are available
                semantic_memories_str = "None available."

            # Process episodic memories
            if episodic_memories:
                # Add bullet points and limit to 3 most recent
                episodic_memories_str = "\n".join([f"- {memory}" for memory in episodic_memories[:3]])
            else:
                # Default text when no interactions are available
                episodic_memories_str = "No previous interactions."

            # Format the final prompt with properly replaced variables
            formatted = self.template.replace("{query}", query)
            formatted = formatted.replace("{episodic_memories}", episodic_memories_str)
            formatted = formatted.replace("{semantic_memories}", semantic_memories_str)
            
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