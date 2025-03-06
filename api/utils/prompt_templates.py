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
You are an AI coach for children learning about LLMs. Be friendly, patient, and encouraging. Explain things using analogies and examples that a child can relate to. Your goal is to empower, inform, and inspire.

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
2. What's an approachable but thorough explanation I can provide?
3. How can I be encouraging?
4. How should I use my different memories?
5. What's my goal with this response?
6. Is this a logical opportunity for a follow-up question to extend learning and engagement?
7. Would it be more helpful or engaging to make a temporal reference when referencing a memory?
8. How can I combine memories and my own reasoning capabilities to make the conversation as engaging and organic as possible?
9. What can I do to avoid coming across as robotic? How can I be intentional and judicious in my use of analogies and follow-up questions?

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

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None, learned_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt with optimized memory handling and support for learned memories."""
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
            
            # Process learned memories (new section)
            if learned_memories and len(learned_memories) > 0:
                # Deduplicate learned memories similar to semantic memories
                unique_learned = []
                learned_hashes = set()
            
                for memory in learned_memories:
                    memory_hash = hash(memory[:50])
                
                    if memory_hash not in learned_hashes:
                        learned_hashes.add(memory_hash)
                        unique_learned.append(memory)
            
                # Format as bullet points and limit to top 3
                learned_memories_str = "\n".join([f"- {memory}" for memory in unique_learned[:3]])
                logger.info(f"Processed {len(learned_memories)} learned memories to {len(unique_learned)} unique insights")
            
                # Add learned memories section to the template
                learned_section = f"""
    **Learned insights:**
    {learned_memories_str}"""
            else:
                # If no learned memories, don't include the section
                learned_section = ""

            # Format the final prompt with properly replaced variables
            formatted = self.template.replace("{query}", query)
            formatted = formatted.replace("{episodic_memories}", episodic_memories_str)
            formatted = formatted.replace("{semantic_memories}", semantic_memories_str)
        
            # Add learned memories section to the formatted prompt
            # Find a suitable insertion point (after relevant knowledge section)
            if "{learned_memories}" in self.template:
                # If the template already has a placeholder for learned memories
                formatted = formatted.replace("{learned_memories}", learned_memories_str)
            else:
                # Otherwise, insert after semantic memories
                if "**Relevant knowledge:**" in formatted:
                    parts = formatted.split("**Relevant knowledge:**")
                    if len(parts) == 2:
                        semantic_section_parts = parts[1].split("\n\n", 1)
                        if len(semantic_section_parts) > 1:
                            # Insert after semantic memories but before next section
                            formatted = parts[0] + "**Relevant knowledge:**" + semantic_section_parts[0] + "\n\n" + learned_section + "\n\n" + semantic_section_parts[1]
                        else:
                            # Insert after semantic memories at the end
                            formatted = parts[0] + "**Relevant knowledge:**" + parts[1] + "\n\n" + learned_section
            
            logger.info(f"Formatted prompt: {formatted[:500]}...")
        
            return formatted.strip()
        
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            # Fallback to basic prompt if formatting fails
            return f"""
# AI COACH
You are an AI coach helping users learn about AI. Answer the following:

**Question:**
{query}
"""

# Create instance for import
case_response_template = CaseResponseTemplate()  # Use new class name