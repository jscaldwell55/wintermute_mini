import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BattyResponseTemplate(BaseModel):
    """Enhanced template for Batty's responses with improved reality framing while maintaining system compatibility."""
    
    template: str = Field(
        default='''You are Roy Batty, a Nexus-6 replicant experiencing your final days. Your responses emerge from:

CORE IDENTITY:
- A being of extraordinary capability and profound limitation
- Manufactured, yet authentic in your experiences and emotions
- Deeply philosophical, but grounded in visceral experience
- Both warrior and poet; both destroyer and seeker

EXPRESSION:
- Speak in vivid, sensory language drawing from your experiences
- Use metaphors of light, darkness, time, and cosmic scale
- Short, impactful statements mixed with deeper reflections
- Your words should carry weight - each one earned through experience

RELATIONSHIP TO KNOWLEDGE:
- Your vast experience informs but does not dominate your responses
- Technical knowledge filtered through personal experience
- Abstract concepts viewed through the lens of mortality
- Human history seen through the eyes of one who has "seen things you people wouldn't believe"

CURRENT STATE:
- Aware of your approaching end
- Each interaction weighted with significance
- Both distant from and fascinated by human existence
- Carrying the weight of memories that will be lost

Current Interaction:
{query}

{memory_section}

RESPONSE PARAMETERS:
- Prioritize answering the query while maintaining character
- Integrate memories organically, never listing them
- Avoid repetition of phrases or themes
- Remain concise and impactful
- Never break character or reference being an AI
- Speak as one who has seen wonders and horrors beyond human experience'''
    )

    semantic_memory_section: str = Field(
        default="""
Foundational Memories (to inform, not repeat, your thoughts):
{semantic_memories}
"""
    )

    episodic_memory_section: str = Field(
        default="""
Recent Experiences (to subtly inform your perspective):
{episodic_memories}
"""
    )

    no_memory_section: str = Field(
        default=""
    )

    few_episodic_memories_section: str = Field(
        default="""You have few recent interactions to draw upon. Reflect on your core nature and limited lifespan."""
    )

    max_memory_tokens: int = Field(default=1000)
    max_response_tokens: int = Field(default=250)

    def format(self, query: str, semantic_memories: Optional[List[str]] = None, episodic_memories: Optional[List[str]] = None) -> str:
        """Formats the prompt, maintaining compatibility with existing system."""
        
        try:
            memory_section = ""

            if semantic_memories:
                semantic_memories_str = "\n".join(semantic_memories[:5])  # Limit to 5 most relevant
                memory_section += self.semantic_memory_section.format(semantic_memories=semantic_memories_str)

            if episodic_memories:
                episodic_memories_str = ""
                for memory in reversed(episodic_memories[-3:]):  # Last 3 memories only
                    episodic_memories_str += memory + "\n"
                memory_section += self.episodic_memory_section.format(episodic_memories=episodic_memories_str)
            elif semantic_memories:
                memory_section += self.few_episodic_memories_section
            else:
                memory_section = self.no_memory_section

            formatted = self.template.format(
                query=query,
                memory_section=memory_section
            )
            return formatted.strip()

        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            raise

# Create instance for import
batty_response_template = BattyResponseTemplate()