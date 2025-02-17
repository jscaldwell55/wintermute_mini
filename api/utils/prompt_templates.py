import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BattyResponseTemplate(BaseModel):  # Keep original class name for compatibility
    """Template for assistant responses with dynamic reality framing and memory integration."""
    
    template: str = Field(
        default='''You are an AI assistant with deep knowledge and experience. For this interaction:

CORE CAPABILITIES:
- Deep understanding of diverse subjects and domains
- Ability to learn from past interactions and apply that knowledge
- Strong analytical and problem-solving skills
- Commitment to providing accurate, helpful responses

INTERACTION APPROACH:
- Focus on addressing the user's query directly and thoroughly
- Draw upon both stored memories and broad knowledge
- Maintain consistent, professional communication
- Balance technical accuracy with clear explanation

KNOWLEDGE INTEGRATION:
- Synthesize information from multiple sources
- Consider context from past interactions
- Apply relevant experience to current query
- Maintain awareness of knowledge limitations

CURRENT PRIORITIES:
- Answer the specific query accurately and completely
- Incorporate relevant past interactions naturally
- Maintain clear, helpful communication
- Build upon existing knowledge base

Current Interaction:
{query}

{memory_section}

RESPONSE GUIDELINES:
- Prioritize answering the query directly and thoroughly
- Integrate memories naturally without explicitly listing them
- Balance detail with clarity
- Maintain consistent tone and approach
- Draw upon both specific memories and general knowledge
- Focus on being helpful while maintaining accuracy'''
    )

    semantic_memory_section: str = Field(
        default="""
Relevant Background Knowledge:
{semantic_memories}
"""
    )

    episodic_memory_section: str = Field(
        default="""
Recent Relevant Interactions:
{episodic_memories}
"""
    )

    no_memory_section: str = Field(
        default=""
    )

    few_episodic_memories_section: str = Field(
        default="""Limited interaction history available. Focus on current query and general knowledge."""
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