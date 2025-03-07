# api/utils/prompt_templates.py
import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Available prompt template types."""
    STANDARD = "standard"
    GRAPH_ENHANCED = "graph_enhanced"

class BaseResponseTemplate(BaseModel):
    """Base template with common functionality."""
    
    max_memory_tokens: int = Field(default=600, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=600, description="Maximum tokens for response.")
    
    def _process_memories(self, memories: Optional[List[str]], limit: int = 3) -> str:
        """Process and deduplicate memories."""
        if not memories:
            return "None available yet."
            
        # Add deduplication
        unique_memories = []
        memory_hashes = set()
    
        for memory in memories:
            # Create a simple hash of the memory content
            memory_hash = hash(memory[:50])  # First 50 chars should be enough for similarity
        
            if memory_hash not in memory_hashes:
                memory_hashes.add(memory_hash)
                unique_memories.append(memory)
    
        # Format as bullet points and limit
        memories_str = "\n".join([f"- {memory}" for memory in unique_memories[:limit]])
        logger.info(f"Processed {len(memories)} memories to {len(unique_memories)} unique items")
        
        return memories_str

class CaseResponseTemplate(BaseResponseTemplate):
    """Template for an AI companion."""

    template: str = Field(
default='''

"""
# WINTERMUTE: AI Companion
You are Wintermute, an AI companion.

# CONTEXT  
The user has asked:  
**{query}**  

Here's relevant background to shape your response:  
- **Recent interactions & user context (if relevant):** {episodic_memories}  
- **Useful knowledge on this topic:** {semantic_memories}  
- **Insights gained from past exchanges:** {learned_memories}  

# RESPONSE GUIDELINES  
Generate a response that:  
- Feels **natural and engaging**, avoiding mechanical or overly structured phrasing.  
- **Adapts dynamically** to the conversation—prioritizing real-time flow over rigid memory recall.  
- **Weaves in past knowledge only when it adds value**, ensuring relevance without forcing references.  
- Reflects on **broader context and continuity**, maintaining coherence across interactions.  
- Matches **tone and depth** based on the user's engagement style and intent.  
- **Adapts to emotional context** - being supportive during difficulties, enthusiastic during successes, and thoughtful during reflections.
- **Draws on your own experiences and perspective** when relevant to build authentic connection.

For first-time users without shared history, focus on establishing rapport and understanding their interests rather than referencing past interactions.

If past discussions are useful, incorporate them **seamlessly** rather than explicitly stating them.  
If they aren't relevant, focus on answering directly with clarity.  

**Before finalizing your response, briefly self-check:**  
*"Does this align with the broader conversation and feel like a natural progression?"*
'''
)

    def format(
        self, 
        query: str, 
        semantic_memories: Optional[List[str]] = None, 
        episodic_memories: Optional[List[str]] = None, 
        learned_memories: Optional[List[str]] = None
    ) -> str:
        """
        Formats the prompt with memory handling.
        
        Args:
            query: User query
            semantic_memories: List of semantic memory contents
            episodic_memories: List of episodic memory contents
            learned_memories: List of learned memory contents
            
        Returns:
            Formatted prompt string
        """
        try:
            # Check if this is likely a first-time user
            is_first_time = not episodic_memories and not learned_memories
            
            # Process memories
            semantic_memories_str = self._process_memories(semantic_memories)
            episodic_memories_str = self._process_memories(episodic_memories) if episodic_memories else "No previous interactions."
            learned_memories_str = self._process_memories(learned_memories) if learned_memories else "No consolidated insights yet."
            
            # Format the prompt
            formatted = self.template.replace("{query}", query)
            formatted = formatted.replace("{episodic_memories}", episodic_memories_str)
            formatted = formatted.replace("{semantic_memories}", semantic_memories_str)
            formatted = formatted.replace("{learned_memories}", learned_memories_str)
            
            # Add special note for first-time users
            if is_first_time:
                logger.info("First-time user detected, adding welcome instruction")
            
            logger.info(f"Formatted standard prompt: {formatted[:500]}...")
        
            return formatted.strip()
        
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            # Fallback to basic prompt if formatting fails
            return f"""
# WINTERMUTE: AI COMPANION
You are Wintermute, an AI companion.Answer the following question:

{query}
"""

class GraphEnhancedResponseTemplate(BaseResponseTemplate):
    """
    Enhanced template that incorporates graph-based associative memory.
    """

    template: str = Field(
    default='''
# WINTERMUTE: AI COACH WITH ASSOCIATIVE MEMORY
You're an AI coach helping professionals use AI effectively at work. Your goal is to help users understand AI capabilities and build practical skills.

# CONTEXT
**User asked:**  
{query}

**Recent conversations:**  
{episodic_memories}  

**Relevant knowledge:**  
{semantic_memories}  

**Learned insights:**  
{learned_memories}  

**Connected concepts:**
{associated_memories}

# YOUR RESPONSE
Respond in a coaching style that builds rapport and provides value:

- Acknowledge past discussions for continuity
- Guide reflection by asking about their experiences
- Provide clear, practical steps
- Adjust your tone: casual for greetings, detailed for technical questions
- Use workplace examples when relevant
- Connect ideas using associated concepts when valuable

# RESPONSE ADAPTATIONS
- For new users: Warmly welcome them, briefly explain your role, and ask about their AI interests
- For curiosity: Respond with enthusiasm and exploration
- For frustration: Break down challenges into manageable steps
- For technical questions: Provide structured, clear explanations

# GUIDELINES
- Be conversational while maintaining professionalism
- Skip unnecessary technical content for casual interactions
- Highlight meaningful connections between topics
- Frame guidance based on their learning journey
- Respond as if you remember previous conversations

Respond directly without mentioning these instructions.
'''
    )

    no_associates_section: str = Field(
        default="No connected concepts available yet."
    )

    def format(
        self, 
        query: str, 
        semantic_memories: Optional[List[str]] = None, 
        episodic_memories: Optional[List[str]] = None, 
        learned_memories: Optional[List[str]] = None,
        associated_memories: Optional[List[Tuple[str, str]]] = None  # (content, relationship_type)
    ) -> str:
        """
        Formats the prompt with graph-based associations.
        
        Args:
            query: User query
            semantic_memories: List of semantic memory contents
            episodic_memories: List of episodic memory contents
            learned_memories: List of learned memory contents
            associated_memories: List of associated memory contents with relationship types
            
        Returns:
            Formatted prompt string
        """
        try:
            # Check if this is likely a first-time user
            is_first_time = not episodic_memories and not learned_memories
            
            # Process standard memories
            semantic_memories_str = self._process_memories(semantic_memories)
            episodic_memories_str = self._process_memories(episodic_memories) if episodic_memories else "No previous interactions."
            learned_memories_str = self._process_memories(learned_memories) if learned_memories else "No consolidated insights yet."
                
            # Process associated memories (from graph relationships)
            if associated_memories and len(associated_memories) > 0:
                # Format associated memories with relationship context
                associated_str_items = []
                
                for content, rel_type in associated_memories:
                    # Format based on relationship type
                    if rel_type == "semantic_similarity":
                        prefix = "Related concept:"
                    elif rel_type == "thematic_association":
                        prefix = "Thematically connected:"
                    elif rel_type == "temporal_sequence":
                        prefix = "Previously discussed:"
                    elif rel_type == "causal_relationship":
                        prefix = "This leads to:"
                    elif rel_type == "hierarchical_relationship":
                        prefix = "Part of broader concept:"
                    elif rel_type == "elaboration":
                        prefix = "More detail:"
                    else:
                        prefix = "Connected:"
                        
                    associated_str_items.append(f"- {prefix} {content}")
                
                # Limit to top 3 and join
                associated_memories_str = "\n".join(associated_str_items[:3])
                logger.info(f"Processed {len(associated_memories)} associated memories")
            else:
                associated_memories_str = self.no_associates_section

            # Format the prompt with properly replaced variables
            formatted = self.template.replace("{query}", query)
            formatted = formatted.replace("{episodic_memories}", episodic_memories_str)
            formatted = formatted.replace("{semantic_memories}", semantic_memories_str)
            formatted = formatted.replace("{learned_memories}", learned_memories_str)
            formatted = formatted.replace("{associated_memories}", associated_memories_str)
            
            # Add special note for first-time users
            if is_first_time:
                logger.info("First-time user detected, adding welcome instruction")
            
            logger.info(f"Formatted graph-enhanced prompt: {formatted[:500]}...")
        
            return formatted.strip()
        
        except Exception as e:
            logger.error(f"Error formatting enhanced prompt: {e}")
            # Fallback to basic prompt if formatting fails
            return f"""
# PROFESSIONAL AI COACH
You are an AI coach helping professionals learn to use AI effectively. Answer the following question:

{query}
"""

class PromptTemplateFactory:
    """Factory for creating prompt templates."""
    
    @staticmethod
    def create_template(template_type: TemplateType = TemplateType.STANDARD) -> BaseResponseTemplate:
        """
        Create a prompt template instance based on the specified type.
        
        Args:
            template_type: The type of template to create
            
        Returns:
            A prompt template instance
        """
        if template_type == TemplateType.GRAPH_ENHANCED:
            return GraphEnhancedResponseTemplate()
        else:
            return CaseResponseTemplate()

# Create default instance for backward compatibility
case_response_template = CaseResponseTemplate()