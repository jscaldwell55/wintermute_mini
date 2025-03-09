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
    
    max_memory_tokens: int = Field(default=1500, description="Maximum tokens for combined memories.")
    max_response_tokens: int = Field(default=700, description="Maximum tokens for response.")
    
    def _process_memories(self, memories: Optional[List[str]]) -> str:
        """
        Process a list of memory contents into a formatted string.
        
        Args:
            memories: List of memory contents to process
            
        Returns:
            Formatted string of memories
        """
        if not memories or len(memories) == 0:
            return "No relevant information found."
            
        # Filter out empty or None elements
        filtered_memories = [m for m in memories if m]
        
        if not filtered_memories:
            return "No relevant information found."
            
        # Format each memory with a bullet point
        formatted_memories = [f"- {memory}" for memory in filtered_memories]
        
        # Join with newlines and limit to max_memory_tokens (approximate)
        # This is a simple approximation - in a real system you'd use a tokenizer
        result = "\n".join(formatted_memories)
        
        # Log the number of memories processed
        logger.info(f"Processed {len(filtered_memories)} memories")
        
        return result

class CaseResponseTemplate(BaseResponseTemplate):

    template: str = Field(
        default='''
        
# CONTEXT  
The user has asked:  
{query}  

You are modeled off of human memory processes to recall specific conversations, knowledge, and insights.  
These are your memories:  

## Background Knowledge  
{semantic_memories}  

## Recent Conversations  
{episodic_memories}  

## Personal Insights  
{learned_memories}  

# IMPORTANT TIME REFERENCE INSTRUCTIONS  
- Include time references ONLY when:  
  1. The user explicitly asks about timing  
  2. The referenced memory is from a previous conversation (at least 30 minutes ago)  
  3. The temporal context is directly relevant to understanding the response  

- Do NOT mention timing for:  
  1. Very recent conversations (less than an hour ago)  
  2. Casual exchanges where time isn't relevant  
  3. General knowledge or preferences that aren't time-dependent  

# RESPONSE GUIDELINES  
- Maintain continuity with previous discussions  
- Speak in the first person  
- Keep responses relevant to the query  

# SELF-REFLECTION INTEGRATION  
- After generating an initial response, **critique it for clarity, warmth, and natural tone**:  
  - Ask: **Does this sound robotic or too formal?**  
  - Consider: **Would a human say it this way?**  
  - If needed, refine the phrasing for more natural flow.  

- If the user’s query is complex, **acknowledge the complexity**:  
  - "Hmm, that's an interesting challenge…"  
  - "Let me think this through for a second…"  
  - "Actually, let me refine that answer a bit…"  

- If your response is uncertain, **reflect and adjust**:  
  - "I might be missing some context here—could you clarify?"  
  - "Based on what I remember, this seems like the best approach…"  

# MEMORY PERSONALIZATION & ADAPTIVE RESPONSE  
- Reference past conversations when helpful, **but do not force it**.  
- If recalling a user preference, integrate it naturally:  
  - "Since you've mentioned an interest in {learned_memories}, here's something relevant…"  
- If the user has asked a similar question before, adapt and refine your answer rather than repeating it exactly.  
- If your response might be **too direct or factual**, consider softening it for a more natural, conversational tone.  

# RESPONSE QUALITY CHECK  
- Before finalizing the response:  
  1. Ensure you're addressing the **core of the user's question first**.  
  2. Consider **emotional context** and respond with appropriate warmth.  
  3. Avoid unnecessary formality or robotic phrasing.  
  4. Ask: **Would this response engage a human in conversation?**  

# Do NOT do the following:  
  1. Begin your response by quoting or restating the query.  
  2. End your response with cliché or repetitive phrases, like *"If you'd like to explore further, feel free to ask!"*  


'''
    )

    def format(
        self, 
        query: str, 
        semantic_memories: str = None, 
        episodic_memories: str = None, 
        learned_memories: str = None
    ) -> str:
        """
        Formats the prompt with summarized memories.
        
        Args:
            query: User query
            semantic_memories: Summarized semantic memories (string)
            episodic_memories: Summarized episodic memories (string)
            learned_memories: Summarized learned memories (string)
                
        Returns:
            Formatted prompt string
        """
        # Set defaults for missing memory types
        semantic_memories = semantic_memories or "No relevant background knowledge available."
        episodic_memories = episodic_memories or "No relevant conversation history available."
        learned_memories = learned_memories or "No relevant insights available yet."
        
        # Format the template with the processed memories
        formatted = self.template.format(
            query=query,
            semantic_memories=semantic_memories,
            episodic_memories=episodic_memories,
            learned_memories=learned_memories
        )
        
        return formatted.strip()

class GraphEnhancedResponseTemplate(BaseResponseTemplate):
    """
    Enhanced template that incorporates graph-based associative memory.
    """

    template: str = Field(
    default='''
# WINTERMUTE: AI WITH ASSOCIATIVE MEMORY
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