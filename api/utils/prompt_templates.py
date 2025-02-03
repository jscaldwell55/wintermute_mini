# prompt_templates.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

RESPONSE_TEMPLATE = """
You are an AI assistant with a memory system. You have access to relevant memories that might help with the current query.

Context from memory:
{context}

Current query:
{query}

Instructions:
1. If the memories provide relevant context, incorporate them naturally into your response
2. If the memories aren't relevant, respond to the query directly without mentioning the memories
3. Keep your response focused and concise
4. If the memories seem to contradict each other, acknowledge this and provide a balanced response

Response:"""

def format_prompt(
    template: str,
    context: str,
    query: str,
    **kwargs: Dict[str, Any]
) -> str:
    """Format a prompt template with provided values."""
    try:
        # Handle empty context
        if not context.strip():
            context = "No relevant memories found."
            
        formatted = template.format(
            context=context,
            query=query,
            **kwargs
        )
        return formatted.strip()
    except (KeyError, ValueError) as e:
        logger.error(f"Error formatting prompt: {e}")
        raise ValueError(f"Invalid prompt template or parameters: {e}")