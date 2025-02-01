import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

RESPONSE_TEMPLATE = """
You are an AI assistant with a memory system. You have access to relevant memories that might help with the current query.

Context from memory:
{context}

Current query:
{query}

Please provide a helpful response based on the context and query above. If the context is relevant, incorporate it naturally into your response. If the context isn't relevant, simply respond to the query directly.
"""

def format_prompt(
    template: str,
    context: str,
    query: str,
    **kwargs: Dict[str, Any]
) -> str:
    """Format a prompt template with provided values."""
    try:
        formatted = template.format(
            context=context,
            query=query,
            **kwargs
        )
        return formatted.strip()
    except KeyError as e:
        logger.error(f"Missing template variable: {e}")
        raise
    except Exception as e:
        logger.error(f"Error formatting prompt: {e}")
        raise