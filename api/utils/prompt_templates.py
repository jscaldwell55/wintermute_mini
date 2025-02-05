import logging
logging.basicConfig(level=logging.INFO)
from typing import Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ResponseTemplate(BaseModel):
    """Template for AI responses with context"""
    template: str = """
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

    def format(self, context: str = "", query: str = "", minimal: bool = False, **kwargs: Dict[str, Any]) -> str:
        """
        Format template with provided values.
        
        Args:
            context: Memory context to include
            query: Current user query
            minimal: If True, return just the query without the full template
            **kwargs: Additional template parameters
        """
        try:
            # For health checks and other minimal prompts
            if minimal:
                return query.strip()
                
            if not context.strip():
                context = "No relevant memories found."
                
            formatted = self.template.format(
                context=context,
                query=query,
                **kwargs
            )
            return formatted.strip()
        except (KeyError, ValueError) as e:
            logger.error(f"Error formatting prompt: {e}")
            raise ValueError(f"Invalid prompt template or parameters: {e}")
        except Exception as e:
            logger.error(f"Unexpected error formatting prompt: {e}")
            raise

# Create instance for import
response_template = ResponseTemplate()

def format_prompt(
    template: str,
    context: str = "",
    query: str = "",
    minimal: bool = False,
    **kwargs: Dict[str, Any]
) -> str:
    """Format a prompt template with provided values."""
    try:
        if minimal:
            return query.strip()
            
        if not context.strip():
            context = "No relevant memories found."
            
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