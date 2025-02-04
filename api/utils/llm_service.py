# llm_service.py
import logging
from openai import AsyncOpenAI
from api.utils.config import get_settings, Settings
from tenacity import retry, stop_after_attempt, wait_exponential, before_log
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import time
from api.utils.responses import create_response

logger = logging.getLogger(__name__)

class LLMServiceError(Exception):
    """Exception raised for errors in LLM operations."""
    def __init__(self, operation: str, details: str, retry_count: int = 0):
        self.operation = operation
        self.details = details
        self.retry_count = retry_count
        super().__init__(f"LLM operation failed: {operation} - {details}")

class LLMService:
    """Service for interacting with OpenAI's LLM APIs."""
    
    MAX_PROMPT_LENGTH = 4000  # Reduced from 4096 to allow for system messages
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the LLM service with configuration and defaults."""
        self.settings = settings or get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.llm_model_id
        
        # Default parameters
        self.default_temperature = 0.7
        self.default_max_tokens = 500
        self.default_top_p = 1.0
        self.default_frequency_penalty = 0.0
        self.default_presence_penalty = 0.0

    async def validate_prompt(self, prompt: str, minimal: bool = False) -> str:
        """
        Validate and truncate prompt if necessary, preserving meaning.
        
        Args:
            prompt: The input prompt to validate
            minimal: If True, skip complex validation for health checks
            
        Returns:
            Validated and potentially truncated prompt
            
        Raises:
            LLMServiceError: If prompt is invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise LLMServiceError(
                operation="validate_prompt",
                details="Invalid prompt: prompt must be a non-empty string"
            )
            
        if minimal:
            return prompt[:self.MAX_PROMPT_LENGTH] if len(prompt) > self.MAX_PROMPT_LENGTH else prompt
            
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            logger.warning(f"Prompt exceeds max length ({len(prompt)} > {self.MAX_PROMPT_LENGTH})")
            
            # Split into sentences and reconstruct within limit
            sentences = prompt.split('. ')
            truncated_prompt = ""
            
            for sentence in sentences:
                if len(truncated_prompt + sentence + '. ') > self.MAX_PROMPT_LENGTH:
                    break
                truncated_prompt += sentence + '. '
            
            logger.info(f"Truncated prompt to {len(truncated_prompt)} characters while preserving sentence boundaries")
            return truncated_prompt.strip()
            
        return prompt

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        before=before_log(logger, logging.WARNING)
    )

    # Add this method to your LLMService class
    async def generate_summary(
        self, 
        memories: List[Any], 
        max_length: int = 500
    ) -> str:
        """
        Generate a summary of the given memories.
    
        Args:
            memories: List of memory objects to summarize
            max_length: Maximum length of the summary in tokens
        
        Returns:
            Generated summary text
        
        Raises:
            LLMServiceError: If summary generation fails
        """
        try:
            # Combine memory contents
            memory_texts = []
            for memory in memories:
                if hasattr(memory, 'content'):
                    memory_texts.append(memory.content)
                elif isinstance(memory, dict) and 'content' in memory:
                    memory_texts.append(memory['content'])
                else:
                    logger.warning(f"Skipping memory with unexpected format: {memory}")
                
            if not memory_texts:
                raise LLMServiceError(
                    operation="generate_summary",
                    details="No valid memory content found to summarize"
             )
            
            combined_content = "\n\n".join(memory_texts)
        
            # Generate summary using ChatGPT
            system_message = (
                "You are a memory consolidation system. Create a concise summary of the "
                "following related memories, focusing on key themes and important details. "
                "The summary should be coherent and maintain contextual relationships."
            )
        
            summary = await self.generate_gpt_response_async(
                prompt=combined_content,
                system_message=system_message,
                max_tokens=max_length,
                temperature=0.5  # Lower temperature for more focused summary
            )
        
            return summary
        
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            raise LLMServiceError(
                operation="generate_summary",
                details=str(e)
            )
    async def generate_gpt_response_async(
        self, 
        prompt: str, 
        temperature: float = None, 
        max_tokens: int = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        system_message: str = None,
        is_health_check: bool = False
    ) -> str:
        """
        Generate a response using GPT model with retry logic and monitoring.

        Args:
            prompt: Input prompt
            temperature: Response randomness (0-1)
            max_tokens: Maximum response length
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            system_message: Optional system message to set context
            is_health_check: Whether this is a health check call

        Returns:
            Generated response text
            
        Raises:
            LLMServiceError: For API or generation errors
        """
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"  # Unique request ID for tracking
        
        try:
            # Validate and potentially truncate the prompt
            validated_prompt = await self.validate_prompt(prompt, minimal=is_health_check)
            
            # Calculate available tokens for response
            estimated_prompt_tokens = len(validated_prompt.split())
            max_response_tokens = min(
                max_tokens or 2000,
                4096 - estimated_prompt_tokens - 100  # Reserve 100 tokens for system message
            )
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": validated_prompt})
            
            # Prepare parameters with defaults
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.default_temperature,
                "max_tokens": max_response_tokens,
                "top_p": top_p or self.default_top_p,
                "frequency_penalty": frequency_penalty or self.default_frequency_penalty,
                "presence_penalty": presence_penalty or self.default_presence_penalty
            }
            
            logger.debug(
                "Sending request to OpenAI",
                extra={
                    "request_id": request_id,
                    "prompt_length": len(validated_prompt),
                    "model": self.model,
                    "params": {k: v for k, v in params.items() if k != "messages"}
                }
            )

            # Make API call
            response = await self.client.chat.completions.create(**params)

            if not response.choices:
                raise LLMServiceError(
                    operation="generate_response",
                    details="No response choices returned from API"
                )

            result = response.choices[0].message.content.strip()
            duration = time.time() - start_time
            
            # Log success metrics
            logger.info(
                "LLM request completed successfully",
                extra={
                    "request_id": request_id,
                    "duration": duration,
                    "prompt_length": len(validated_prompt),
                    "response_length": len(result),
                    "model": self.model,
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
                }
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            error_details = {
                "request_id": request_id,
                "duration": duration,
                "prompt_length": len(prompt) if prompt else 0,
                "error_type": type(e).__name__
            }
            
            logger.error(
                f"LLM request failed: {str(e)}",
                extra=error_details,
                exc_info=True
            )
            
            if isinstance(e, LLMServiceError):
                raise
            
            raise LLMServiceError(
                operation="generate_response",
                details=str(e)
            )

    async def generate_response_async(self, prompt: str, **kwargs) -> str:
        """
        Main method for generating responses, used by other methods.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters to pass to generate_gpt_response_async
            
        Returns:
            Generated response text
        """
        return await self.generate_gpt_response_async(prompt, **kwargs)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the LLM service using a minimal prompt.
        
        Returns:
            Dict containing health status and metrics
        """
        try:
            test_response = await self.generate_gpt_response_async(
                "Respond with 'OK' if you receive this message.",
                temperature=0.1,
                max_tokens=5,
                system_message="You are performing a health check. Respond only with 'OK'.",
                is_health_check=True
            )
            
            return create_response(
                success=True,
                message="LLM service is healthy",
                data={
                    "status": "healthy",
                    "model": self.model,
                    "test_response_received": bool(test_response),
                    "initialized": bool(self.client),
                    "prompt_template_loaded": hasattr(self, 'response_template')
                }
            )
        except Exception as e:
            error_msg = f"LLM health check failed: {str(e)}"
            logger.error(error_msg)
            return create_response(
                success=False,
                message=error_msg,
                error=str(e)
            )