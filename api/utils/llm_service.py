# llm_service.py
import logging
from openai import AsyncOpenAI
from api.utils.config import get_settings
from tenacity import retry, stop_after_attempt, wait_exponential, before_log
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import time



logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with OpenAI's LLM APIs."""
    
    MAX_PROMPT_LENGTH = 4096  # Adjust based on model limits
    
    def __init__(self):
        """Initialize the LLM service with configuration and defaults."""
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.llm_model_id
        
        # Default parameters
        self.default_temperature = 0.7
        self.default_max_tokens = 500
        self.default_top_p = 1.0
        self.default_frequency_penalty = 0.0
        self.default_presence_penalty = 0.0
    
    async def validate_prompt(self, prompt: str) -> Optional[str]:
        """
        Validate prompt length and content.
        
        Args:
            prompt: The input prompt to validate
            
        Returns:
            Validated and potentially truncated prompt
            
        Raises:
            ValueError: If prompt is invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Invalid prompt: prompt must be a non-empty string")
            
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            logger.warning(f"Prompt exceeds max length ({len(prompt)} > {self.MAX_PROMPT_LENGTH}). Truncating.")
            return prompt[:self.MAX_PROMPT_LENGTH]
            
        return prompt

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
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        before=before_log(logger, logging.WARNING)
    )
    
    async def generate_gpt_response_async(
        self, 
        prompt: str, 
        temperature: float = None, 
        max_tokens: int = 4096,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        system_message: str = None
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

        Returns:
            Generated response text
            
        Raises:
            ValueError: If prompt validation fails
            Exception: For API or generation errors
        """
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"  # Unique request ID for tracking
        
        try:
            # ✅ Validate and truncate prompt if necessary
            prompt = await self.validate_prompt(prompt)

            if len(prompt) > max_tokens:
                logger.warning(f"Prompt too long ({len(prompt)} > {max_tokens}), truncating...")
                prompt = prompt[:max_tokens]  # Trim to max length
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters with defaults
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.default_temperature,
                "max_tokens": min(max_tokens, 4096),  # ✅ Ensure OpenAI API max limit
                "top_p": top_p or self.default_top_p,
                "frequency_penalty": frequency_penalty or self.default_frequency_penalty,
                "presence_penalty": presence_penalty or self.default_presence_penalty
            }
            
            logger.debug(
                "Sending request to OpenAI",
                extra={
                    "request_id": request_id,
                    "prompt_length": len(prompt),
                    "model": self.model,
                    "params": {k: v for k, v in params.items() if k != "messages"}
                }
            )

            # Make API call
            response = await self.client.chat.completions.create(**params)

            if not response.choices:
                raise ValueError("No response choices returned from API")

            result = response.choices[0].message.content.strip()
            duration = time.time() - start_time
            
            # Log success metrics
            logger.info(
                "LLM request completed successfully",
                extra={
                    "request_id": request_id,
                    "duration": duration,
                    "prompt_length": len(prompt),
                    "response_length": len(result),
                    "model": self.model,
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None,
                }
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"LLM request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "duration": duration,
                    "prompt_length": len(prompt) if prompt else 0,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise

    async def generate_summary(
        self, 
        context: str,
        max_length: int = 200
    ) -> str:
        """
        Generate a summary of memory contexts using GPT model.
    
        Args:
            context: The context to summarize (typically multiple memories combined)
            max_length: Maximum length of the summary in tokens
        
        Returns:
            A summarized version of the input context
            
        Raises:
            Exception: If summary generation fails
        """
        summary_prompt = f"""Please create a concise summary of the following memories, 
capturing the key themes and relationships:

Context to summarize:
{context}

Requirements:
1. Focus on key information and themes
2. Maintain factual accuracy
3. Avoid introducing new information
4. Keep the summary cohesive and well-structured

Summary:"""

        try:
            return await self.generate_gpt_response_async(
                prompt=summary_prompt,
                temperature=0.5,  # Lower temperature for more focused summaries
                max_tokens=max_length,
                frequency_penalty=0.3,  # Slightly increase to encourage diverse language
                presence_penalty=0.2    # Slightly increase to discourage repetition
            )
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the LLM service.
        
        Returns:
            Dict containing health status and metrics
        """
        try:
            # Simple test prompt
            test_response = await self.generate_gpt_response_async(
                "Test health check. Respond with 'OK'.",
                temperature=0.1,
                max_tokens=5
            )
            
            return {
                "status": "healthy",
                "model": self.model,
                "test_response_received": bool(test_response),
                "initialized": bool(self.client)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model
            }