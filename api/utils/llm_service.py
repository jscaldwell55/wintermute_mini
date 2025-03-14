# api/utils/llm_service.py
import logging
logging.basicConfig(level=logging.INFO)
from openai import AsyncOpenAI
from api.utils.config import get_settings, Settings
from tenacity import retry, stop_after_attempt, wait_exponential, before_log
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import time
import random
from api.utils.responses import create_response
from api.utils.response_cache import ResponseCache
import hashlib

logger = logging.getLogger(__name__)

class LLMServiceError(Exception):
    """Exception raised for errors in LLM operations."""
    def __init__(self, operation: str, details: str, retry_count: int = 0):
        self.operation = operation
        self.details = details
        self.retry_count = retry_count
        super().__init__(f"LLM operation failed: {operation} - {details}")

class DummyCache:
    """A dummy cache that always misses and does nothing when setting values."""
    async def get(self, *args, **kwargs):
        return None
        
    async def set(self, *args, **kwargs):
        pass
        
    async def clear(self):
        return 0

class LLMService:
    """Service for interacting with OpenAI's LLM APIs."""

    MAX_PROMPT_LENGTH = 4000  # Reduced from 4096 to allow for system messages
    MAX_TEMPERATURE = 2.0

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the LLM service with configuration and defaults."""
        self.settings = settings or get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.llm_model_id

        # Default parameters
        self.default_temperature = round(random.uniform(1.3, 1.6), 2) 
        self.default_max_tokens = 700
        self.default_top_p = 0.8
        self.default_frequency_penalty = 0.0
        self.default_presence_penalty = 0.0
        self.response_cache = DummyCache()

    class DummyCache:
        """A dummy cache that always misses and does nothing when setting values."""
        async def get(self, *args, **kwargs):
            return None
            
        async def set(self, *args, **kwargs):
            pass
            
        async def clear(self):
            return 0
        


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
    async def generate_summary(
        self,
        text: str,
        max_length: int = 500
    ) -> str: # Corrected argument
        """
        Generate a summary of the given text.

        Args:
            text: The combined text of memories to summarize.
            max_length: Maximum length of the summary in tokens

        Returns:
            Generated summary text

        Raises:
            LLMServiceError: If summary generation fails
        """
        try:
            # Use generate_gpt_response_async for summary
            summary = await self.generate_gpt_response_async(
                prompt=text,
                system_message="Summarize the following text concisely:", # Simplified system message
                max_tokens=max_length,
                temperature=1.5  # Lower temperature for more focused summary
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
        is_health_check: bool = False,
        model_override: str = None,
        use_cache: bool = False,
        window_id: str = None
    ) -> str:
        use_cache = False
        logger.info(f"LLMService.generate_gpt_response_async called with prompt: '{prompt[:500]}...' (truncated), temperature: {temperature}, max_tokens: {max_tokens}, top_p: {top_p}, frequency_penalty: {frequency_penalty}, presence_penalty: {presence_penalty}, system_message: '{system_message[:500] if system_message else None}' (truncated), is_health_check: {is_health_check}")

        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"  # Unique request ID
        model = model_override or self.model  # Use override if provided

        # Skip cache for health checks
        if is_health_check:
            use_cache = False

        # Create a cache key that incorporates the parameters
        if use_cache:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            temp_str = f"{temperature or self.default_temperature:.2f}"
            max_tok_str = f"{max_tokens or self.default_max_tokens}"
            model_str = model.replace("-", "_")
            cache_query = f"{prompt_hash}:temp={temp_str}:max={max_tok_str}:model={model_str}"
            
            # Check cache first if enabled
            cached_result = await self.response_cache.get(cache_query, window_id)
            if cached_result:
                response_text, similarity = cached_result
                cache_duration = time.time() - start_time
                logger.info(f"Using cached response with similarity {similarity:.3f} (retrieved in {cache_duration:.3f}s)")
                return response_text

        try:
            validated_prompt = await self.validate_prompt(prompt, minimal=is_health_check)
            estimated_prompt_tokens = len(validated_prompt.split())

            # Use provided max_tokens, or the default if none provided.
            max_response_tokens = min(
                max_tokens or self.default_max_tokens,
                4096 - estimated_prompt_tokens - 100
            )
            max_response_tokens = max(0, max_response_tokens)  # Ensure non-negative
            logger.info(f"Request {request_id}: prompt_length={len(validated_prompt)} chars, estimated_tokens={estimated_prompt_tokens}, max_response_tokens={max_response_tokens}")

            # HERE IS WHERE YOU'D ADD THE MODEL TYPE CHECK
            if "instruct" in model:
                # For instruct models, we need to format differently - no messages format
                if system_message:
                    # Include system message as part of the prompt for instruct models
                    full_prompt = f"{system_message}\n\n{validated_prompt}"
                else:
                    full_prompt = validated_prompt
                    
                response = await self.client.completions.create(
                    model=model,
                    prompt=full_prompt,
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_response_tokens,
                    top_p=top_p or self.default_top_p,
                    frequency_penalty=frequency_penalty or self.default_frequency_penalty,
                    presence_penalty=presence_penalty or self.default_presence_penalty
                )
                result = response.choices[0].text.strip()
            else:
                # For chat models, use the existing messages approach
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": validated_prompt})

                params = {
                    "model": model,
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
                        "model": model,
                        "params": {k: v for k, v in params.items() if k != "messages"}
                    }
                )

                response = await self.client.chat.completions.create(**params)
                result = response.choices[0].message.content.strip()

            duration = time.time() - start_time
            logger.info(
                "LLM request completed successfully",
                extra={
                    "request_id": request_id,
                    "duration": duration,
                    "prompt_length": len(validated_prompt),
                    "response_length": len(result),
                    "model": model,
                    "total_tokens": response.usage.total_tokens if response.usage else None,
                    "response_time_ms": int(duration * 1000)
                }
            )

            # Cache the result if caching is enabled
            if use_cache:
                await self.response_cache.set(
                    query=cache_query,
                    response=result,
                    window_id=window_id,
                    metadata={
                        "temperature": temperature or self.default_temperature,
                        "max_tokens": max_tokens or self.default_max_tokens,
                        "model": model,
                        "token_count": response.usage.total_tokens if response.usage else None,
                        "generation_time": duration
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
            ) from e
        

    # Keep generate_response_async for backward compatibility
    async def generate_response_async(self, prompt: str, max_tokens: int = None, **kwargs) -> str:
        return await self.generate_gpt_response_async(prompt, max_tokens=max_tokens, **kwargs)

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
        
    async def clear_cache(self) -> int:
        """
        Clear all entries from the response cache.
        
        Returns:
            Number of entries cleared
        """
        if hasattr(self.response_cache, 'clear'):
            return await self.response_cache.clear()
        else:
            logger.warning("Response cache doesn't have a clear method")
            return 0