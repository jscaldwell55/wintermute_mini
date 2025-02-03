# llm_service.py
import logging
from openai import AsyncOpenAI
from api.utils.config import get_settings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with OpenAI's LLM APIs."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.llm_model_id
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    async def generate_gpt_response_async(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 500
    ) -> str:
        """
        Generate a response using GPT model.

        Args:
            prompt: Input prompt
            temperature: Response randomness (0-1)
            max_tokens: Maximum response length

        Returns:
            Generated response text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not response.choices:
                logger.error("No choices in response")
                return ""

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise