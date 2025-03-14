import asyncio
import time
import logging

logger = logging.getLogger(__name__)

class TokenBucket:
    def __init__(self, tokens_per_second, max_tokens):
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        
    async def consume(self, tokens=1):
        async with self.lock:
            await self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
            
    async def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.tokens_per_second
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill = now

# Create a global instance for OpenAI calls
# Adjust rates based on your actual API limits
# OpenAI typically allows 3500 RPM for gpt-3.5-turbo which is ~60 RPS
# But we'll be much more conservative
openai_limiter = TokenBucket(tokens_per_second=10, max_tokens=15)