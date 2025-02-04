from pydantic import BaseModel
from typing import Optional
import os

class CLIConfig(BaseModel):
    api_url: str = "https://wintermute-staging-x-49dd432d3500.herokuapp.com"
    default_window: str = "cli-session"
    timeout: int = 30
    max_retries: int = 3

    @classmethod
    def from_env(cls):
        return cls(
            api_url=os.getenv('WINTERMUTE_API_URL', cls().api_url),
            default_window=os.getenv('WINTERMUTE_WINDOW', cls().default_window),
        )

config = CLIConfig.from_env()