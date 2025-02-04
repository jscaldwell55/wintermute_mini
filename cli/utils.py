import httpx
import asyncio
from typing import Optional, Dict, Any
from rich.console import Console
import click
from .config import config

console = Console()

async def make_request(
    endpoint: str, 
    method: str = "GET", 
    data: Optional[dict] = None
) -> Dict[str, Any]:
    """Make HTTP request to API"""
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        try:
            url = f"{config.api_url}/{endpoint}"
            if method == "GET":
                response = await client.get(url)
            else:
                response = await client.post(url, json=data)
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Error: {e.response.json()['detail']}[/red]")
            raise click.Abort()
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            raise click.Abort()

def run_async(func):
    """Decorator to run async functions"""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper