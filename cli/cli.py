import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from datetime import datetime
from .config import config
from .utils import make_request, run_async

console = Console()

@click.group()
def cli():
    """Wintermute CLI - Interact with your AI Memory System"""
    pass

@cli.command()
@click.argument('content')
@click.option('--type', 'memory_type', default="EPISODIC", type=click.Choice(['EPISODIC', 'SEMANTIC']))
@click.option('--window', default=None, help="Window ID for context")
@run_async
async def add(content: str, memory_type: str, window: str):
    """Add a new memory"""
    data = {
        "content": content,
        "memory_type": memory_type,
        "window_id": window or config.default_window,
        "metadata": {
            "source": "cli",
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    with console.status("Adding memory..."):
        result = await make_request("memories", "POST", data)
    
    console.print(Panel(
        f"[green]Memory added successfully![/green]\n\n"
        f"ID: {result['id']}\n"
        f"Type: {result['memory_type']}\n"
        f"Window: {result['window_id']}\n"
        f"Content: {result['content']}"
    ))

@cli.command()
@click.argument('query')
@click.option('--window', default=None, help="Window ID for context")
@click.option('--matches', default=5, help="Number of matches to return")
@run_async
async def ask(query: str, window: str, matches: int):
    """Query memories and get a response"""
    data = {
        "prompt": query,
        "window_id": window or config.default_window,
        "top_k": matches
    }
    
    with console.status("Processing query..."):
        result = await make_request("query", "POST", data)
    
    if result['matches']:
        table = Table(title="Relevant Memories")
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("Content", style="green")
        
        for memory, score in zip(result['matches'], result['similarity_scores']):
            table.add_row(
                f"{score:.3f}",
                memory['content']
            )
        
        console.print(table)
        console.print("\n")
    
    if 'response' in result:
        console.print(Panel(
            Markdown(result['response']),
            title="AI Response",
            border_style="blue"
        ))

@cli.command()
@click.argument('memory_id')
@run_async
async def get(memory_id: str):
    """Retrieve a specific memory by ID"""
    with console.status(f"Retrieving memory {memory_id}..."):
        result = await make_request(f"memories/{memory_id}")
    
    console.print(Panel(
        f"ID: {result['id']}\n"
        f"Type: {result['memory_type']}\n"
        f"Created: {result['created_at']}\n"
        f"Window: {result['window_id']}\n\n"
        f"Content: {result['content']}"
    ))

@cli.command()
@run_async
async def health():
    """Check API health status"""
    with console.status("Checking system health..."):
        result = await make_request("health")
    
    status_color = "green" if result['status'] == "healthy" else "red"
    
    table = Table(title="System Health Status")
    table.add_column("Component", style="blue")
    table.add_column("Status", style=status_color)
    
    table.add_row("System", result['status'])
    table.add_row("Initialized", str(result['initialized']))
    
    if 'services' in result:
        for service, status in result['services'].items():
            table.add_row(
                service,
                status.get('status', 'unknown')
            )
    
    console.print(table)