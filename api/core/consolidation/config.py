# api/core/consolidation/config.py
from dataclasses import dataclass

@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3
    consolidation_prompt: str = "Summarize the following memories, focusing on common themes and insights: {memories_context}"
    context_length: int = 10  # Number of memories to use for context
    # Removed consolidation_interval_hours, as it's not used in the consolidator.
    # If you need it later, add it back, but you'll need to handle it.