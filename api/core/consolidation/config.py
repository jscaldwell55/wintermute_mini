# api/core/consolidation/config.py
from dataclasses import dataclass

@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3
    # Removed: consolidation_prompt, context_length