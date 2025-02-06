# api/core/consolidation/models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3  # Minimum memories in a cluster
    eps: float = 0.3            # Initial DBSCAN epsilon (will be adjusted)
    max_age_days: int = 7       # Archive episodic memories older than this
    consolidation_interval_hours: int = 24 # How often to run in hours