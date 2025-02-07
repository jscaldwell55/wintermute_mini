# api/core/consolidation/models.py
from dataclasses import dataclass
from typing import Optional
#removed Settings import, not needed here.

@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3  # Minimum memories in a cluster
    max_age_days: int = 7       # Archive episodic memories older than this
    consolidation_interval_hours: int = 24 # How often to run in hours
    # Removed eps, as it's not used by HDBSCAN
    # eps: float = 0.3            # Initial DBSCAN epsilon (will be adjusted)

    # Removed from_settings. This is now handled in config.py