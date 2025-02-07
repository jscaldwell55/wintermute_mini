# api/core/consolidation/models.py
from dataclasses import dataclass
from typing import Optional
from api.utils.config import Settings, get_settings # Import settings

@dataclass
class ConsolidationConfig:
    min_cluster_size: int  # Minimum memories in a cluster
    max_age_days: int      # Archive episodic memories older than this
    consolidation_interval_hours: int = 24 # How often to run in hours, default to 24
    # Removed eps, as it's not used by HDBSCAN

    # Add a method to get settings from the config file.
    @classmethod
    def from_settings(cls, settings: Settings = None) -> 'ConsolidationConfig':
        if settings is None:
           settings = get_settings()
        return cls(
            min_cluster_size=settings.min_cluster_size,
            max_age_days=settings.memory_max_age_days,
            consolidation_interval_hours=settings.consolidation_interval_hours,
        )