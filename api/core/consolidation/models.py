# api/core/consolidation/models.py
from dataclasses import dataclass
from typing import Optional
from api.utils.config import get_settings, Settings # Import Settings


@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3  # Minimum memories in a cluster
    max_age_days: int = 7       # Archive episodic memories older than this
    consolidation_interval_hours: int = 24 # How often to run in hours
    # Removed eps, as it's not used by HDBSCAN
    # eps: float = 0.3            # Initial DBSCAN epsilon (will be adjusted)

    # Add a method to get settings from the config file.
    @classmethod
    def from_settings(cls, settings: Settings) -> 'ConsolidationConfig':
        """
        Creates a ConsolidationConfig instance from a Settings object,
        overriding default values with those from Settings if present.
        """
        return cls(
            min_cluster_size=settings.min_cluster_size,
            max_age_days=settings.memory_max_age_days,
            consolidation_interval_hours=settings.consolidation_interval_hours,
            # eps=settings.eps #removed, since we are using HDBSCAN now
        )