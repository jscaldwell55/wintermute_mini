# consolidation/config.py
from dataclasses import dataclass

@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3
    max_episodic_age_days: int = 7  # Only consider episodic memories up to 7 days old
    max_memories_per_consolidation: int = 1000
    consolidation_interval_hours: int = 72