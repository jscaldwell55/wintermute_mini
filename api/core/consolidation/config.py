from dataclasses import dataclass

@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3
    max_age_days: int = 7
    consolidation_interval_hours: int = 24