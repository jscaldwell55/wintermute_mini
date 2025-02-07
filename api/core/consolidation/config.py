from dataclasses import dataclass

@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3
    consolidation_interval_hours: int = 24