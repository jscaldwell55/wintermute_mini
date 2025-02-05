# consolidation/models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConsolidationConfig:
    min_cluster_size: int = 3
    eps: float = 0.3
    max_age_days: int = 7
    consolidation_interval_hours: int = 24