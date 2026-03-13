"""
Signals package. Exports AggregatedSignal and the aggregator module.
"""

from dataclasses import dataclass, field
from typing import List

from src.signals import aggregator


@dataclass
class AggregatedSignal:
    """Result of aggregating all active signals for a market."""
    prob_adjustment: float = 0.0        # net adjustment to base probability (-1 to +1)
    signal_agreement: float = 0.0       # 0.0 = full disagreement, 1.0 = all agree
    active_signals: int = 0             # number of signals that fired
    model_uncertainty: float = 0.0      # higher = less confident
    notes: List[str] = field(default_factory=list)


__all__ = ["AggregatedSignal", "aggregator"]
