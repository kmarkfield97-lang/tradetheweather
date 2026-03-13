"""
Signal aggregator. Combines outputs from all signal modules into a single
AggregatedSignal that the analysis engine can apply to base probability estimates.
"""

from typing import List


def aggregate(signals: List[dict]) -> "AggregatedSignal":  # noqa: F821
    """
    Aggregates a list of signal dicts into an AggregatedSignal.

    Each signal dict should have:
        prob_adjustment: float  — signed probability nudge
        confidence: float       — 0-1 weight for this signal
        note: str               — human-readable label
    """
    from src.signals import AggregatedSignal

    if not signals:
        return AggregatedSignal(
            prob_adjustment=0.0,
            signal_agreement=0.0,
            active_signals=0,
            model_uncertainty=0.5,
            notes=[],
        )

    active = [s for s in signals if abs(s.get("prob_adjustment", 0.0)) > 0.001]

    if not active:
        return AggregatedSignal(
            prob_adjustment=0.0,
            signal_agreement=0.5,
            active_signals=0,
            model_uncertainty=0.3,
            notes=[],
        )

    # Weighted average probability adjustment
    total_weight = sum(s.get("confidence", 0.5) for s in active)
    if total_weight == 0:
        total_weight = len(active)

    weighted_adj = sum(
        s.get("prob_adjustment", 0.0) * s.get("confidence", 0.5)
        for s in active
    ) / total_weight

    # Agreement: fraction of signals pointing same direction as net adjustment
    if weighted_adj >= 0:
        agreeing = [s for s in active if s.get("prob_adjustment", 0.0) >= 0]
    else:
        agreeing = [s for s in active if s.get("prob_adjustment", 0.0) < 0]
    agreement = len(agreeing) / len(active) if active else 0.5

    # Uncertainty: inversely proportional to agreement and number of signals
    uncertainty = max(0.1, 1.0 - agreement) * (1.0 / min(len(active), 5))

    notes = [s.get("note", "") for s in active if s.get("note")]

    return AggregatedSignal(
        prob_adjustment=round(weighted_adj, 4),
        signal_agreement=round(agreement, 3),
        active_signals=len(active),
        model_uncertainty=round(uncertainty, 3),
        notes=notes,
    )
