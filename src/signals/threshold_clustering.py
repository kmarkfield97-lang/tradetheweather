"""
Threshold clustering signal.
Weather outcomes tend to cluster around "round number" thresholds (70°F, 75°F, 80°F).
Markets near these round numbers are harder to predict. Adjusts confidence accordingly.
"""


ROUND_NUMBER_THRESHOLDS = [60, 65, 70, 75, 80, 85, 90, 95, 100, 32, 40, 50]


def compute(threshold: float, market_type: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.
    Near round-number thresholds → increased model uncertainty, reduced confidence.
    """
    if threshold is None:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no threshold"}

    # Find distance to nearest round number
    min_dist = min(abs(threshold - rn) for rn in ROUND_NUMBER_THRESHOLDS)

    if market_type not in ("temp_high", "temp_low"):
        # Not meaningful for rain/snow
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "not a temp market"}

    if min_dist <= 1:
        # Very close to a round number — high clustering risk
        adj = 0.0
        confidence = 0.3  # we're less confident
        note = f"threshold_clustering: {threshold}°F is a round-number threshold (dist={min_dist})"
    elif min_dist <= 3:
        adj = 0.0
        confidence = 0.5
        note = f"threshold_clustering: {threshold}°F near round number (dist={min_dist})"
    else:
        # Far from round numbers → outcomes more predictable
        adj = 0.0
        confidence = 0.7
        note = f"threshold_clustering: {threshold}°F away from round numbers (dist={min_dist})"

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
