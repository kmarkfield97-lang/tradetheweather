"""
Temperature trajectory signal.
Looks at the recent hourly temperature trend to estimate whether the market
threshold will be reached before settlement.
"""


def compute(report: dict, threshold: float, market_type: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.

    Args:
        report:      Full weather report from WeatherPipeline.get_full_report()
        threshold:   Market threshold in °F
        market_type: "temp_high" or "temp_low"
    """
    temp_trend = report.get("temp_trend")  # degrees/hr, positive = warming
    forecast = report.get("forecast", {}) or {}

    if temp_trend is None:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no temp trend data"}

    if market_type == "temp_high":
        current_high = forecast.get("high_temp_f")
        if current_high is None:
            return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no high temp forecast"}
        diff = current_high - threshold
        # Positive trend (warming) when already near threshold → increase probability
        if diff > 0:
            # Already above threshold, warming trend reinforces YES
            adj = min(0.08, temp_trend * 0.5) if temp_trend > 0 else 0.0
        else:
            # Below threshold, only warming trend that's strong enough matters
            adj = min(0.05, temp_trend * 0.3) if temp_trend > 0 else max(-0.05, temp_trend * 0.3)
        confidence = min(0.8, abs(temp_trend) * 2.0)
        note = f"temp_trajectory: trend={temp_trend:+.2f}°F/hr, high={current_high}°F vs threshold={threshold}°F"

    elif market_type == "temp_low":
        current_low = forecast.get("low_temp_f")
        if current_low is None:
            return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no low temp forecast"}
        diff = current_low - threshold
        if diff > 0:
            adj = max(-0.05, -temp_trend * 0.3) if temp_trend < 0 else 0.0
        else:
            adj = max(-0.05, temp_trend * 0.2)
        confidence = min(0.7, abs(temp_trend) * 1.5)
        note = f"temp_trajectory: trend={temp_trend:+.2f}°F/hr, low={current_low}°F vs threshold={threshold}°F"

    else:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "not a temp market"}

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
