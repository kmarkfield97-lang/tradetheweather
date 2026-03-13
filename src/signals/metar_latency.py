"""
METAR latency signal.
Checks how stale the most recent METAR observation is. If the last observation
was many hours ago, forecast reliability drops. Also flags if observed temperature
is already near or past the threshold.
"""

from datetime import datetime, timezone


def compute(report: dict, threshold: float, market_type: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.
    Stale observations reduce confidence; fresh observations near threshold boost it.
    """
    observations = report.get("recent_observations", [])
    if not observations:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no recent observations"}

    latest = observations[0]
    latest_time = latest.get("timestamp")
    latest_temp = latest.get("temp_f")

    # Check observation age
    age_hours = None
    if latest_time:
        try:
            obs_dt = datetime.fromisoformat(latest_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_hours = (now - obs_dt).total_seconds() / 3600
        except Exception:
            pass

    if age_hours is not None and age_hours > 3:
        # Stale observation — reduce confidence
        return {
            "prob_adjustment": 0.0,
            "confidence": 0.2,
            "note": f"metar_latency: observation is {age_hours:.1f}h old (stale)",
        }

    if latest_temp is None or threshold is None:
        return {"prob_adjustment": 0.0, "confidence": 0.3, "note": "metar_latency: no current temp"}

    if market_type == "temp_high":
        if latest_temp >= threshold:
            # Already at or above threshold — very strong YES signal
            adj = min(0.12, (latest_temp - threshold) / 5.0 * 0.06)
            confidence = 0.85
            note = f"metar_latency: current temp {latest_temp}°F already >= threshold {threshold}°F"
        elif latest_temp >= threshold - 3:
            # Very close — moderate boost
            adj = 0.04
            confidence = 0.7
            note = f"metar_latency: current temp {latest_temp}°F within 3°F of threshold {threshold}°F"
        else:
            adj = 0.0
            confidence = 0.5
            note = f"metar_latency: current temp {latest_temp}°F vs threshold {threshold}°F"
    elif market_type == "temp_low":
        if latest_temp <= threshold:
            adj = min(0.10, (threshold - latest_temp) / 5.0 * 0.05)
            confidence = 0.8
            note = f"metar_latency: current temp {latest_temp}°F already <= threshold {threshold}°F"
        else:
            adj = 0.0
            confidence = 0.5
            note = f"metar_latency: current temp {latest_temp}°F vs threshold {threshold}°F"
    else:
        adj = 0.0
        confidence = 0.3
        note = "metar_latency: not a temp market"

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
