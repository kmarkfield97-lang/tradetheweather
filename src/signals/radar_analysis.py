"""
Radar analysis signal.
Stub implementation — in production this would analyze NEXRAD radar composites
to detect incoming precipitation cells. Currently returns neutral signal based
on NWS alert data as a proxy.
"""


def compute(report: dict, market_type: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.
    Currently uses NWS alerts as a proxy for radar-detectable weather.
    """
    if market_type not in ("rain", "snow"):
        return {
            "prob_adjustment": 0.0,
            "confidence": 0.0,
            "note": "radar_analysis: only meaningful for precip markets",
        }

    alerts = report.get("alerts", [])
    severe_precip_keywords = [
        "flash flood", "tornado", "severe thunderstorm",
        "winter storm", "blizzard", "ice storm",
    ]
    has_severe = any(
        any(kw in a.get("event", "").lower() for kw in severe_precip_keywords)
        for a in alerts
    )

    if has_severe:
        return {
            "prob_adjustment": 0.08,
            "confidence": 0.65,
            "note": "radar_analysis: severe weather alert suggests active precipitation",
        }

    has_precip_alert = any(
        any(kw in a.get("event", "").lower() for kw in ["rain", "flood", "snow", "precipitation"])
        for a in alerts
    )
    if has_precip_alert:
        return {
            "prob_adjustment": 0.04,
            "confidence": 0.45,
            "note": "radar_analysis: precipitation alert in effect",
        }

    return {
        "prob_adjustment": 0.0,
        "confidence": 0.0,
        "note": "radar_analysis: no radar-relevant alerts (NEXRAD integration pending)",
    }
