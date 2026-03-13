"""
Weather models signal.
Stub implementation — in production this would compare GFS and NAM model output
to NWS forecasts. Currently returns neutral signal.
"""


def compute(report: dict, threshold: float, market_type: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.
    Currently a neutral stub pending integration with model output data.
    """
    # Placeholder: check if there are any strong weather alerts that might indicate
    # model disagreement with NWS short-term forecast
    alerts = report.get("alerts", [])
    has_severe = any(
        a.get("severity", "").lower() in ("extreme", "severe")
        for a in alerts
    )

    if has_severe and market_type in ("rain", "snow"):
        return {
            "prob_adjustment": 0.05,
            "confidence": 0.4,
            "note": "weather_models: severe alert suggests models agree on precip",
        }

    return {
        "prob_adjustment": 0.0,
        "confidence": 0.0,
        "note": "weather_models: neutral (model data not integrated)",
    }
