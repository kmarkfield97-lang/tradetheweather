"""
Station bias signal.
Reads the historical forecast_errors.json to estimate per-city, per-market-type
systematic bias in NWS forecasts, then translates that into a probability adjustment.
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
ERRORS_FILE = os.path.join(DATA_DIR, "forecast_errors.json")


def compute(report: dict, threshold: float, market_type: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.
    Positive adjustment = station historically runs cold (YES more likely).
    """
    city = report.get("city", "UNKNOWN").upper()

    try:
        with open(ERRORS_FILE) as f:
            data = json.load(f)
        errors = data.get("forecast_errors", [])
    except Exception:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no bias data"}

    relevant = [
        e["error"] for e in errors
        if e.get("city", "").upper() == city
        and e.get("market_type") == market_type
        and isinstance(e.get("error"), (int, float))
    ]

    if len(relevant) < 3:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": f"insufficient bias data for {city}"}

    recent = relevant[-14:]  # last 2 weeks
    mean_bias = sum(recent) / len(recent)

    # mean_bias = forecast - actual; positive means NWS runs warm → actual temps tend to be lower
    if market_type == "temp_high":
        # For temp_high: warm bias → actual high < forecast → YES (high exceeds threshold) less likely
        adj = -mean_bias / 3.0 * 0.04   # cap at ~0.10 for 7.5°F bias
        adj = max(-0.10, min(0.10, adj))
    elif market_type == "temp_low":
        # For temp_low: YES wins when actual low <= threshold.
        # Warm bias → actual low tends to be lower than forecast → easier to hit threshold → YES more likely
        adj = mean_bias / 3.0 * 0.04   # sign inverted vs temp_high
        adj = max(-0.10, min(0.10, adj))
    else:
        adj = 0.0  # bias not meaningful for rain/snow in same way

    confidence = min(0.8, len(recent) / 14.0 * 0.7)
    note = f"station_bias: {city} {market_type} mean_bias={mean_bias:+.2f}°F over {len(recent)} days"

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
