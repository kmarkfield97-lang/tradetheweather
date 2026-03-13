"""
Probability surface signal.
Models the full probability distribution over possible temperature outcomes
using hourly forecast data, rather than just comparing point estimates.
"""

import math


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of normal distribution."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def compute(report: dict, threshold: float, market_type: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.
    Uses the hourly forecast temperature distribution to estimate probability.
    """
    hourly = report.get("hourly", [])
    forecast = report.get("forecast", {}) or {}

    if not hourly or threshold is None:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no hourly data"}

    if market_type not in ("temp_high", "temp_low"):
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "not a temp market"}

    temps = [h["temp_f"] for h in hourly if h.get("temp_f") is not None]
    if len(temps) < 6:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "insufficient hourly temps"}

    if market_type == "temp_high":
        extreme_val = max(temps)
        # Model as: actual high ~ N(forecast_high, sigma=3)
        forecast_val = forecast.get("high_temp_f") or extreme_val
        sigma = 3.0
        surface_prob = 1.0 - _normal_cdf(threshold, forecast_val, sigma)
    else:
        extreme_val = min(temps)
        forecast_val = forecast.get("low_temp_f") or extreme_val
        sigma = 3.0
        # P(low < threshold) = CDF(threshold)
        surface_prob = _normal_cdf(threshold, forecast_val, sigma)

    # The surface probability is our alternative estimate. If it agrees with
    # the base estimate, small positive confidence boost; else neutral.
    base_prob = surface_prob  # use directly as secondary estimate; report delta
    adj = 0.0  # we don't double-count; this is informational
    confidence = 0.5

    note = (
        f"probability_surface: {market_type} surface_prob={surface_prob:.3f} "
        f"(model={forecast_val}°F, sigma={sigma}, threshold={threshold}°F)"
    )

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
        "surface_prob": round(surface_prob, 4),   # extra field for engine use
    }
