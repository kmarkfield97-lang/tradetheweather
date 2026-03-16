"""
Temperature trajectory signal.
Looks at the recent hourly temperature trend to estimate whether the market
threshold will be reached before settlement.

Improvements over the original:
- Fixes bug: cooling above threshold now returns a small negative adjustment
  (previously returned 0.0)
- Adds path-to-threshold feasibility: projects current obs forward using
  temp_trend × hours_to_close and compares to threshold
- Weights path projection by time remaining (less weight when far out)
"""

from typing import Optional


def compute(
    report: dict,
    threshold: float,
    market_type: str,
    hours_to_close: Optional[float] = None,
) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.

    Args:
        report:          Full weather report from WeatherPipeline.get_full_report()
        threshold:       Market threshold in °F
        market_type:     "temp_high" or "temp_low"
        hours_to_close:  Hours until market settlement (used for path projection)
    """
    temp_trend = report.get("temp_trend")  # degrees/hr, positive = warming
    forecast = report.get("forecast", {}) or {}

    if temp_trend is None:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no temp trend data"}

    adj = 0.0
    note = ""

    if market_type == "temp_high":
        current_high = forecast.get("high_temp_f")
        if current_high is None:
            return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no high temp forecast"}

        diff = current_high - threshold
        if diff > 0:
            # Already above threshold in forecast
            if temp_trend > 0:
                # Warming while already above — reinforce YES
                adj = min(0.08, temp_trend * 0.5)
            else:
                # Cooling while above — slight penalty (previously returned 0.0 — bug fixed)
                adj = max(-0.04, temp_trend * 0.3)
        else:
            # Below threshold in forecast
            if temp_trend > 0:
                # Warming trend — positive nudge
                adj = min(0.05, temp_trend * 0.3)
            else:
                # Cooling trend when already below threshold — negative adjustment
                adj = max(-0.05, temp_trend * 0.3)

        confidence = min(0.80, abs(temp_trend) * 2.0)
        note = (
            f"temp_trajectory: trend={temp_trend:+.2f}°F/hr, "
            f"forecast_high={current_high}°F vs threshold={threshold}°F "
            f"(diff={diff:+.1f}°F)"
        )

    elif market_type == "temp_low":
        current_low = forecast.get("low_temp_f")
        if current_low is None:
            return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no low temp forecast"}

        diff = current_low - threshold
        if diff > 0:
            # Low forecast above threshold — needs to cool more
            if temp_trend < 0:
                # Cooling toward threshold — positive nudge (symmetric with temp_high)
                adj = min(0.05, -temp_trend * 0.3)
            else:
                # Warming away from threshold — negative adjustment
                adj = max(-0.05, -temp_trend * 0.3)
        else:
            # Already at or below threshold — warming trend is bad for YES
            adj = max(-0.05, temp_trend * 0.2)

        confidence = min(0.70, abs(temp_trend) * 1.5)
        note = (
            f"temp_trajectory: trend={temp_trend:+.2f}°F/hr, "
            f"forecast_low={current_low}°F vs threshold={threshold}°F "
            f"(diff={diff:+.1f}°F)"
        )

    else:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "not a temp market"}

    # ── Path-to-threshold feasibility projection ──────────────────────────────
    # Only meaningful when we have recent observations and hours remaining
    if hours_to_close is not None and hours_to_close > 0:
        obs = report.get("recent_observations", [])
        current_obs_temp = obs[0].get("temp_f") if obs else None

        if current_obs_temp is not None and temp_trend is not None:
            projected = current_obs_temp + temp_trend * hours_to_close

            if market_type == "temp_high":
                path_gap = projected - threshold   # positive = projected to cross
            else:
                path_gap = threshold - projected   # positive = projected to cross

            # Weight projection by recency — more meaningful with < 12h remaining
            weight = max(0.0, 1.0 - hours_to_close / 12.0)

            if path_gap > 2.0:
                path_adj = min(0.06, 0.06 * weight)
            elif path_gap < -2.0:
                path_adj = max(-0.06, -0.06 * weight)
            else:
                path_adj = 0.0

            if path_adj != 0.0:
                adj = max(-0.12, min(0.12, adj + path_adj))
                note += (
                    f" | path_proj={projected:.1f}°F gap={path_gap:+.1f}°F "
                    f"path_adj={path_adj:+.3f} (weight={weight:.2f})"
                )

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
