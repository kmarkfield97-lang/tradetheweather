"""
METAR latency signal.
Checks how stale the most recent METAR observation is. Fresher observations
carry higher confidence and can justify larger probability adjustments.
Also flags if the observed temperature is already near or past the threshold.

Observation age tiers:
  <1h  → base_conf 0.95 (very fresh)
  1–2h → base_conf 0.80
  2–3h → base_conf 0.60
  >3h  → confidence 0.20 (stale — heavily down-weighted in aggregation)
"""

from datetime import datetime, timezone
from typing import Optional


def compute(
    report: dict,
    threshold: float,
    market_type: str,
    stale_station: bool = False,
    local_hour: Optional[int] = None,
) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.

    Args:
        report:        Full weather report from WeatherPipeline.get_full_report()
        threshold:     Market threshold in °F
        market_type:   "temp_high" or "temp_low"
        stale_station: Engine pre-flag when obs age is borderline (1–2h)
        local_hour:    Local hour at the city (0–23), for time-of-day context on temp_high
    """
    observations = report.get("recent_observations", [])
    if not observations:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no recent observations"}

    latest = observations[0]
    latest_time = latest.get("timestamp")
    latest_temp = latest.get("temp_f")

    # ── Observation age → base confidence ────────────────────────────────────
    age_hours: Optional[float] = None
    if latest_time:
        try:
            obs_dt = datetime.fromisoformat(latest_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_hours = (now - obs_dt).total_seconds() / 3600
        except Exception:
            pass

    if age_hours is not None:
        if age_hours < 1.0:
            base_conf = 0.95
        elif age_hours < 2.0:
            base_conf = 0.80
        elif age_hours < 3.0:
            base_conf = 0.60
        else:
            return {
                "prob_adjustment": 0.0,
                "confidence": 0.20,
                "note": f"metar_latency: observation is {age_hours:.1f}h old (stale)",
            }
    else:
        base_conf = 0.70  # unknown age — treat as moderately confident

    # Engine pre-flag: cap if borderline stale
    if stale_station:
        base_conf = min(base_conf, 0.55)

    if latest_temp is None or threshold is None:
        return {
            "prob_adjustment": 0.0,
            "confidence": round(base_conf * 0.5, 3),
            "note": "metar_latency: no current temp reading",
        }

    fresh = age_hours is not None and age_hours < 1.0

    # ── Market-type specific signal ──────────────────────────────────────────
    if market_type == "temp_high":
        if latest_temp >= threshold:
            # Already at or above threshold — very strong YES signal
            # Allow larger adjustment when obs is very fresh
            adj_cap = 0.20 if fresh else 0.12
            adj = min(adj_cap, (latest_temp - threshold) / 5.0 * 0.06)
            confidence = base_conf
            age_tag = f" ({age_hours:.1f}h old)" if age_hours is not None else ""
            note = (
                f"metar_latency: obs {latest_temp}°F >= threshold {threshold}°F{age_tag}"
            )

        elif latest_temp >= threshold - 3:
            adj = 0.04
            confidence = base_conf * 0.85
            note = (
                f"metar_latency: obs {latest_temp}°F within 3°F of threshold {threshold}°F"
            )

        else:
            # Below threshold by >3°F — apply time-of-day context
            gap = threshold - latest_temp
            if local_hour is not None:
                if local_hour < 11:
                    # Morning — gap is expected, don't penalize
                    adj = 0.0
                    confidence = base_conf * 0.75
                    note = (
                        f"metar_latency: obs {latest_temp}°F vs {threshold}°F "
                        f"gap={gap:.1f}°F (early morning, gap expected)"
                    )
                elif local_hour >= 14:
                    # Afternoon — heating window closing, gap is concerning
                    adj = -0.03
                    confidence = base_conf * 0.90
                    note = (
                        f"metar_latency: obs {latest_temp}°F vs {threshold}°F "
                        f"gap={gap:.1f}°F (afternoon, gap concerning)"
                    )
                else:
                    # Mid-morning — neutral
                    adj = 0.0
                    confidence = base_conf * 0.85
                    note = (
                        f"metar_latency: obs {latest_temp}°F vs {threshold}°F "
                        f"gap={gap:.1f}°F"
                    )
            else:
                adj = 0.0
                confidence = base_conf * 0.80
                note = f"metar_latency: obs {latest_temp}°F vs threshold {threshold}°F"

    elif market_type == "temp_low":
        if latest_temp <= threshold:
            adj_cap = 0.15 if fresh else 0.10
            adj = min(adj_cap, (threshold - latest_temp) / 5.0 * 0.05)
            confidence = base_conf
            note = f"metar_latency: obs {latest_temp}°F already <= threshold {threshold}°F"
        elif latest_temp <= threshold + 2:
            adj = 0.02
            confidence = base_conf * 0.85
            note = f"metar_latency: obs {latest_temp}°F within 2°F above threshold {threshold}°F"
        else:
            adj = 0.0
            confidence = base_conf * 0.75
            note = f"metar_latency: obs {latest_temp}°F vs threshold {threshold}°F"

    else:
        adj = 0.0
        confidence = base_conf * 0.40
        note = "metar_latency: not a temp market"

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
