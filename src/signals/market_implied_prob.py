"""
Market implied probability signal.

Computes the market-implied probability and diagnoses significant disagreement
rather than just flagging it. Distinguishes between:
  - Opportunity (our data is fresher or more precise than the market)
  - Warning (market likely knows something we don't)

Diagnosis checklist when disagreement > 0.20:
  1. Forecast recency — stale forecast makes market more likely correct
  2. Station state — obs temp already past threshold in wrong direction
  3. Orderbook skew — bid/ask volume aligns with market price direction
  4. Time remaining — late disagreement is more suspicious
"""

from datetime import datetime, timezone
from typing import Optional


def compute(
    liquidity: dict,
    our_prob: float,
    side: str,
    forecast_report: Optional[dict] = None,
    hours_left: Optional[float] = None,
) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.

    Parameters
    ----------
    liquidity       : dict from KalshiClient.get_liquidity()
    our_prob        : our estimated P(YES)
    side            : "yes" or "no" — the side we intend to trade
    forecast_report : optional forecast sub-dict (for freshness check)
    hours_left      : optional hours until settlement
    """
    best_yes = liquidity.get("best_yes_price")
    best_no = liquidity.get("best_no_price")

    if best_yes is None or best_no is None:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no market prices"}

    market_prob = best_yes / 100.0
    disagreement = abs(our_prob - market_prob)
    market_is_higher = market_prob > our_prob

    # ── Small disagreement — no action ──────────────────────────────────────
    if disagreement <= 0.20:
        return {
            "prob_adjustment": 0.0,
            "confidence": 0.5,
            "note": (
                f"market_implied: normal disagreement model={our_prob:.2f} "
                f"market={market_prob:.2f} (diff={disagreement:.2f})"
            ),
        }

    # ── Significant or extreme disagreement — run diagnosis ─────────────────
    warning_flags = []
    opportunity_flags = []

    # 1. Forecast recency — stale forecast favors trusting the market
    if forecast_report:
        generated_at = forecast_report.get("generated_at")
        if generated_at:
            try:
                fc_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                fc_age_h = (datetime.now(timezone.utc) - fc_dt).total_seconds() / 3600
                if fc_age_h > 3.0:
                    warning_flags.append(f"stale_forecast({fc_age_h:.1f}h)")
                else:
                    opportunity_flags.append(f"fresh_forecast({fc_age_h:.1f}h)")
            except Exception:
                pass

    # 2. Time-remaining suspicion — late disagreement is more concerning
    if hours_left is not None:
        if hours_left < 2.0:
            warning_flags.append(f"final_window({hours_left:.1f}h)")
        elif hours_left < 6.0:
            warning_flags.append(f"late_day({hours_left:.1f}h)")
        else:
            opportunity_flags.append(f"time_ok({hours_left:.1f}h)")

    # 3. Orderbook skew — does the bid/ask volume alignment agree with market price?
    yes_vol = liquidity.get("yes_volume", 0) or 0
    no_vol = liquidity.get("no_volume", 0) or 0
    total_vol = yes_vol + no_vol
    if total_vol > 0:
        yes_frac = yes_vol / total_vol
        # If market_prob is high AND most volume is on YES, the crowd aligns with market
        if market_is_higher and yes_frac > 0.65:
            warning_flags.append(f"ob_skew_vs_model(yes_frac={yes_frac:.0%})")
        elif not market_is_higher and yes_frac < 0.35:
            warning_flags.append(f"ob_skew_vs_model(yes_frac={yes_frac:.0%})")
        else:
            opportunity_flags.append("ob_skew_neutral")

    # ── Score the diagnosis ─────────────────────────────────────────────────
    n_warn = len(warning_flags)
    n_opp = len(opportunity_flags)

    if disagreement > 0.40:
        # Extreme disagreement — base confidence very low regardless
        if n_warn > n_opp:
            # Market looks more likely to be right
            adj = (market_prob - our_prob) * 0.15   # 15% reversion toward market
            confidence = 0.15
            verdict = "EXTREME_WARN"
        else:
            # Potential large opportunity, but be cautious
            adj = 0.0
            confidence = 0.25
            verdict = "EXTREME_OPP"
    else:
        # Large disagreement (0.20–0.40)
        if n_warn > n_opp:
            adj = (market_prob - our_prob) * 0.10   # 10% reversion
            confidence = 0.30
            verdict = "LARGE_WARN"
        else:
            adj = 0.0
            confidence = 0.45
            verdict = "LARGE_OPP"

    warn_str = ",".join(warning_flags) if warning_flags else "none"
    opp_str = ",".join(opportunity_flags) if opportunity_flags else "none"
    note = (
        f"market_implied: {verdict} model={our_prob:.2f} market={market_prob:.2f} "
        f"(diff={disagreement:.2f}) | warn=[{warn_str}] opp=[{opp_str}] "
        f"→ adj={adj:+.3f} conf={confidence:.2f}"
    )

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
