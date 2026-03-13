"""
Orderbook microstructure signal.
Examines the shape of the Kalshi orderbook to detect imbalances that may indicate
informed trading (i.e., someone knows the weather outcome better than the market).
"""


def compute(liquidity: dict, our_prob: float, side: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.

    Args:
        liquidity:  Output of KalshiClient.get_liquidity()
        our_prob:   Our estimated probability (0-1)
        side:       "yes" or "no"
    """
    yes_volume = liquidity.get("yes_volume", 0)
    no_volume = liquidity.get("no_volume", 0)
    total_volume = yes_volume + no_volume

    if total_volume == 0:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "empty orderbook"}

    # Imbalance ratio: >0 means more YES volume (smart money bullish)
    yes_frac = yes_volume / total_volume
    no_frac = no_volume / total_volume
    imbalance = yes_frac - 0.5   # -0.5 to +0.5

    # Only act on significant imbalances
    if abs(imbalance) < 0.15:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "balanced orderbook"}

    # If imbalance agrees with our direction, slight confirmation
    # If imbalance disagrees, slight reduction
    if side == "yes":
        adj = imbalance * 0.06   # max ±0.03
    else:
        adj = -imbalance * 0.06

    adj = max(-0.05, min(0.05, adj))
    confidence = min(0.6, abs(imbalance) * 1.5)

    direction = "heavy YES" if imbalance > 0 else "heavy NO"
    note = f"orderbook: {direction} ({yes_volume}y/{no_volume}n), imbalance={imbalance:+.2f}"

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
