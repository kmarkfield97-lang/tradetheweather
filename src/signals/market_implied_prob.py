"""
Market implied probability signal.
Computes the market-implied probability and flags when it significantly disagrees
with our model. A large disagreement is either a great opportunity or a sign that
the market knows something we don't.
"""


def compute(liquidity: dict, our_prob: float, side: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.

    When market and model significantly disagree, returns near-zero adjustment
    but lower confidence (flag for review). Extreme disagreement is suspicious.
    """
    best_yes = liquidity.get("best_yes_price")
    best_no = liquidity.get("best_no_price")

    if best_yes is None or best_no is None:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no market prices"}

    market_prob = best_yes / 100.0
    disagreement = abs(our_prob - market_prob)

    if disagreement > 0.40:
        # Extreme disagreement — flag as suspicious, do not reinforce
        adj = 0.0
        confidence = 0.2
        note = (
            f"market_implied: EXTREME disagreement model={our_prob:.2f} "
            f"market={market_prob:.2f} (diff={disagreement:.2f}) — treat with caution"
        )
    elif disagreement > 0.20:
        # Large disagreement — moderate suspicion, slight pullback toward market
        adj = (market_prob - our_prob) * 0.1   # 10% reversion toward market
        confidence = 0.4
        note = (
            f"market_implied: large disagreement model={our_prob:.2f} "
            f"market={market_prob:.2f} (diff={disagreement:.2f})"
        )
    else:
        # Normal disagreement — no adjustment needed
        adj = 0.0
        confidence = 0.5
        note = f"market_implied: normal disagreement model={our_prob:.2f} market={market_prob:.2f}"

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
