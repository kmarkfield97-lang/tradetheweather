"""
Daily P&L tracker.
Enforces hard rules:
  - Stop trading at +5% gain
  - Stop trading and exit positions at -5% loss
  - Max 5 trades per day
  - Max 20% of daily budget per position
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
DAILY_FILE = os.path.join(DATA_DIR, "daily_state.json")

PROFIT_TARGET_PCT = 0.05   # +5%
STOP_LOSS_PCT = 0.05       # -5%
MAX_TRADES = 5
MAX_POSITION_PCT = 0.20


@dataclass
class Position:
    ticker: str
    order_id: str
    side: str
    contracts: int
    entry_price: int       # cents
    cost_dollars: float
    status: str            # "open" / "closed" / "expired"
    pnl_dollars: float = 0.0
    exit_price: Optional[int] = None
    placed_at: str = ""


@dataclass
class DailyState:
    date: str
    starting_balance: float
    current_balance: float
    trades_placed: int = 0
    positions: list[Position] = field(default_factory=list)
    trading_halted: bool = False
    halt_reason: str = ""
    goal_met: bool = False
    realized_pnl: float = 0.0


class PnLTracker:
    def __init__(self, kalshi_client=None):
        self.kalshi = kalshi_client
        os.makedirs(DATA_DIR, exist_ok=True)
        self.state = self._load_or_init()

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def _load_or_init(self) -> DailyState:
        today = date.today().isoformat()
        if os.path.exists(DAILY_FILE):
            with open(DAILY_FILE) as f:
                data = json.load(f)
            if data.get("date") == today:
                state = DailyState(**{
                    **data,
                    "positions": [Position(**p) for p in data.get("positions", [])]
                })
                return state
        # New day — fetch starting balance
        starting_balance = self._fetch_balance()
        state = DailyState(
            date=today,
            starting_balance=starting_balance,
            current_balance=starting_balance,
        )
        self._save(state)
        return state

    def _save(self, state: DailyState = None):
        state = state or self.state
        data = asdict(state)
        with open(DAILY_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def _fetch_balance(self) -> float:
        if self.kalshi:
            try:
                return self.kalshi.get_balance()
            except Exception:
                pass
        return 50.0  # default fallback

    # -------------------------------------------------------------------------
    # Rule checks
    # -------------------------------------------------------------------------

    def refresh_balance(self):
        """Sync current balance from Kalshi."""
        self.state.current_balance = self._fetch_balance()
        self._check_rules()
        self._save()

    def _check_rules(self):
        starting = self.state.starting_balance
        current = self.state.current_balance
        pnl_pct = (current - starting) / starting

        if pnl_pct >= PROFIT_TARGET_PCT and not self.state.goal_met:
            self.state.goal_met = True
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily profit target reached (+{pnl_pct * 100:.1f}%)"

        if pnl_pct <= -STOP_LOSS_PCT and not self.state.trading_halted:
            self.state.trading_halted = True
            self.state.halt_reason = f"Stop loss triggered ({pnl_pct * 100:.1f}%)"

    def can_trade(self) -> tuple[bool, str]:
        """Returns (True, '') if trading is allowed, else (False, reason)."""
        if self.state.trading_halted:
            return False, self.state.halt_reason
        if self.state.trades_placed >= MAX_TRADES:
            return False, f"Maximum trades for the day reached ({MAX_TRADES})"
        return True, ""

    def validate_position_size(self, cost_dollars: float) -> tuple[bool, str]:
        """Validates a proposed position size against the 20% rule."""
        max_allowed = self.state.starting_balance * MAX_POSITION_PCT
        if cost_dollars > max_allowed:
            return False, f"Position size ${cost_dollars:.2f} exceeds 20% limit (${max_allowed:.2f})"
        return True, ""

    # -------------------------------------------------------------------------
    # Trade recording
    # -------------------------------------------------------------------------

    def record_trade(self, ticker: str, order_id: str, side: str,
                     contracts: int, entry_price: int, cost_dollars: float):
        """Records a newly placed trade."""
        pos = Position(
            ticker=ticker,
            order_id=order_id,
            side=side,
            contracts=contracts,
            entry_price=entry_price,
            cost_dollars=cost_dollars,
            status="open",
            placed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.state.positions.append(pos)
        self.state.trades_placed += 1
        self._save()

    def record_exit(self, order_id: str, exit_price: int, pnl_dollars: float):
        """Records the exit of a position."""
        for pos in self.state.positions:
            if pos.order_id == order_id and pos.status == "open":
                pos.status = "closed"
                pos.exit_price = exit_price
                pos.pnl_dollars = pnl_dollars
                self.state.realized_pnl += pnl_dollars
                break
        self._save()

    # -------------------------------------------------------------------------
    # Stop loss — exit all positions
    # -------------------------------------------------------------------------

    def trigger_stop_loss(self) -> list[str]:
        """
        Attempts to exit all open positions.
        Returns list of tickers that were exited or attempted.
        """
        self.state.trading_halted = True
        self.state.halt_reason = "Stop loss triggered — exiting all positions"
        self._save()

        exited = []
        if not self.kalshi:
            return exited

        for pos in self.state.positions:
            if pos.status != "open":
                continue
            try:
                # Get current market price to exit at
                liquidity = self.kalshi.get_liquidity(pos.ticker)
                exit_price = liquidity.get("best_no_price") if pos.side == "yes" else liquidity.get("best_yes_price")
                if exit_price:
                    self.kalshi.exit_position(pos.ticker, pos.side, pos.contracts, exit_price)
                    pnl = (exit_price / 100 - pos.entry_price / 100) * pos.contracts
                    self.record_exit(pos.order_id, exit_price, pnl)
                    exited.append(pos.ticker)
            except Exception:
                continue

        self._save()
        return exited

    # -------------------------------------------------------------------------
    # Daily summary
    # -------------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Returns a human-readable daily summary dict."""
        starting = self.state.starting_balance
        current = self.state.current_balance
        pnl = current - starting
        pnl_pct = pnl / starting * 100

        open_positions = [p for p in self.state.positions if p.status == "open"]
        closed_positions = [p for p in self.state.positions if p.status == "closed"]

        return {
            "date": self.state.date,
            "starting_balance": f"${starting:.2f}",
            "current_balance": f"${current:.2f}",
            "pnl": f"${pnl:+.2f}",
            "pnl_pct": f"{pnl_pct:+.1f}%",
            "trades_placed": self.state.trades_placed,
            "trades_remaining": max(0, MAX_TRADES - self.state.trades_placed),
            "open_positions": len(open_positions),
            "realized_pnl": f"${self.state.realized_pnl:+.2f}",
            "goal_met": self.state.goal_met,
            "trading_halted": self.state.trading_halted,
            "halt_reason": self.state.halt_reason,
            "profit_target": f"${starting * (1 + PROFIT_TARGET_PCT):.2f} (+5%)",
            "stop_loss_level": f"${starting * (1 - STOP_LOSS_PCT):.2f} (-5%)",
        }

    def format_summary(self) -> str:
        """Returns a Telegram-ready summary string."""
        s = self.get_summary()
        status = ""
        if s["goal_met"]:
            status = "GOAL MET - Trading paused"
        elif s["trading_halted"]:
            status = f"HALTED: {s['halt_reason']}"
        else:
            status = "Active"

        return (
            f"Daily P&L Summary — {s['date']}\n"
            f"Balance: {s['current_balance']} ({s['pnl']} / {s['pnl_pct']})\n"
            f"Target: {s['profit_target']} | Floor: {s['stop_loss_level']}\n"
            f"Trades: {s['trades_placed']}/{MAX_TRADES} | Open positions: {s['open_positions']}\n"
            f"Realized P&L: {s['realized_pnl']}\n"
            f"Status: {status}"
        )
