"""
Telegram bot interface for TradeTheWeather.
Provides commands for monitoring, controlling, and receiving trade alerts.
"""

import logging
import os
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

# -------------------------------------------------------------------------
# Bot class
# -------------------------------------------------------------------------

class TradeTheWeatherBot:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.app = Application.builder().token(TOKEN).build()
        self._register_handlers()

    def _register_handlers(self):
        handlers = [
            CommandHandler("start", self._cmd_start),
            CommandHandler("status", self._cmd_status),
            CommandHandler("scan", self._cmd_scan),
            CommandHandler("positions", self._cmd_positions),
            CommandHandler("halt", self._cmd_halt),
            CommandHandler("resume", self._cmd_resume),
            CommandHandler("history", self._cmd_history),
        ]
        for h in handlers:
            self.app.add_handler(h)

    def _is_authorized(self, update: Update) -> bool:
        return update.effective_user and update.effective_user.id == USER_ID

    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        await update.message.reply_text(
            "TradeTheWeather bot is running.\n"
            "Commands: /status /scan /positions /halt /resume /history"
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            summary = self.orchestrator.tracker.format_summary()
            uptime = datetime.now(timezone.utc) - self.orchestrator.started_at
            hours, rem = divmod(int(uptime.total_seconds()), 3600)
            minutes = rem // 60
            text = f"{summary}\n\nUptime: {hours}h {minutes}m"
        except Exception as e:
            text = f"Error getting status: {e}"
        await update.message.reply_text(text)

    async def _cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Starting market scan...")
        try:
            await self.orchestrator.run_scan()
            await update.message.reply_text("Scan complete.")
        except Exception as e:
            await update.message.reply_text(f"Scan error: {e}")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            positions = [
                p for p in self.orchestrator.tracker.state.positions
                if p.status == "open"
            ]
            if not positions:
                await update.message.reply_text("No open positions.")
                return
            lines = ["Open Positions:"]
            for pos in positions:
                lines.append(
                    f"  {pos.ticker} {pos.side.upper()} x{pos.contracts} "
                    f"@ {pos.entry_price}¢ | Cost: ${pos.cost_dollars:.2f}"
                )
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_halt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        self.orchestrator.tracker.state.trading_halted = True
        self.orchestrator.tracker.state.halt_reason = "Manual halt via Telegram"
        self.orchestrator.tracker._save()
        await update.message.reply_text("Trading halted manually.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        self.orchestrator.tracker.state.trading_halted = False
        self.orchestrator.tracker.state.halt_reason = ""
        self.orchestrator.tracker._save()
        await update.message.reply_text("Trading resumed.")

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        try:
            text = self.orchestrator.history.format_insights()
        except Exception as e:
            text = f"Error loading history: {e}"
        await update.message.reply_text(text)

    # -------------------------------------------------------------------------
    # Outgoing alerts
    # -------------------------------------------------------------------------

    async def send_alert(self, text: str):
        """Sends a plain text message to USER_ID."""
        if not USER_ID or not TOKEN:
            return
        try:
            await self.app.bot.send_message(chat_id=USER_ID, text=text)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def send_trade_alert(self, rec):
        """Sends a formatted trade recommendation."""
        try:
            text = (
                f"Trade Placed\n"
                f"Market: {rec.market_title}\n"
                f"Ticker: {rec.ticker}\n"
                f"Side: {rec.side.upper()} @ {rec.market_price}¢\n"
                f"Contracts: {rec.contracts} | Cost: ${rec.cost_dollars:.2f}\n"
                f"Our probability: {rec.our_probability:.1%} | Edge: +{rec.edge:.1%}\n"
                f"Confidence: {rec.confidence.upper()}\n"
                f"Forecast: {rec.forecast_summary}\n"
                f"Reasoning: {rec.reasoning}"
            )
            if rec.alerts:
                text += f"\nAlerts: {', '.join(rec.alerts)}"
            await self.send_alert(text)
        except Exception as e:
            logger.error(f"Failed to send trade alert: {e}")

    async def send_eod_report(self, daily_state):
        """Sends end-of-day P&L report. Splits into chunks if > 4000 chars."""
        try:
            from src.tracker.pnl import MAX_TRADES

            starting = daily_state.starting_balance
            ending = daily_state.current_balance
            pnl = ending - starting
            pnl_pct = (pnl / starting * 100) if starting > 0 else 0.0

            lines = [
                f"End-of-Day Report — {daily_state.date}",
                f"Balance: ${ending:.2f} ({pnl:+.2f} / {pnl_pct:+.1f}%)",
                f"Trades: {daily_state.trades_placed}/{MAX_TRADES}",
                f"Realized P&L: ${daily_state.realized_pnl:+.2f}",
                f"Status: {'HALTED — ' + daily_state.halt_reason if daily_state.trading_halted else 'Active'}",
                "",
                "Positions:",
            ]

            for pos in daily_state.positions:
                pd = pos if isinstance(pos, dict) else vars(pos)
                status = pd.get("status", "?")
                pnl_pos = pd.get("pnl_dollars", 0.0)
                lines.append(
                    f"  {pd.get('ticker', '?')} {pd.get('side', '?').upper()} "
                    f"x{pd.get('contracts', 0)} | {status} | P&L: ${pnl_pos:+.2f}"
                )

            full_text = "\n".join(lines)

            # Split into ≤4000 char chunks
            chunks = _split_message(full_text, max_len=4000)
            for chunk in chunks:
                await self.send_alert(chunk)

        except Exception as e:
            logger.error(f"Failed to send EOD report: {e}")

    async def send_learning_analysis(self, insights, daily_state):
        """Sends learning insights summary."""
        try:
            text = self.orchestrator.history.format_insights()
            await self.send_alert(f"Learning Analysis:\n{text}")
        except Exception as e:
            logger.error(f"Failed to send learning analysis: {e}")

    async def send_morning_briefing(self, context: dict):
        """Sends morning briefing with balance and market count."""
        try:
            balance = context.get("balance", 0.0)
            open_markets = context.get("open_markets", 0)
            today = context.get("date", "")
            text = (
                f"Good morning! TradeTheWeather — {today}\n"
                f"Balance: ${balance:.2f}\n"
                f"Open weather markets: {open_markets}\n"
                f"Ready to trade."
            )
            await self.send_alert(text)
        except Exception as e:
            logger.error(f"Failed to send morning briefing: {e}")

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    def run(self):
        """Start the Telegram bot polling loop."""
        logger.info("Starting Telegram bot polling...")
        self.app.run_polling(allowed_updates=["message", "callback_query"], drop_pending_updates=True)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _split_message(text: str, max_len: int = 4000) -> list:
    """Splits a long message into chunks of at most max_len characters."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline boundary
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
