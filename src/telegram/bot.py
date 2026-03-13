"""
Telegram bot interface for TradeTheWeather.
Provides commands for monitoring, controlling, and receiving trade alerts.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
PENDING_FILE = os.path.join(DATA_DIR, "pending_approvals.json")


# -------------------------------------------------------------------------
# Approval store
# -------------------------------------------------------------------------

class ApprovalStore:
    """Stores pending GPT suggestions awaiting user approval."""

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    def _load(self) -> dict:
        try:
            with open(PENDING_FILE) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: dict):
        with open(PENDING_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, suggestion_id: str, suggestion: dict, gpt_run_date: str):
        data = self._load()
        data[suggestion_id] = {
            "suggestion": suggestion,
            "gpt_run_date": gpt_run_date,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        self._save(data)

    def get(self, suggestion_id: str) -> Optional[dict]:
        data = self._load()
        return data.get(suggestion_id)

    def update_status(self, suggestion_id: str, status: str):
        data = self._load()
        if suggestion_id in data:
            data[suggestion_id]["status"] = status
            data[suggestion_id]["resolved_at"] = datetime.now(timezone.utc).isoformat()
            self._save(data)

    def list_pending(self) -> list:
        data = self._load()
        return [
            {"id": sid, **v}
            for sid, v in data.items()
            if v.get("status") == "pending"
        ]


# -------------------------------------------------------------------------
# Bot class
# -------------------------------------------------------------------------

class TradeTheWeatherBot:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.approvals = ApprovalStore()
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
            CommandHandler("suggest", self._cmd_suggest),
            CommandHandler("pending", self._cmd_pending),
            CommandHandler("approve", self._cmd_approve),
            CommandHandler("reject", self._cmd_reject),
            CallbackQueryHandler(self._handle_callback),
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
            "Commands: /status /scan /positions /halt /resume /history /suggest /pending"
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

    async def _cmd_suggest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Running GPT advisor (this may take 30 seconds)...")
        try:
            await self.orchestrator._run_gpt_advisor()
        except Exception as e:
            await update.message.reply_text(f"GPT advisor error: {e}")

    async def _cmd_pending(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        pending = self.approvals.list_pending()
        if not pending:
            await update.message.reply_text("No pending suggestions.")
            return
        lines = ["Pending Approvals:"]
        for item in pending:
            s = item.get("suggestion", {})
            lines.append(
                f"  [{item['id']}] {s.get('title', '?')} "
                f"({s.get('priority', '?')}/{s.get('category', '?')})"
            )
        lines.append("\nUse /approve <id> or /reject <id>")
        await update.message.reply_text("\n".join(lines))

    async def _cmd_approve(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /approve <id>")
            return
        suggestion_id = context.args[0]
        entry = self.approvals.get(suggestion_id)
        if not entry:
            await update.message.reply_text(f"No pending suggestion with id: {suggestion_id}")
            return
        suggestion = entry.get("suggestion", {})
        from src.advisor.auto_implementer import implement_suggestion
        result = implement_suggestion(suggestion)
        self.approvals.update_status(suggestion_id, "approved")
        await update.message.reply_text(
            f"Approved: {suggestion.get('title', suggestion_id)}\nResult: {result}"
        )

    async def _cmd_reject(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /reject <id>")
            return
        suggestion_id = context.args[0]
        entry = self.approvals.get(suggestion_id)
        if not entry:
            await update.message.reply_text(f"No pending suggestion with id: {suggestion_id}")
            return
        self.approvals.update_status(suggestion_id, "rejected")
        await update.message.reply_text(
            f"Rejected: {entry.get('suggestion', {}).get('title', suggestion_id)}"
        )

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button presses for approve/reject."""
        query = update.callback_query
        if not query:
            return
        await query.answer()

        data = query.data or ""
        if data.startswith("approve:"):
            suggestion_id = data[len("approve:"):]
            entry = self.approvals.get(suggestion_id)
            if entry:
                from src.advisor.auto_implementer import implement_suggestion
                result = implement_suggestion(entry.get("suggestion", {}))
                self.approvals.update_status(suggestion_id, "approved")
                await query.edit_message_text(
                    f"Approved: {entry.get('suggestion', {}).get('title', suggestion_id)}\n{result}"
                )
            else:
                await query.edit_message_text("Suggestion not found.")

        elif data.startswith("reject:"):
            suggestion_id = data[len("reject:"):]
            entry = self.approvals.get(suggestion_id)
            if entry:
                self.approvals.update_status(suggestion_id, "rejected")
                await query.edit_message_text(
                    f"Rejected: {entry.get('suggestion', {}).get('title', suggestion_id)}"
                )
            else:
                await query.edit_message_text("Suggestion not found.")

    # -------------------------------------------------------------------------
    # Approval queue
    # -------------------------------------------------------------------------

    async def queue_for_approval(self, suggestion: dict, gpt_run_date: str):
        """Stores a suggestion for approval and sends an inline keyboard message."""
        suggestion_id = suggestion.get("id", f"s_{datetime.now(timezone.utc).timestamp():.0f}")
        self.approvals.add(suggestion_id, suggestion, gpt_run_date)

        priority = suggestion.get("priority", "?")
        category = suggestion.get("category", "?")
        title = suggestion.get("title", "?")
        desc = suggestion.get("description", "")[:200]

        text = (
            f"Approval needed [{priority}/{category}]\n"
            f"Title: {title}\n"
            f"Description: {desc}\n"
            f"ID: {suggestion_id}"
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Approve", callback_data=f"approve:{suggestion_id}"),
                InlineKeyboardButton("Reject", callback_data=f"reject:{suggestion_id}"),
            ]
        ])

        try:
            await self.app.bot.send_message(
                chat_id=USER_ID,
                text=text,
                reply_markup=keyboard,
            )
        except Exception as e:
            logger.error(f"Failed to send approval request: {e}")

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
            from src.tracker.history import DailyHistoryTracker
            tracker = DailyHistoryTracker()
            text = tracker.format_insights()
            await self.send_alert(f"Learning Analysis:\n{text}")
        except Exception as e:
            logger.error(f"Failed to send learning analysis: {e}")

    async def send_gpt_suggestions(self, advice: dict):
        """Formats and sends GPT strategy suggestions."""
        try:
            summary = advice.get("summary", "No summary.")
            suggestions = advice.get("suggestions", [])

            lines = [f"GPT Advisor Report\n{summary}", ""]
            for i, s in enumerate(suggestions, 1):
                lines.append(
                    f"{i}. [{s.get('priority','?')}/{s.get('category','?')}] "
                    f"{s.get('title','?')}"
                )
                lines.append(f"   {s.get('description','')}")
                lines.append(f"   Expected impact: {s.get('expected_impact','?')}")
                lines.append("")

            mdf_rec = advice.get("market_disagreement_filter_recommendation", "")
            if mdf_rec:
                lines.append(
                    f"Market Disagreement Filter: {mdf_rec}\n"
                    f"{advice.get('market_disagreement_filter_reasoning','')}"
                )

            full_text = "\n".join(lines)
            chunks = _split_message(full_text, max_len=4000)
            for chunk in chunks:
                await self.send_alert(chunk)

        except Exception as e:
            logger.error(f"Failed to send GPT suggestions: {e}")

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
