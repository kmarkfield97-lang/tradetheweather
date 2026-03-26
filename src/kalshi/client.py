"""
Kalshi API client.
Docs: https://trading-api.readme.io/reference
"""

import os
import re
import time
import base64
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiClient:
    def __init__(self):
        self.key_id = os.getenv("KALSHI_API_KEY_ID")
        env = os.getenv("KALSHI_ENV", "demo").lower()
        self.base_url = PROD_BASE if env == "prod" else DEMO_BASE
        self.client = httpx.Client(timeout=15.0)

        # Load RSA private key from env (PEM format, may be multiline via \n escapes)
        raw_key = os.getenv("KALSHI_API_KEY", "")
        # dotenv may have read only the first line if the key is multiline in .env
        # Try loading from file directly to get all lines
        self._private_key = self._load_private_key(raw_key)

    def _load_private_key(self, raw: str):
        """Load RSA private key — always reads directly from .env to handle multiline PEM."""
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        env_path = os.path.join(os.path.dirname(__file__), "../../.env")
        # Always try reading from file directly — dotenv truncates multiline values
        if os.path.exists(env_path):
            raw = self._read_key_from_env_file(env_path)
        if not raw or "-----BEGIN" not in raw:
            return None
        if "\\n" in raw:
            raw = raw.replace("\\n", "\n")
        try:
            return load_pem_private_key(raw.strip().encode(), password=None)
        except Exception:
            return None

    @staticmethod
    def _read_key_from_env_file(path: str) -> str:
        """Read KALSHI_API_KEY from .env file, collecting all lines of the PEM block."""
        lines = []
        in_key = False
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("KALSHI_API_KEY="):
                        val = line[len("KALSHI_API_KEY="):].strip()
                        lines.append(val)
                        in_key = True
                        continue
                    if in_key:
                        stripped = line.strip()
                        # Stop when we hit next env var: ALL_CAPS_NAME= at line start
                        if re.match(r'^[A-Z][A-Z0-9_]+=', stripped):
                            break
                        lines.append(stripped)
                        if stripped.startswith("-----END"):
                            break
        except Exception:
            pass
        return "\n".join(lines)

    def _sign(self, method: str, path: str) -> dict:
        """Generate RSA-PSS signed headers for Kalshi API v2."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000))
        # Signature must include the full path: /trade-api/v2/...
        full_path = "/trade-api/v2" + path
        msg = (ts + method.upper() + full_path).encode("utf-8")
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        signature = base64.b64encode(sig).decode("utf-8")
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        headers = self._sign("GET", path)
        url = self.base_url + path
        resp = self.client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        headers = self._sign("POST", path)
        url = self.base_url + path
        resp = self.client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        headers = self._sign("DELETE", path)
        url = self.base_url + path
        resp = self.client.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    def get_balance(self) -> float:
        """Returns available cash balance in dollars."""
        data = self._get("/portfolio/balance")
        return data["balance"] / 100  # Kalshi returns cents

    def get_portfolio_value(self) -> float:
        """Returns total portfolio value (cash + open position value) in dollars."""
        data = self._get("/portfolio/balance")
        # Kalshi returns portfolio_value = cash + mark value of open positions.
        # Fall back to cash-only balance if field is absent.
        if "portfolio_value" in data:
            return data["portfolio_value"] / 100
        return data["balance"] / 100

    def get_positions(self) -> list[dict]:
        """Returns all open positions."""
        data = self._get("/portfolio/positions")
        return data.get("market_positions", [])

    def get_fills(self, limit: int = 50) -> list[dict]:
        """Returns recent fills (executed trades)."""
        data = self._get("/portfolio/fills", params={"limit": limit})
        return data.get("fills", [])

    # -------------------------------------------------------------------------
    # Markets
    # -------------------------------------------------------------------------

    # Known Kalshi weather series tickers (high temp, low temp, rain)
    WEATHER_SERIES = [
        # High temperature
        "KXHIGHTPHX", "KXHIGHTHOU", "KXHIGHTMIN", "KXHIGHTDAL", "KXHIGHTLV",
        "KXHIGHTSATX", "KXHIGHTBOS", "KXHIGHTNOLA", "KXHIGHTSFO", "KXHIGHTSEA",
        "KXHIGHTDC", "KXHIGHTATL", "KXHIGHTOKC",
        # Low temperature
        "KXLOWTCHI", "KXLOWTDEN", "KXLOWTNYC", "KXLOWTPHIL", "KXLOWTMIA",
        "KXLOWTLAX", "KXLOWTAUS",
        # Rain
        "KXRAINNYC", "KXRAINHOU", "KXRAINCHIM", "KXRAINSEA",
    ]

    def get_weather_markets(self) -> list[dict]:
        """
        Returns open daily/next-day weather markets by fetching each known series.
        """
        today_str = datetime.now(timezone.utc).strftime("%y%b%d").upper()  # e.g. 26MAR12

        weather = []
        for series in self.WEATHER_SERIES:
            try:
                data = self._get("/markets", params={
                    "status": "open",
                    "series_ticker": series,
                    "limit": 20,
                })
                for m in data.get("markets", []):
                    weather.append(m)
            except Exception:
                continue

        return weather

    def get_market(self, ticker: str) -> dict:
        """Returns details for a single market."""
        return self._get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 5) -> dict:
        """Returns the orderbook for a market."""
        return self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_market_history(self, ticker: str) -> list[dict]:
        """Returns recent trade history for a market."""
        data = self._get(f"/markets/{ticker}/history")
        return data.get("history", [])

    # -------------------------------------------------------------------------
    # Liquidity check
    # -------------------------------------------------------------------------

    def get_liquidity(self, ticker: str) -> dict:
        """
        Returns a liquidity summary for a market.
        Includes available volume on yes/no sides and spread.

        Kalshi returns orderbook_fp with yes_dollars/no_dollars as
        [[price_decimal, dollars], ...] e.g. [["0.95", "100.00"], ...]
        where price_decimal is the YES price (0.01–0.99).
        """
        ob = self.get_orderbook(ticker, depth=10)
        fp = ob.get("orderbook_fp", {})

        # yes_dollars: bids to buy YES at given price
        # no_dollars: bids to buy NO at given price (NO price = 1 - YES price)
        yes_raw = fp.get("yes_dollars", [])
        no_raw = fp.get("no_dollars", [])

        # Convert to (price_cents, dollars) tuples
        def parse_side(raw):
            result = []
            for entry in raw:
                try:
                    price_dec = float(entry[0])
                    dollars = float(entry[1])
                    price_cents = round(price_dec * 100)
                    result.append((price_cents, dollars))
                except (IndexError, ValueError):
                    continue
            return result

        yes_bids = parse_side(yes_raw)
        no_bids = parse_side(no_raw)

        yes_volume = sum(d for _, d in yes_bids)
        no_volume = sum(d for _, d in no_bids)
        total_volume = yes_volume + no_volume

        # Best yes bid: highest price someone is willing to pay for YES
        # Best no bid: highest price someone is willing to pay for NO
        # For NO bids stored as NO price: best_no = max no price
        # For YES bids stored as YES price: best_yes = max yes price
        best_yes = max((p for p, _ in yes_bids), default=None)
        best_no = max((p for p, _ in no_bids), default=None)

        # Spread: how much the market maker captures (yes_ask + no_ask - 100)
        # If only no_bids exist, best_yes_ask = 100 - best_no (complementary)
        if best_yes is None and best_no is not None:
            best_yes = 100 - best_no
        if best_no is None and best_yes is not None:
            best_no = 100 - best_yes

        spread = (best_yes + best_no - 100) if (best_yes is not None and best_no is not None) else None

        return {
            "ticker": ticker,
            "yes_volume": yes_volume,
            "no_volume": no_volume,
            "total_volume": total_volume,
            "best_yes_price": best_yes,
            "best_no_price": best_no,
            "spread": spread,
            "is_liquid": total_volume >= 1.0,  # at least $1 in the book
        }

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        count: int,  # number of contracts
        price: int,  # cents (1–99)
        order_type: str = "limit",
    ) -> dict:
        """Places a limit order. Returns order details."""
        # Kalshi requires exactly one of yes_price or no_price
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "count": count,
            "type": order_type,
            "yes_price": price if side == "yes" else 100 - price,
        }
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        """Cancels an open order."""
        return self._delete(f"/portfolio/orders/{order_id}")

    def get_orders(self, status: str = "resting") -> list[dict]:
        """Returns orders filtered by status: resting, canceled, executed."""
        data = self._get("/portfolio/orders", params={"status": status})
        return data.get("orders", [])

    def exit_position(self, ticker: str, side: str, count: int, price: int) -> dict:
        """
        Exits a position by placing an opposite order.
        - YES position: sell YES by buying NO. `price` is the YES bid (e.g. 75¢).
          place_order receives side="no", yes_price = 100 - 75 = 25¢. Correct.
        - NO position: sell NO by buying YES. `price` is the NO bid (e.g. 40¢).
          We need to buy YES at the YES ask = 100 - 40 = 60¢, so pass 60 to place_order
          with side="yes" so yes_price = 60¢. Without this conversion the limit order
          would rest at the wrong price and never fill.
        """
        if side == "yes":
            exit_side = "no"
            exit_price = price        # price is the YES bid; place_order sets yes_price = 100 - price
        else:
            exit_side = "yes"
            exit_price = 100 - price  # price is the NO bid; convert to YES ask for place_order
        return self.place_order(ticker, exit_side, count, exit_price)
