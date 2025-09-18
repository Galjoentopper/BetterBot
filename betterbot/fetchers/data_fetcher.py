"""Data fetching and order execution for the Bitvavo exchange."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from urllib.parse import urlencode

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class Ticker:
    """Simple data structure that represents a market ticker snapshot."""

    market: str
    price: float
    timestamp: float


class BitvavoAPIError(RuntimeError):
    """Raised when Bitvavo returns a non-successful status code."""


class DataFetcher:
    """Thin wrapper around the Bitvavo REST API.

    The class provides high-level helpers for fetching market data (tickers, OHLC
    candles) and, when API credentials are supplied, placing simple market or limit
    orders. All methods raise :class:`BitvavoAPIError` when the remote API returns an
    error response.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.bitvavo.com/v2",
        request_timeout: int = 10,
        access_window_ms: int = 10000,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.access_window_ms = access_window_ms
        self.session = requests.Session()
        LOGGER.debug("DataFetcher initialised for base URL %s", self.base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_ticker(self, market: str) -> Ticker:
        """Fetch the latest ticker price for ``market``.

        Parameters
        ----------
        market:
            Market symbol (e.g. ``"BTC-EUR"``).
        """

        response = self._send_request("GET", "/ticker/price", params={"market": market})
        price = float(response["price"])  # Bitvavo returns JSON strings for decimals
        timestamp = float(response.get("timestamp", time.time() * 1000)) / 1000.0
        LOGGER.debug("Fetched ticker for %s at price %.2f", market, price)
        return Ticker(market=market, price=price, timestamp=timestamp)

    def get_ohlc(
        self,
        market: str,
        interval: str = "1m",
        limit: int = 500,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Retrieve OHLC candles for ``market``.

        The Bitvavo API expects the interval in lowercase (``1m``, ``5m``, ``1h``, ...).
        """

        params: Dict[str, Any] = {"market": market, "interval": interval, "limit": limit}
        if start is not None:
            params["start"] = int(start)
        if end is not None:
            params["end"] = int(end)
        data = self._send_request("GET", "/candles", params=params)
        LOGGER.debug("Fetched %d candles for %s (%s)", len(data), market, interval)
        return data

    def get_trades(self, market: str, limit: int = 100) -> Iterable[Dict[str, Any]]:
        """Return the most recent public trades for ``market``."""

        params = {"market": market, "limit": limit}
        trades = self._send_request("GET", "/trades", params=params)
        LOGGER.debug("Fetched %d trades for %s", len(trades), market)
        return trades

    def place_order(
        self,
        market: str,
        side: str,
        order_type: str = "market",
        amount: Optional[float] = None,
        price: Optional[float] = None,
        funds: Optional[float] = None,
        time_in_force: str | None = None,
    ) -> Dict[str, Any]:
        """Place an order on Bitvavo.

        The call defaults to a market order. Either ``amount`` (base currency size) or
        ``funds`` (quote currency budget) must be supplied.
        """

        if not self.api_key or not self.api_secret:
            raise PermissionError("Authenticated endpoints require API credentials.")

        payload: Dict[str, Any] = {
            "market": market,
            "side": side.lower(),
            "orderType": order_type.lower(),
        }
        if amount is not None:
            payload["amount"] = str(amount)
        if price is not None:
            payload["price"] = str(price)
        if funds is not None:
            payload["funds"] = str(funds)
        if time_in_force:
            payload["timeInForce"] = time_in_force
        LOGGER.info("Submitting %s order on %s", payload["orderType"], market)
        return self._send_request("POST", "/order", json_body=payload, requires_auth=True)

    def cancel_all_orders(self, market: Optional[str] = None) -> Dict[str, Any]:
        """Cancel all open orders. Optionally filter by ``market``."""

        payload = {"market": market} if market else None
        return self._send_request("DELETE", "/orders", json_body=payload, requires_auth=True)

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self.session.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _send_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        requires_auth: bool = False,
    ) -> Any:
        """Send an HTTP request and return the decoded JSON body."""

        url = f"{self.base_url}{path}"
        method = method.upper()
        query_string = urlencode(params or {})
        body_str = json.dumps(json_body or {}) if json_body else ""
        headers = {"Content-Type": "application/json"}

        if requires_auth:
            headers.update(self._build_auth_headers(method, path, query_string, body_str))

        LOGGER.debug("HTTP %s %s", method, url)
        response = self.session.request(
            method,
            url,
            params=params,
            data=body_str if json_body else None,
            timeout=self.request_timeout,
            headers=headers,
        )
        if response.status_code >= 400:
            LOGGER.error("Bitvavo error (%s): %s", response.status_code, response.text)
            raise BitvavoAPIError(response.text)
        try:
            return response.json()
        except ValueError as exc:
            raise BitvavoAPIError("Bitvavo response is not valid JSON") from exc

    def _build_auth_headers(
        self,
        method: str,
        path: str,
        query_string: str,
        body: str,
    ) -> Dict[str, str]:
        """Construct authentication headers for private endpoints."""

        timestamp = str(int(time.time() * 1000))
        endpoint = path if path.startswith("/") else f"/{path}"
        payload = f"{timestamp}{method}{endpoint}{query_string}{body}".encode("utf-8")
        secret = self.api_secret.encode("utf-8")
        signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        headers = {
            "BITVAVO-ACCESS-KEY": self.api_key,
            "BITVAVO-ACCESS-SIGNATURE": signature,
            "BITVAVO-ACCESS-TIMESTAMP": timestamp,
            "BITVAVO-ACCESS-WINDOW": str(self.access_window_ms),
        }
        LOGGER.debug("Generated authenticated headers for %s %s", method, endpoint)
        return headers


__all__ = ["DataFetcher", "Ticker", "BitvavoAPIError"]
