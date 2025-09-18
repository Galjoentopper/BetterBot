"""Live (or paper) trading executor for BetterBot."""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from betterbot.fetchers.data_fetcher import DataFetcher
from betterbot.features.feature_engineering import FeatureEngineer
from betterbot.utils.config import load_config
from betterbot.utils.logging_utils import configure_logging
from betterbot.utils.registry import ModelRegistry
from betterbot.utils.telegram_alerts import send_alert

LOGGER = logging.getLogger(__name__)


@dataclass
class Portfolio:
    mode: str
    cash: float
    holdings: float = 0.0
    max_order_quote: float = 500.0
    fee_rate: float = 0.001

    def value(self, price: float) -> float:
        return self.cash + self.holdings * price

    def cash_utilisation(self, initial_cash: float) -> float:
        return max(0.0, min(1.0, 1 - self.cash / initial_cash if initial_cash else 0.0))

    def buy(self, price: float) -> float:
        if price <= 0:
            return 0.0
        budget = min(self.cash, self.max_order_quote)
        if budget <= 0:
            return 0.0
        units = budget / price
        cost = units * price * (1 + self.fee_rate)
        if cost > self.cash:
            return 0.0
        self.cash -= cost
        self.holdings += units
        LOGGER.info("Bought %.6f units at %.2f (cost %.2f)", units, price, cost)
        return units

    def sell_all(self, price: float) -> Tuple[float, float]:
        if self.holdings <= 0:
            return (0.0, 0.0)
        proceeds = self.holdings * price * (1 - self.fee_rate)
        units = self.holdings
        self.cash += proceeds
        self.holdings = 0.0
        LOGGER.info("Sold %.6f units at %.2f (proceeds %.2f)", units, price, proceeds)
        return units, proceeds


def _load_latest_bundle(registry_root: Path) -> Dict:
    registry = ModelRegistry(registry_root)
    return registry.load()


def _ensure_model_fresh(metadata: Dict, expiry_hours: float) -> None:
    created_at = metadata.get("created_at")
    if not created_at:
        return
    created = datetime.fromisoformat(created_at)
    age_hours = (datetime.utcnow() - created).total_seconds() / 3600
    if age_hours > expiry_hours:
        raise RuntimeError(
            f"Model is too old ({age_hours:.1f}h). Refresh required before trading."
        )


def _fetch_candles(fetcher: DataFetcher, market: str, interval: str, limit: int) -> pd.DataFrame:
    candles = fetcher.get_ohlc(market, interval=interval, limit=limit)
    frame = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
    frame.set_index("timestamp", inplace=True)
    return frame.astype(float)


def _build_observation(
    feature_frame: pd.DataFrame,
    metadata: Dict,
    lightgbm,
    gru,
    portfolio: Portfolio,
    initial_cash: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    feature_cols = metadata["data"]["feature_columns"]
    env_features = metadata["data"].get("env_features", feature_cols)
    seq_len = metadata["models"]["gru"]["sequence_length"]

    latest = feature_frame.iloc[-1]
    feature_vector = latest[feature_cols].to_numpy(dtype=np.float32)
    lgb_proba = float(lightgbm.predict_proba(feature_vector.reshape(1, -1))[0])

    sequence = feature_frame[feature_cols].tail(seq_len).to_numpy(dtype=np.float32)
    if len(sequence) < seq_len:
        raise ValueError("Insufficient history for GRU sequence generation.")
    gru_pred = float(gru.predict(sequence.reshape(1, seq_len, -1))[0])

    derived_features = {col: latest.get(col, np.nan) for col in feature_cols}
    derived_features["lgb_proba"] = lgb_proba
    derived_features["gru_signal"] = gru_pred

    env_vector = np.array([derived_features[col] for col in env_features], dtype=np.float32)
    observation = np.concatenate(
        [
            env_vector,
            np.array(
                [portfolio.holdings, portfolio.cash_utilisation(initial_cash)],
                dtype=np.float32,
            ),
        ]
    )
    diagnostics = {
        "lgb_proba": lgb_proba,
        "gru_signal": gru_pred,
    }
    return observation, diagnostics


def _determine_regime(feature_frame: pd.DataFrame, threshold: float) -> str:
    volatility = feature_frame["return_1"].rolling(30).std().iloc[-1]
    if np.isnan(volatility):
        return "unknown"
    return "high_vol" if volatility > threshold else "low_vol"


def _execute_live_order(fetcher: DataFetcher, market: str, side: str, funds: float) -> None:
    try:
        fetcher.place_order(market=market, side=side, funds=funds)
    except Exception as exc:  # pragma: no cover - network side effects
        LOGGER.exception("Failed to place %s order: %s", side, exc)
        send_alert(f"BetterBot live order failure: {exc}")
        raise


def trading_loop(args: argparse.Namespace | None = None) -> None:
    cfg = load_config(args.config if args and args.config else None)
    configure_logging(cfg.get("logging.level", "INFO"), cfg.get("paths.logs_dir"))

    bundle = _load_latest_bundle(Path(cfg["paths"]["registry_root"]))
    metadata = bundle["metadata"]
    _ensure_model_fresh(metadata, cfg.get("trading.model_expiry_hours", 24))

    lightgbm = bundle["lightgbm"]
    gru = bundle["gru"]
    ppo = bundle["ppo"]

    fetcher = DataFetcher(
        api_key=cfg.get("bitvavo.api_key", ""),
        api_secret=cfg.get("bitvavo.api_secret", ""),
        base_url=cfg.get("bitvavo.base_url", "https://api.bitvavo.com/v2"),
        request_timeout=int(cfg.get("bitvavo.request_timeout", 10)),
    )

    mode = cfg.get("trading.mode", "paper")
    initial_cash = float(cfg.get("trading.initial_cash", 500.0))
    portfolio = Portfolio(
        mode=mode,
        cash=initial_cash,
        max_order_quote=float(cfg.get("trading.max_order_quote", 500.0)),
        fee_rate=0.001,
    )

    engineer = FeatureEngineer()
    market = cfg.get("bitvavo.market", "BTC-EUR")
    interval = cfg.get("bitvavo.interval", "1m")
    limit = int(cfg.get("trading.feature_candles", 200))

    refresh = int(cfg.get("trading.refresh_interval", 60))
    volatility_threshold = float(cfg.get("trading.regime_vol_threshold", 0.03))

    last_regime = None

    def run_once() -> None:
        nonlocal last_regime
        candles = _fetch_candles(fetcher, market, interval, limit)
        features = engineer.compute(candles)
        if features.empty:
            raise RuntimeError("Feature frame is empty; not enough data fetched.")

        regime = _determine_regime(features, volatility_threshold)
        if regime != last_regime:
            LOGGER.info("Regime changed: %s -> %s", last_regime, regime)
            last_regime = regime

        price = float(features.iloc[-1]["close"])
        observation, diagnostics = _build_observation(features, metadata, lightgbm, gru, portfolio, initial_cash)
        action = ppo.predict(observation)

        LOGGER.info(
            "Observation diagnostics | price=%.2f | lgb=%.3f | gru=%.3f | action=%s",
            price,
            diagnostics["lgb_proba"],
            diagnostics["gru_signal"],
            action,
        )

        if action == 1:
            if portfolio.holdings == 0:
                if mode == "live":
                    funds = min(portfolio.cash, portfolio.max_order_quote)
                    if funds > 0:
                        _execute_live_order(fetcher, market, "buy", funds=funds)
                units = portfolio.buy(price)
                if units == 0:
                    LOGGER.info("Buy signal skipped (insufficient funds).")
            else:
                LOGGER.debug("Buy action ignored; already in position.")
        elif action == 2:
            if portfolio.holdings > 0:
                if mode == "live":
                    _execute_live_order(fetcher, market, "sell", funds=portfolio.holdings * price)
                portfolio.sell_all(price)
            else:
                LOGGER.debug("Sell action ignored; already flat.")
        else:
            LOGGER.debug("Hold action executed.")

        LOGGER.info(
            "Portfolio | value=%.2f | cash=%.2f | holdings=%.6f",
            portfolio.value(price),
            portfolio.cash,
            portfolio.holdings,
        )

    try:
        run_once()
        if args and args.loop:
            while True:
                time.sleep(refresh)
                run_once()
    except Exception as exc:  # pragma: no cover - operational safeguard
        LOGGER.exception("Trading loop terminated with error: %s", exc)
        send_alert(f"BetterBot trading error: {exc}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BetterBot trading loop")
    parser.add_argument("--config", help="Path to configuration file", default=None)
    parser.add_argument("--loop", action="store_true", help="Continuously run instead of single tick")
    return parser


if __name__ == "__main__":
    trading_loop(build_parser().parse_args())
