"""Shadow backtesting utilities for BetterBot."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from betterbot.fetchers.data_fetcher import DataFetcher
from betterbot.features.feature_engineering import FeatureEngineer
from betterbot.ppo.env import TradingEnv
from betterbot.utils.config import load_config
from betterbot.utils.registry import ModelRegistry

LOGGER = logging.getLogger(__name__)


def _load_recent_candles(fetcher: DataFetcher, market: str, interval: str, hours: int) -> pd.DataFrame:
    limit = max(hours, 100)
    candles = fetcher.get_ohlc(market, interval=interval, limit=limit)
    frame = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
    frame.set_index("timestamp", inplace=True)
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    frame = frame[frame.index >= cutoff]
    return frame.astype(float)


def run_shadow_backtest(lookback_days: int = 7, config_path: str | None = None) -> Dict[str, float]:
    cfg = load_config(config_path)
    registry = ModelRegistry(Path(cfg["paths"]["registry_root"]))
    bundle = registry.load()
    metadata = bundle["metadata"]
    lightgbm = bundle["lightgbm"]
    gru = bundle["gru"]
    ppo = bundle["ppo"]

    fetcher = DataFetcher(
        api_key=cfg.get("bitvavo.api_key", ""),
        api_secret=cfg.get("bitvavo.api_secret", ""),
        base_url=cfg.get("bitvavo.base_url", "https://api.bitvavo.com/v2"),
        request_timeout=int(cfg.get("bitvavo.request_timeout", 10)),
    )

    market = cfg.get("bitvavo.market", "BTC-EUR")
    interval = cfg.get("bitvavo.interval", "1h")
    candles = _load_recent_candles(fetcher, market, interval, lookback_days * 24)

    engineer = FeatureEngineer()
    features = engineer.compute(candles)
    feature_cols = metadata['data']['feature_columns']
    env_features = metadata['data'].get('env_features', feature_cols)
    seq_len = metadata['models']['gru']['sequence_length']

    X = features[feature_cols].to_numpy(dtype=np.float32)
    if len(X) <= seq_len:
        raise ValueError('Not enough data for GRU evaluation.')

    lightgbm_proba = lightgbm.predict_proba(X)

    sequences = []
    for idx in range(seq_len, len(X)):
        sequences.append(X[idx - seq_len:idx])
    gru_preds = gru.predict(np.stack(sequences))
    padded = np.concatenate([np.repeat(gru_preds[0], seq_len), gru_preds])

    features = features.copy()
    features['lgb_proba'] = lightgbm_proba
    features['gru_signal'] = padded[: len(features)]

    env_data = features[['close'] + env_features]
    env = TradingEnv(
        env_data,
        feature_columns=env_features,
        price_column="close",
        initial_cash=float(cfg.get("trading.initial_cash", 1000.0)),
        trading_cost=0.001,
    )

    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    info = {"portfolio_value": env.initial_cash}
    while not done:
        action = ppo.predict(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    start_price = env_data["close"].iloc[0]
    end_price = env_data["close"].iloc[-1]
    buy_and_hold_return = (end_price - start_price) / start_price
    portfolio_return = (info["portfolio_value"] - env.initial_cash) / env.initial_cash

    metrics = {
        "total_reward": total_reward,
        "ppo_return": portfolio_return,
        "buy_and_hold": buy_and_hold_return,
    }
    LOGGER.info("Shadow backtest completed: %s", metrics)
    return metrics


__all__ = ["run_shadow_backtest"]
