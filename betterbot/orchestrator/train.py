"""Training orchestration for BetterBot models."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from betterbot.fetchers.data_fetcher import DataFetcher
from betterbot.features.feature_engineering import FeatureEngineer
from betterbot.forecasters.gru_model import GRUForecaster, TrainingConfig
from betterbot.forecasters.lightgbm_model import LightGBMForecaster
from betterbot.ppo.env import TradingEnv
from betterbot.ppo.policy_wrapper import PPOPolicyWrapper
from betterbot.utils import metadata_validator
from betterbot.utils.config import load_config
from betterbot.utils.logging_utils import configure_logging
from betterbot.utils.registry import ModelRegistry
from betterbot.utils.telegram_alerts import send_alert

LOGGER = logging.getLogger(__name__)


def _load_candles(fetcher: DataFetcher, market: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        candles = fetcher.get_ohlc(market, interval=interval, limit=limit)
        frame = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
        frame.set_index("timestamp", inplace=True)
        frame = frame.astype(float)
        LOGGER.info("Fetched %s historical candles for %s", len(frame), market)
        return frame
    except Exception as exc:  # pragma: no cover - network tends to fail in tests
        LOGGER.warning("Failed to download candles: %s. Falling back to synthetic data.", exc)
        return _generate_synthetic_data(limit)


def _generate_synthetic_data(length: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    price = 20000 + rng.normal(0, 20, size=length).cumsum()
    high = price + rng.uniform(0, 5, size=length)
    low = price - rng.uniform(0, 5, size=length)
    open_ = price + rng.normal(0, 2, size=length)
    volume = rng.uniform(10, 50, size=length)
    index = pd.date_range(end=datetime.utcnow(), periods=length, freq="H")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": price, "volume": volume}, index=index
    )


def _build_targets(feature_frame: pd.DataFrame) -> pd.Series:
    forward_return = feature_frame["close"].pct_change().shift(-1)
    target = (forward_return > 0).astype(int)
    return target.dropna()


def _create_sequences(array: np.ndarray, targets: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    labels = []
    for idx in range(seq_len, len(array)):
        sequences.append(array[idx - seq_len : idx])
        labels.append(targets[idx])
    return np.stack(sequences), np.array(labels)


def _evaluate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    if len(targets) == 0:
        return float("nan")
    correct = (predictions == targets).sum()
    return float(correct) / len(targets)


def _evaluate_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    if len(targets) == 0:
        return float("nan")
    return float(np.mean((predictions - targets) ** 2))


def orchestrate_training(args: argparse.Namespace | None = None) -> Dict:
    cfg = load_config(args.config if args and args.config else None)
    configure_logging(cfg.get("logging.level", "INFO"), cfg.get("paths.logs_dir"))

    paths = {k: cfg["paths"][k] for k in ("registry_root", "checkpoints_dir")}
    Path(paths["registry_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)

    fetcher = DataFetcher(
        api_key=cfg.get("bitvavo.api_key", ""),
        api_secret=cfg.get("bitvavo.api_secret", ""),
        base_url=cfg.get("bitvavo.base_url", "https://api.bitvavo.com/v2"),
        request_timeout=int(cfg.get("bitvavo.request_timeout", 10)),
    )

    candles = _load_candles(
        fetcher,
        market=cfg.get("bitvavo.market", "BTC-EUR"),
        interval=cfg.get("bitvavo.interval", "1h"),
        limit=int(cfg.get("training.candles", 500)),
    )

    engineer = FeatureEngineer()
    features = engineer.compute(candles)
    target = _build_targets(features)
    features = features.loc[target.index]

    metadata_validator.check_data_complete(features)
    metadata_validator.assert_finite(features, name="features")
    metadata_validator.check_no_leakage(features.drop(columns=["close"], errors="ignore"), target)

    feature_cols = [
        col
        for col in features.columns
        if col not in {"open", "high", "low", "close", "volume"}
    ]
    if not feature_cols:
        raise ValueError("Feature computation produced no engineered columns.")

    X = features[feature_cols].to_numpy(dtype=np.float32)
    y = target.to_numpy(dtype=np.float32)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    lightgbm = LightGBMForecaster()
    lightgbm.train(X_train, y_train, eval_set=[(X_val, y_val)] if len(X_val) else None)
    val_pred = lightgbm.predict((X_val if len(X_val) else X_train))
    accuracy = _evaluate_accuracy(val_pred, (y_val if len(y_val) else y_train))

    seq_len = 20
    if len(X) <= seq_len:
        raise ValueError("Not enough data to build GRU sequences.")
    X_seq, y_seq = _create_sequences(X, y, seq_len=seq_len)
    split_seq = int(len(X_seq) * 0.8)
    gru_train_X, gru_val_X = X_seq[:split_seq], X_seq[split_seq:]
    gru_train_y, gru_val_y = y_seq[:split_seq], y_seq[split_seq:]

    gru = GRUForecaster(
        input_size=X.shape[1],
        training=TrainingConfig(epochs=cfg.get("training.gru_epochs", 10)),
    )
    gru.train(gru_train_X, gru_train_y)
    gru_val_pred = gru.predict(gru_val_X if len(gru_val_X) else gru_train_X)
    gru_mse = _evaluate_mse(gru_val_pred, gru_val_y if len(gru_val_y) else gru_train_y)

    features["lgb_proba"] = lightgbm.predict_proba(X)
    gru_prediction_full = gru.predict(X_seq)
    padded_gru = np.concatenate([
        np.full(seq_len, gru_prediction_full[0]),
        gru_prediction_full,
    ])
    features["gru_signal"] = padded_gru[: len(features)]

    env_data = features[["close"] + feature_cols + ["lgb_proba", "gru_signal"]]
    feature_columns_for_env = [col for col in env_data.columns if col != "close"]

    def make_env() -> TradingEnv:
        return TradingEnv(
            env_data,
            feature_columns=feature_columns_for_env,
            price_column="close",
            initial_cash=float(cfg.get("trading.initial_cash", 1000.0)),
            trading_cost=0.001,
            volatility_penalty=0.0,
            max_position=float(cfg.get("trading.max_position_size", 1.0)),
        )

    ppo = PPOPolicyWrapper(
        learning_rate=float(cfg.get("ppo.learning_rate", 3e-4)),
        gamma=float(cfg.get("ppo.gamma", 0.99)),
        batch_size=int(cfg.get("ppo.batch_size", 256)),
        gae_lambda=float(cfg.get("ppo.gae_lambda", 0.95)),
    )

    ppo_timesteps = int(cfg.get("ppo.timesteps", 50000))
    ppo.train(make_env, ppo_timesteps)

    evaluation_env = make_env()
    obs, _ = evaluation_env.reset()
    cumulative_reward = 0.0
    done = False
    while not done:
        action = ppo.predict(obs)
        obs, reward, done, _, _info = evaluation_env.step(action)
        cumulative_reward += reward

    registry = ModelRegistry(paths["registry_root"])
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "data": {
            "rows": len(features),
            "feature_columns": feature_cols,
            "env_features": feature_columns_for_env,
        },
        "metrics": {
            "lightgbm_accuracy": accuracy,
            "gru_mse": gru_mse,
            "ppo_reward": cumulative_reward,
        },
        "models": {
            "lightgbm": {"n_features": len(feature_cols)},
            "gru": {
                "input_size": X.shape[1],
                "hidden_size": 64,
                "num_layers": 2,
                "sequence_length": seq_len,
            },
            "ppo": {"timesteps": ppo_timesteps},
        },
        "config": {"market": cfg.get("bitvavo.market", "BTC-EUR")},
    }

    model_path = registry.register(metadata, lightgbm, gru, ppo)
    LOGGER.info("Registered model bundle at %s", model_path)

    if cfg.get("reporting.enable_telegram", False):
        send_alert(
            f"BetterBot training complete. Acc={accuracy:.3f}, MSE={gru_mse:.4f}, Reward={cumulative_reward:.2f}"
        )

    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BetterBot models")
    parser.add_argument("--config", help="Path to configuration file", default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    orchestrate_training(args)
