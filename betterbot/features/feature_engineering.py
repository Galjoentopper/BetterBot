"""Feature engineering utilities for BetterBot."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

PriceFrame = pd.DataFrame


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index implementation using exponential smoothing."""

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace({0: np.nan})
    return 100 - (100 / (1 + rs))


def _rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Annualised rolling volatility based on log returns."""

    return returns.rolling(window=window).std() * np.sqrt(365 * 24 * 60)


@dataclass
class FeatureEngineer:
    """Compute technical indicators over OHLCV time series."""

    price_column: str = "close"
    volume_column: str = "volume"
    windows: Mapping[str, Iterable[int]] = field(
        default_factory=lambda: {
            "sma": (5, 10, 20),
            "ema": (5, 10, 20),
            "volatility": (10, 30),
            "returns": (1, 5, 10),
        }
    )
    rsi_window: int = 14

    def compute(self, candles: PriceFrame) -> PriceFrame:
        """Return a feature matrix aligned with ``candles``.

        Parameters
        ----------
        candles:
            DataFrame with at least ``open``, ``high``, ``low``, ``close``, ``volume`` columns.
        """

        if self.price_column not in candles:
            raise KeyError(f"Price column '{self.price_column}' not available in data frame.")
        frame = candles.copy()
        frame.sort_index(inplace=True)

        price = frame[self.price_column].astype(float)
        frame["log_return_1"] = np.log(price).diff().replace([np.inf, -np.inf], np.nan)

        for window in self.windows.get("returns", []):
            frame[f"return_{window}"] = price.pct_change(periods=window)

        for window in self.windows.get("sma", []):
            frame[f"sma_{window}"] = price.rolling(window=window).mean()

        for window in self.windows.get("ema", []):
            frame[f"ema_{window}"] = price.ewm(span=window, adjust=False).mean()

        frame["rsi"] = _compute_rsi(price, window=self.rsi_window)

        for window in self.windows.get("volatility", []):
            frame[f"volatility_{window}"] = _rolling_volatility(frame["log_return_1"], window=window)

        if self.volume_column in frame:
            volume = frame[self.volume_column].astype(float)
            frame["volume_z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

        highest = frame["high"].rolling(window=20).max()
        lowest = frame["low"].rolling(window=20).min()
        frame["stochastic_k"] = (price - lowest) / (highest - lowest)
        frame["stochastic_d"] = frame["stochastic_k"].rolling(window=3).mean()

        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        return frame

    def compute_vector(self, candles: PriceFrame) -> Dict[str, float]:
        """Return the most recent feature vector as a dictionary."""

        features = self.compute(candles)
        if features.empty:
            raise ValueError("Not enough data to compute features; resulting frame is empty.")
        last_row = features.iloc[-1]
        return last_row.to_dict()

    def feature_columns(self, candles: PriceFrame) -> List[str]:
        """Return the list of feature columns produced by :meth:`compute`."""

        features = self.compute(candles)
        return [col for col in features.columns if col not in candles.columns]


def compute_features(candles: PriceFrame) -> PriceFrame:
    """Convenience wrapper for default feature set."""

    return FeatureEngineer().compute(candles)


__all__ = ["FeatureEngineer", "compute_features"]
