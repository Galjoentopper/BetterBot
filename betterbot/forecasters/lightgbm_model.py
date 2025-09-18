"""LightGBM based forecaster used for directional trading signals."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

try:  # pragma: no cover - dependency loading is environment specific
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LightGBM is required. Install via 'pip install lightgbm'."
    ) from exc

LOGGER = logging.getLogger(__name__)


class LightGBMForecaster:
    """Wrapper around ``lightgbm.LGBMClassifier`` for convenience."""

    def __init__(self, **parameters: Any) -> None:
        defaults = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
        }
        defaults.update(parameters)
        self.model = lgb.LGBMClassifier(**defaults)
        LOGGER.debug("Initialised LightGBMForecaster with params: %s", defaults)

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        eval_set: Optional[Iterable[tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        """Fit the model on the supplied dataset."""

        LOGGER.info("Training LightGBM model on %s samples", len(features))
        if not eval_set:
            self.model.fit(features, targets)
            return

        fit_params: dict[str, Any] = {
            "eval_set": eval_set,
            "eval_metric": "auc",
        }

        try:
            self.model.fit(features, targets, early_stopping_rounds=20, **fit_params)
        except TypeError:
            # LightGBM >=4.0 expects callbacks instead of early_stopping_rounds
            callbacks = fit_params.pop("callbacks", [])
            callbacks.append(lgb.early_stopping(20, verbose=False))
            fit_params["callbacks"] = callbacks
            self.model.fit(features, targets, **fit_params)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted class labels (0 = flat, 1 = long)."""

        self._check_trained()
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities for the positive class."""

        self._check_trained()
        proba = self.model.predict_proba(features)
        return proba[:, 1]

    def feature_importances(self) -> np.ndarray:
        """Expose LightGBM's feature importances."""

        self._check_trained()
        return self.model.feature_importances_

    def save_model(self, path: str | Path) -> None:
        """Persist the trained model using pickle."""

        self._check_trained()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Saved LightGBM model to %s", destination)

    def load_model(self, path: str | Path) -> None:
        """Load a previously saved model."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"LightGBM model file not found: {source}")
        with source.open("rb") as handle:
            self.model = pickle.load(handle)
        LOGGER.info("Loaded LightGBM model from %s", source)

    def _check_trained(self) -> None:
        if not getattr(self.model, "booster_", None):
            raise RuntimeError("LightGBM model has not been trained yet.")


__all__ = ["LightGBMForecaster"]
