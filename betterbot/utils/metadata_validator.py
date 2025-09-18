"""Validation helpers for BetterBot datasets and model artefacts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_MODEL_FILES = {"lightgbm.pkl", "gru.pt", "ppo.zip", "metadata.json"}


def check_no_leakage(features: pd.DataFrame, target: pd.Series, *, threshold: float = 0.98) -> None:
    """Raise if any feature is *too* correlated with the target variable."""

    aligned = features.reindex(target.index)
    correlations = aligned.corrwith(target)
    offenders = correlations[correlations.abs() > threshold]
    if not offenders.empty:
        raise ValueError(
            "Potential leakage detected: feature-target correlations too high -> "
            + ", ".join(f"{col}:{corr:.2f}" for col, corr in offenders.items())
        )


def check_data_complete(data: pd.DataFrame, *, max_missing_fraction: float = 0.01) -> None:
    """Ensure the dataset has no large gaps or missing values."""

    missing_ratio = data.isna().mean()
    offenders = missing_ratio[missing_ratio > max_missing_fraction]
    if not offenders.empty:
        raise ValueError(
            "Dataset contains too many missing values: "
            + ", ".join(f"{col}:{ratio:.2%}" for col, ratio in offenders.items())
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError("Dataset index must be sorted ascending for time-series modelling.")


def validate_model_files(folder: str | Path, required_files: Iterable[str] | None = None) -> None:
    """Check that the given folder contains all expected artefacts."""

    required = set(required_files or REQUIRED_MODEL_FILES)
    folder_path = Path(folder)
    missing = [fname for fname in required if not (folder_path / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model folder {folder_path} is missing files: {', '.join(missing)}"
        )


def assert_finite(values: pd.DataFrame | pd.Series | np.ndarray, *, name: str = "values") -> None:
    array = np.asarray(values)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite entries detected in {name}.")


__all__ = [
    "check_no_leakage",
    "check_data_complete",
    "validate_model_files",
    "assert_finite",
]
