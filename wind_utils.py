"""Utility helpers for computing wind speed and direction statistics."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindStats:
    """Container for aggregated wind statistics."""

    mean_speed_scalar: float
    mean_speed_resultant: float
    mean_dir: float
    R: float

    def to_dict(self) -> dict:
        return {
            "mean_speed_scalar": self.mean_speed_scalar,
            "mean_speed_resultant": self.mean_speed_resultant,
            "mean_dir": self.mean_dir,
            "R": self.R,
        }


def _prepare_wind_components(
    speeds: pd.Series, directions: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare filtered speed and direction arrays and their radian representation."""
    speeds_values = pd.to_numeric(speeds, errors="coerce").to_numpy(dtype=float)
    directions_values = pd.to_numeric(directions, errors="coerce").to_numpy(dtype=float)

    valid_mask = (~np.isnan(speeds_values)) & (~np.isnan(directions_values))
    return speeds_values, directions_values, np.radians(directions_values[valid_mask])


def calculate_wind_stats(
    df: pd.DataFrame,
    speed_col: str = "WIND_SPEED",
    dir_col: str = "WIND_ANGLE",
) -> WindStats:
    """Calculate aggregated wind statistics for the provided dataframe.

    Parameters
    ----------
    df:
        DataFrame containing wind data.
    speed_col:
        Column name containing wind speed values (m/s).
    dir_col:
        Column name containing wind direction values (degrees).

    Returns
    -------
    WindStats
        Dataclass with scalar mean speed, resultant mean speed, mean direction and
        the consistency coefficient ``R``.
    """
    if speed_col not in df.columns or dir_col not in df.columns:
        return WindStats(np.nan, np.nan, np.nan, np.nan)

    speeds_values, directions_values, rad = _prepare_wind_components(
        df[speed_col], df[dir_col]
    )

    valid_mask = (~np.isnan(speeds_values)) & (~np.isnan(directions_values))

    if not np.any(valid_mask):
        if speeds_values.size == 0 or np.all(np.isnan(speeds_values)):
            scalar_mean = np.nan
        else:
            scalar_mean = float(np.nanmean(speeds_values))
        return WindStats(scalar_mean, np.nan, np.nan, np.nan)

    speeds_valid = speeds_values[valid_mask]

    u = speeds_valid * np.cos(rad)
    v = speeds_valid * np.sin(rad)

    u_mean = u.mean() if u.size else np.nan
    v_mean = v.mean() if v.size else np.nan

    resultant_speed = float(np.sqrt(u_mean**2 + v_mean**2)) if u.size else np.nan
    mean_direction = float((np.degrees(np.arctan2(v_mean, u_mean))) % 360) if u.size else np.nan

    if speeds_values.size == 0 or np.all(np.isnan(speeds_values)):
        scalar_mean = np.nan
    else:
        scalar_mean = float(np.nanmean(speeds_values))

    if np.isnan(resultant_speed) or np.isnan(scalar_mean) or scalar_mean == 0:
        consistency = np.nan
    else:
        consistency = resultant_speed / scalar_mean

    return WindStats(scalar_mean, resultant_speed, mean_direction, consistency)
