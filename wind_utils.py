"""Utility helpers for computing wind vector statistics.

The functions in this module are shared between the hourly aggregation
script and the Flask application so that both components calculate wind
statistics in the exact same way.
"""
from __future__ import annotations

from typing import Iterable, List, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_series(values: Iterable[float]) -> pd.Series:
    """Convert *values* to a numeric pandas Series.

    Any value that cannot be coerced to a float will be converted to ``NaN``
    which allows downstream logic to drop it safely.
    """

    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(list(values))
    return pd.to_numeric(series, errors="coerce")


def direction_average(values: Iterable[float]) -> float:
    """Return the average wind direction for *values*.

    The calculation converts directions to vectors on the unit circle and
    averages their sine and cosine components.  The result is expressed in
    degrees within the [0, 360) range.  If *values* does not contain any
    finite numbers the function returns ``numpy.nan``.
    """

    series = _to_series(values).dropna()
    if series.empty:
        return float("nan")

    radians = np.deg2rad(series)
    sin_sum = np.sin(radians).sum()
    cos_sum = np.cos(radians).sum()
    if np.isclose(sin_sum, 0.0) and np.isclose(cos_sum, 0.0):
        return float("nan")

    angle = np.degrees(np.arctan2(sin_sum, cos_sum))
    return float((angle + 360.0) % 360.0)


def wind_vector_mean(
    speeds: Iterable[float],
    directions: Iterable[float],
) -> Tuple[float, float]:
    """Compute the vector mean of wind *speeds* and *directions*.

    The result is a tuple ``(mean_speed, mean_direction)``.  The mean speed
    is calculated from the averaged vector components which ensures that the
    magnitude properly reflects the vector nature of the signal.  The
    direction is derived from the same vector.  If there are no valid pairs
    of observations the function returns ``(nan, nan)``.  When valid
    directions exist but all speed values are missing, the function falls
    back to :func:`direction_average` for the angle and returns ``nan`` for
    the speed component.
    """

    series = pd.DataFrame({"speed": speeds, "direction": directions})
    series["speed"] = pd.to_numeric(series["speed"], errors="coerce")
    series["direction"] = pd.to_numeric(series["direction"], errors="coerce")
    series = series.dropna(subset=["direction"])

    if series.empty:
        return float("nan"), float("nan")

    if series["speed"].isna().all():
        return float("nan"), direction_average(series["direction"])

    series = series.dropna(subset=["speed"])
    if series.empty:
        return float("nan"), float("nan")

    radians = np.deg2rad(series["direction"])
    u = (series["speed"] * np.cos(radians)).sum()
    v = (series["speed"] * np.sin(radians)).sum()

    mean_speed = np.hypot(u, v) / len(series)
    if np.isclose(u, 0.0) and np.isclose(v, 0.0):
        mean_direction = direction_average(series["direction"])
    else:
        mean_direction = float((np.degrees(np.arctan2(v, u)) + 360.0) % 360.0)

    return float(mean_speed), float(mean_direction)


def wind_vector_resample(
    df: pd.DataFrame,
    freq: str,
    *,
    direction_column: str,
    speed_columns: Sequence[str],
) -> pd.DataFrame:
    """Resample wind measurements contained in ``df``.

    The returned dataframe has the same columns as ``speed_columns`` plus the
    ``direction_column`` (if present in *df*) and contains vector means for
    each resampled period.
    """

    available_speeds: List[str] = [col for col in speed_columns if col in df.columns]
    include_direction = direction_column in df.columns

    resampled_index = df.resample(freq).mean().index
    if not available_speeds and not include_direction:
        return pd.DataFrame(index=resampled_index)

    if not include_direction:
        return df[available_speeds].resample(freq).mean()

    rows: List[MutableMapping[str, float]] = []
    for _, group in df.resample(freq):
        row: MutableMapping[str, float] = {}
        direction_series = (
            group[direction_column] if include_direction else pd.Series(dtype=float)
        )

        computed_direction = float("nan")
        for speed_col in available_speeds:
            speed_mean, dir_mean = wind_vector_mean(group[speed_col], direction_series)
            row[speed_col] = speed_mean
            if not np.isnan(dir_mean):
                computed_direction = dir_mean

        if include_direction:
            if np.isnan(computed_direction):
                computed_direction = direction_average(direction_series)
            row[direction_column] = computed_direction

        rows.append(row)

    result = pd.DataFrame(rows, index=resampled_index)
    return result
