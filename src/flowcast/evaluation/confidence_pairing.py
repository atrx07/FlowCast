"""Exact-row deep versus classical volume error comparisons."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from flowcast.evaluation.regression import regression_metrics


def _dimension_groups(
    frame: pd.DataFrame,
    dimensions: Sequence[str],
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    yield "overall", "all", frame
    for dimension in dimensions:
        for value, group in frame.groupby(dimension, sort=True, observed=True):
            yield dimension, str(value), group


def paired_volume_frame(regression: pd.DataFrame) -> pd.DataFrame:
    """Align deep and classical volume forecasts on their exact common rows."""

    volume = regression.loc[regression["target"].eq("volume")].copy()
    classical = volume.loc[
        volume["model_version"].eq("classical_regression_v1")
    ]
    deep = volume.loc[volume["model_version"].eq("recurrent_volume_v1")]
    keys = [
        "road_id",
        "timestamp",
        "target_timestamp",
        "split",
        "horizon_windows",
        "horizon_minutes",
    ]
    context = list(
        dict.fromkeys(
            [
                *keys,
                "actual",
                "origin_hour",
                "weekday",
                "weekday_type",
                "peak_status",
                "weather_condition",
                "actual_congestion",
            ]
        )
    )
    comparison = [
        "prediction",
        "absolute_error",
        "interval_lower",
        "interval_upper",
        "interval_covered",
    ]
    paired = classical.loc[:, context + comparison].merge(
        deep.loc[:, keys + ["actual", *comparison]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_classical", "_deep"),
    )
    if not np.allclose(paired["actual_classical"], paired["actual_deep"]):
        raise RuntimeError("Paired volume rows disagree on actual values")
    paired = paired.rename(columns={"actual_classical": "actual"}).drop(
        columns="actual_deep"
    )
    paired["absolute_error_delta_deep_minus_classical"] = (
        paired["absolute_error_deep"] - paired["absolute_error_classical"]
    )
    paired["winner"] = np.select(
        [
            paired["absolute_error_deep"].lt(paired["absolute_error_classical"]),
            paired["absolute_error_classical"].lt(paired["absolute_error_deep"]),
        ],
        ["deep", "classical"],
        default="tie",
    )
    return paired.sort_values(keys, kind="stable").reset_index(drop=True)


def paired_volume_slices(
    frame: pd.DataFrame,
    dimensions: Sequence[str],
    minimum_rows: int,
) -> pd.DataFrame:
    """Compare deep and classical volume error on identical supported slices."""

    rows: list[dict[str, Any]] = []
    keys = ["horizon_windows", "horizon_minutes", "split"]
    metric_names = (
        "deep_rmse",
        "classical_rmse",
        "rmse_delta_deep_minus_classical",
        "deep_mae",
        "classical_mae",
        "mean_absolute_error_delta",
        "deep_row_win_rate",
    )
    for key, base in frame.groupby(keys, sort=True, observed=True):
        identity = dict(zip(keys, key, strict=True))
        for dimension, value, group in _dimension_groups(base, dimensions):
            sufficient = len(group) >= minimum_rows
            if sufficient:
                deep = regression_metrics(group["actual"], group["prediction_deep"])
                classical = regression_metrics(
                    group["actual"], group["prediction_classical"]
                )
                metrics = {
                    "deep_rmse": deep["rmse"],
                    "classical_rmse": classical["rmse"],
                    "rmse_delta_deep_minus_classical": deep["rmse"]
                    - classical["rmse"],
                    "deep_mae": deep["mae"],
                    "classical_mae": classical["mae"],
                    "mean_absolute_error_delta": float(
                        group["absolute_error_delta_deep_minus_classical"].mean()
                    ),
                    "deep_row_win_rate": float(group["winner"].eq("deep").mean()),
                }
            else:
                metrics = {name: np.nan for name in metric_names}
            rows.append(
                {
                    **identity,
                    "dimension": dimension,
                    "slice_value": value,
                    "rows": len(group),
                    "sufficient_support": sufficient,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
