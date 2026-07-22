"""Unit tests for training-only model-family preprocessing."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from flowcast.modelling.config import load_model_config
from flowcast.modelling.preprocessing import fit_preprocessors
from flowcast.settings import load_settings


def _feature_records() -> list[dict[str, str]]:
    return [
        {
            "name": "numeric",
            "dtype": "Float64",
            "leakage_status": "known_at_origin",
        },
        {
            "name": "bounded",
            "dtype": "Float64",
            "leakage_status": "known_at_origin",
        },
        {
            "name": "flag",
            "dtype": "boolean",
            "leakage_status": "known_at_origin",
        },
        {
            "name": "band",
            "dtype": "string",
            "leakage_status": "known_at_origin",
        },
    ]


def test_preprocessors_learn_only_training_statistics(tmp_path) -> None:
    settings = load_settings()
    config = deepcopy(load_model_config(settings))
    config["preprocessing"]["explicit_binary_features"] = []
    config["preprocessing"]["bounded_numeric_features"] = ["bounded"]
    train = pd.DataFrame(
        {
            "numeric": pd.Series([1.0, 2.0, pd.NA], dtype="Float64"),
            "bounded": pd.Series([0.0, 10.0, 5.0], dtype="Float64"),
            "flag": pd.Series([True, False, pd.NA], dtype="boolean"),
            "band": pd.Series(["low", "high", "low"], dtype="string"),
        }
    )
    validation = pd.DataFrame(
        {
            "numeric": pd.Series([100.0], dtype="Float64"),
            "bounded": pd.Series([20.0], dtype="Float64"),
            "flag": pd.Series([True], dtype="boolean"),
            "band": pd.Series(["unseen"], dtype="string"),
        }
    )

    groups, fitted = fit_preprocessors(
        train,
        _feature_records(),
        config,
        tmp_path,
        settings,
    )

    assert groups.input_features == ("numeric", "bounded", "flag", "band")
    linear_stats = fitted["linear"].metadata["training_statistics"]
    assert linear_stats["numeric"]["imputer_statistics"]["numeric"] == 1.5
    assert linear_stats["numeric"]["scaler"]["mean"]["numeric"] == 1.5
    recurrent = fitted["recurrent"].metadata["training_statistics"]
    assert recurrent["bounded_numeric"]["scaler"]["data_max"]["bounded"] == 10.0
    assert fitted["tree"].metadata["training_statistics"]["numeric"][
        "scaler"
    ] == {"type": "none"}
    for record in fitted.values():
        transformed = record.processor.transform(validation[list(groups.input_features)])
        assert transformed.shape == (1, 5)
        assert np.isfinite(np.asarray(transformed, dtype=float)).all()
        assert record.path.is_file()
