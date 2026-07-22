"""Training-only preprocessing for classical and recurrent model families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from flowcast.data.artifacts import artifact_record
from flowcast.modelling.config import MODEL_FAMILIES
from flowcast.settings import Settings


@dataclass(frozen=True)
class FeatureGroups:
    """Ordered input features grouped by deterministic preprocessing policy."""

    input_features: tuple[str, ...]
    numeric: tuple[str, ...]
    bounded_numeric: tuple[str, ...]
    binary: tuple[str, ...]
    categorical: tuple[str, ...]


@dataclass(frozen=True)
class FittedPreprocessor:
    """One fitted family preprocessor plus its persisted metadata."""

    family: str
    path: Path
    processor: ColumnTransformer
    metadata: dict[str, Any]


def build_feature_groups(
    feature_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> FeatureGroups:
    """Classify manifest features without admitting keys, lineage, or targets."""

    names = [str(record["name"]) for record in feature_records]
    if not names or len(names) != len(set(names)):
        raise ValueError("Feature manifest names must be unique and non-empty")
    if any(record.get("leakage_status") != "known_at_origin" for record in feature_records):
        raise ValueError("All preprocessing inputs must be known at origin")
    preprocessing = config["preprocessing"]
    explicit_binary = set(preprocessing["explicit_binary_features"])
    bounded = set(preprocessing["bounded_numeric_features"])
    unknown = (explicit_binary | bounded) - set(names)
    if unknown:
        raise ValueError(f"Configured preprocessing features are absent: {unknown}")

    binary: list[str] = []
    categorical: list[str] = []
    numeric: list[str] = []
    for record in feature_records:
        name = str(record["name"])
        dtype = str(record["dtype"]).lower()
        if name in explicit_binary or dtype in {"bool", "boolean"}:
            binary.append(name)
        elif dtype in {"str", "string", "category"}:
            categorical.append(name)
        elif dtype.startswith("int") or dtype.startswith("float"):
            numeric.append(name)
        else:
            raise ValueError(f"Unsupported model feature dtype for {name}: {dtype}")
    if not bounded.issubset(numeric):
        raise ValueError("Bounded preprocessing features must be numeric")
    bounded_ordered = [name for name in numeric if name in bounded]
    standard_ordered = [name for name in numeric if name not in bounded]
    grouped = standard_ordered + bounded_ordered + binary + categorical
    if set(grouped) != set(names) or len(grouped) != len(names):
        raise RuntimeError("Feature grouping lost or duplicated a model input")
    return FeatureGroups(
        input_features=tuple(names),
        numeric=tuple(standard_ordered),
        bounded_numeric=tuple(bounded_ordered),
        binary=tuple(binary),
        categorical=tuple(categorical),
    )


def _numeric_pipeline(scaling: str) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        )
    ]
    if scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaling != "none":
        raise ValueError(f"Unsupported numeric scaling policy: {scaling}")
    return Pipeline(steps)


def build_preprocessor(
    groups: FeatureGroups,
    family_policy: dict[str, Any],
) -> ColumnTransformer:
    """Build an unfitted dense preprocessing graph for one model family."""

    transformers: list[tuple[str, Any, Iterable[str]]] = []
    if groups.numeric:
        transformers.append(
            (
                "numeric",
                _numeric_pipeline(str(family_policy["numeric_scaling"])),
                list(groups.numeric),
            )
        )
    if groups.bounded_numeric:
        transformers.append(
            (
                "bounded_numeric",
                _numeric_pipeline(str(family_policy["bounded_scaling"])),
                list(groups.bounded_numeric),
            )
        )
    if groups.binary:
        transformers.append(
            (
                "binary",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                list(groups.binary),
            )
        )
    if groups.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                list(groups.categorical),
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _values(values: Any) -> list[Any]:
    result = []
    for value in np.asarray(values).tolist():
        if isinstance(value, float) and np.isnan(value):
            result.append(None)
        elif isinstance(value, np.generic):
            result.append(value.item())
        else:
            result.append(value)
    return result


def _mapping(columns: Iterable[str], values: Any) -> dict[str, Any]:
    return dict(zip(columns, _values(values), strict=True))


def _numeric_metadata(
    processor: Pipeline,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "columns": list(columns),
        "imputer_statistics": _mapping(
            columns, processor.named_steps["imputer"].statistics_
        ),
    }
    scaler = processor.named_steps.get("scaler")
    if scaler is None:
        record["scaler"] = {"type": "none"}
    elif isinstance(scaler, StandardScaler):
        record["scaler"] = {
            "type": "standard",
            "mean": _mapping(columns, scaler.mean_),
            "scale": _mapping(columns, scaler.scale_),
        }
    elif isinstance(scaler, MinMaxScaler):
        record["scaler"] = {
            "type": "minmax",
            "data_min": _mapping(columns, scaler.data_min_),
            "data_max": _mapping(columns, scaler.data_max_),
            "scale": _mapping(columns, scaler.scale_),
        }
    else:
        raise TypeError(f"Unexpected fitted scaler: {type(scaler).__name__}")
    return record


def _training_statistics(
    processor: ColumnTransformer,
    groups: FeatureGroups,
) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    if groups.numeric:
        statistics["numeric"] = _numeric_metadata(
            processor.named_transformers_["numeric"], groups.numeric
        )
    if groups.bounded_numeric:
        statistics["bounded_numeric"] = _numeric_metadata(
            processor.named_transformers_["bounded_numeric"],
            groups.bounded_numeric,
        )
    if groups.binary:
        binary = processor.named_transformers_["binary"]
        statistics["binary"] = {
            "columns": list(groups.binary),
            "imputer_statistics": _mapping(groups.binary, binary.statistics_),
        }
    if groups.categorical:
        categorical = processor.named_transformers_["categorical"]
        imputer = categorical.named_steps["imputer"]
        encoder = categorical.named_steps["encoder"]
        statistics["categorical"] = {
            "columns": list(groups.categorical),
            "imputer_statistics": _mapping(
                groups.categorical, imputer.statistics_
            ),
            "categories": {
                name: _values(values)
                for name, values in zip(
                    groups.categorical,
                    encoder.categories_,
                    strict=True,
                )
            },
        }
    return statistics


def fit_preprocessors(
    train_frame: pd.DataFrame,
    feature_records: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    settings: Settings,
) -> tuple[FeatureGroups, dict[str, FittedPreprocessor]]:
    """Fit every approved preprocessor on train rows only and persist it."""

    if train_frame.empty:
        raise ValueError("Training partition cannot be empty")
    groups = build_feature_groups(feature_records, config)
    missing = set(groups.input_features) - set(train_frame.columns)
    if missing:
        raise ValueError(f"Training frame is missing model inputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted: dict[str, FittedPreprocessor] = {}
    policies = config["preprocessing"]["families"]
    for family in MODEL_FAMILIES:
        processor = build_preprocessor(groups, policies[family])
        processor.fit(train_frame[list(groups.input_features)])
        sample = processor.transform(
            train_frame.loc[:, list(groups.input_features)].head(2_048)
        )
        if sample.ndim != 2 or not np.isfinite(np.asarray(sample, dtype=float)).all():
            raise RuntimeError(f"{family} preprocessing produced invalid values")
        path = output_dir / f"{family}.joblib"
        joblib.dump(processor, path, compress=3)
        output_features = [str(name) for name in processor.get_feature_names_out()]
        metadata = {
            "family": family,
            "input_feature_count": len(groups.input_features),
            "input_features": list(groups.input_features),
            "output_feature_count": len(output_features),
            "output_features": output_features,
            "policy": {
                "numeric_scaling": str(policies[family]["numeric_scaling"]),
                "bounded_scaling": str(policies[family]["bounded_scaling"]),
                "categorical_encoding": "one_hot_ignore_unknown",
            },
            "training_statistics": _training_statistics(processor, groups),
            "artifact": artifact_record(path, settings),
            "libraries": {
                "joblib": joblib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        fitted[family] = FittedPreprocessor(
            family=family,
            path=path,
            processor=processor,
            metadata=metadata,
        )
    return groups, fitted
