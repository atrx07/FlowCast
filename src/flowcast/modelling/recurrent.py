"""Step 15 orchestration for the multi-horizon recurrent volume model."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd

from flowcast.data.artifacts import (
    validate_artifact_version,
    write_parquet,
)
from flowcast.modelling.classical_artifacts import (
    load_classical_regression_model,
)
from flowcast.modelling.classical_report import write_csv
from flowcast.modelling.inputs import (
    load_modeling_partition,
    load_preprocessor,
    load_verified_modeling_artifacts,
)
from flowcast.modelling.recurrent_artifacts import (
    RecurrentPaths,
    recurrent_paths,
)
from flowcast.modelling.recurrent_config import load_recurrent_config
from flowcast.modelling.recurrent_finalize import (
    RecurrentFinalization,
    finalize_recurrent_run,
)
from flowcast.modelling.recurrent_outputs import (
    choose_candidate,
    compare_with_classical,
    metric_frame,
    prediction_frame,
)
from flowcast.modelling.recurrent_training import (
    CandidateTrainingResult,
    predict_partition,
    seed_torch,
    select_device,
    train_candidate,
)
from flowcast.modelling.recurrent_support import (
    build_feature_manifest,
    candidate_endpoints,
    input_features,
    persist_pretest_freeze,
    portable_record_path,
    reload_model,
    write_environment_snapshot,
    target_columns,
)
from flowcast.modelling.registry_artifacts import load_classical_registry
from flowcast.modelling.sequence_data import (
    endpoint_keys,
    fit_target_scaler,
    prepare_partition,
)
from flowcast.settings import Settings


@dataclass(frozen=True)
class RecurrentVolumeResult:
    """Completed recurrent artifacts and canonical summary."""

    paths: RecurrentPaths
    summary: dict[str, Any]


def run_recurrent_volume(
    settings: Settings,
    *,
    version: str | None = None,
    device: str | None = None,
) -> RecurrentVolumeResult:
    """Tune, freeze, test, compare, persist, and verify the recurrent model."""

    started = time.perf_counter()
    config, candidates, config_path = load_recurrent_config(settings)
    if device is not None:
        normalized_device = str(device).lower()
        if normalized_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unsupported recurrent device override: {device}")
        config["device"]["policy"] = normalized_device
    selected_version = validate_artifact_version(version or str(config["version"]))
    paths = recurrent_paths(settings, selected_version)
    write_environment_snapshot(paths.environment_path)
    modeling = load_verified_modeling_artifacts(settings)
    if modeling.summary["version"] != config["upstream"]["modelling_version"]:
        raise RuntimeError("Recurrent modelling upstream version changed")
    load_classical_registry(
        settings,
        version=str(config["upstream"]["classical_registry_version"]),
    )
    preprocessor = load_preprocessor(settings, "recurrent")
    features = input_features(modeling)
    targets = target_columns(config)
    train_frame = load_modeling_partition(settings, "train")
    validation_frame = load_modeling_partition(settings, "validation")
    training = prepare_partition(
        "train",
        train_frame,
        preprocessor,
        features,
        targets,
    )
    validation = prepare_partition(
        "validation",
        validation_frame,
        preprocessor,
        features,
        targets,
    )
    maximum_length = max(candidate.sequence_length for candidate in candidates)
    longest = max(candidates, key=lambda candidate: candidate.sequence_length)
    common_train_endpoints = candidate_endpoints(training, longest, config)
    common_validation_endpoints = candidate_endpoints(validation, longest, config)
    validation_keys = endpoint_keys(validation, common_validation_endpoints)
    scaler = fit_target_scaler(training.frame, common_train_endpoints, targets)
    feature_manifest = build_feature_manifest(modeling, training, features, config)
    results: list[CandidateTrainingResult] = []
    endpoint_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate in candidates:
        train_endpoints = candidate_endpoints(training, candidate, config)
        validation_endpoints = candidate_endpoints(
            validation,
            candidate,
            config,
            validation_keys,
        )
        endpoint_map[candidate.candidate_id] = (
            train_endpoints,
            validation_endpoints,
        )
        results.append(
            train_candidate(
                candidate,
                training,
                validation,
                train_endpoints,
                validation_endpoints,
                scaler,
                config,
                settings.seed,
            )
        )
    selected = choose_candidate(results)
    selected_train, selected_validation = endpoint_map[
        selected.candidate.candidate_id
    ]
    selection_record, scaler_metadata = persist_pretest_freeze(
        paths,
        settings,
        config,
        candidates,
        results,
        selected,
        training,
        validation,
        selected_train,
        selected_validation,
        scaler,
        feature_manifest,
    )

    test_frame = load_modeling_partition(
        settings,
        "test",
        purpose="final_evaluation",
    )
    test = prepare_partition(
        "test",
        test_frame,
        preprocessor,
        features,
        targets,
    )
    common_test_endpoints = candidate_endpoints(test, longest, config)
    test_keys = endpoint_keys(test, common_test_endpoints)
    selected_test = candidate_endpoints(
        test,
        selected.candidate,
        config,
        test_keys,
    )
    device_config = config["device"]
    seed_torch(
        settings.seed,
        bool(device_config["deterministic_algorithms"]),
        int(device_config["cpu_threads"]),
    )
    device = select_device(str(device_config["policy"]))
    model = reload_model(
        paths.checkpoint_path,
        selected.candidate,
        training.features.shape[1],
        device,
    )
    reloaded_validation, validation_prediction_seconds = predict_partition(
        model,
        validation,
        selected_validation,
        selected.candidate.sequence_length,
        scaler,
        selected.candidate.batch_size,
        int(device_config["dataloader_workers"]),
        device,
        settings.seed,
    )
    if not np.allclose(
        reloaded_validation,
        selected.validation_predictions,
        rtol=1e-6,
        atol=1e-5,
    ):
        raise RuntimeError("Reloaded checkpoint changed validation predictions")
    test_predictions, test_prediction_seconds = predict_partition(
        model,
        test,
        selected_test,
        selected.candidate.sequence_length,
        scaler,
        selected.candidate.batch_size,
        int(device_config["dataloader_workers"]),
        device,
        settings.seed,
    )
    validation_predictions = prediction_frame(
        validation,
        selected_validation,
        reloaded_validation,
        "validation",
        selected.candidate,
        selected_version,
    )
    test_prediction_frame = prediction_frame(
        test,
        selected_test,
        test_predictions,
        "test",
        selected.candidate,
        selected_version,
    )
    predictions = pd.concat(
        [validation_predictions, test_prediction_frame],
        ignore_index=True,
    )
    write_parquet(predictions, paths.predictions_path)
    validation_actual = validation.frame.iloc[selected_validation][
        list(targets)
    ].to_numpy(dtype=float)
    test_actual = test.frame.iloc[selected_test][list(targets)].to_numpy(dtype=float)
    metrics_table, metrics = metric_frame(
        validation_actual,
        reloaded_validation,
        test_actual,
        test_predictions,
    )
    write_csv(metrics_table, paths.metrics_path)
    _, _, classical_summary = load_classical_regression_model(
        settings,
        "volume",
        1,
        version=str(config["upstream"]["classical_regression_version"]),
    )
    classical_path = portable_record_path(
        classical_summary["artifacts"]["predictions"],
        settings,
    )
    classical_predictions = pd.read_parquet(classical_path)
    comparison_table, comparison = compare_with_classical(
        test_prediction_frame,
        classical_predictions,
    )
    write_csv(comparison_table, paths.comparison_path)
    summary = finalize_recurrent_run(
        RecurrentFinalization(
            settings=settings,
            paths=paths,
            config=config,
            config_path=config_path,
            modeling=modeling,
            candidates=candidates,
            results=results,
            selected=selected,
            training=training,
            validation=validation,
            test=test,
            selected_train=selected_train,
            selected_validation=selected_validation,
            selected_test=selected_test,
            feature_manifest=feature_manifest,
            scaler_metadata=scaler_metadata,
            selection_record=selection_record,
            metrics=metrics,
            comparison=comparison,
            classical_predictions_sha256=classical_summary["artifacts"][
                "predictions"
            ]["sha256"],
            predictions=predictions,
            validation_predictions=validation_predictions,
            test_predictions=test_prediction_frame,
            device=device,
            started=started,
            validation_prediction_seconds=validation_prediction_seconds,
            test_prediction_seconds=test_prediction_seconds,
        )
    )
    return RecurrentVolumeResult(paths=paths, summary=summary)
