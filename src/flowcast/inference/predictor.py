"""Single public prediction interface over frozen FlowCast artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from flowcast.inference.confidence import (
    accident_risk_band,
    interval_width,
    probability_confidence,
    regression_interval,
)
from flowcast.inference.feature_prep import (
    available_roads,
    latest_common_origin,
    normalize_origin,
    recurrent_feature_sequences,
    select_origin_rows,
)
from flowcast.inference.inputs import (
    VerifiedInferenceContext,
    load_verified_inference_context,
)
from flowcast.inference.model_router import FrozenModelRouter
from flowcast.inference.schemas import (
    CONGESTION_LABELS,
    PredictionRequest,
    validate_prediction_frame,
)
from flowcast.modelling.inputs import load_preprocessor
from flowcast.modelling.recurrent_artifacts import load_recurrent_volume_model
from flowcast.settings import Settings


@dataclass(frozen=True)
class PredictionResult:
    """Validated forecast rows, lineage, and measured service runtime."""

    request: PredictionRequest
    frame: pd.DataFrame
    lineage: dict[str, Any]
    initialization_seconds: float
    prediction_seconds: float

    @property
    def total_seconds(self) -> float:
        """Return cold service initialization plus request execution time."""

        return self.initialization_seconds + self.prediction_seconds


class Predictor:
    """Load frozen models once and serve validated multi-target forecasts."""

    def __init__(self, settings: Settings, *, device: str | None = None) -> None:
        started = time.perf_counter()
        self.settings = settings
        self.context: VerifiedInferenceContext = load_verified_inference_context(
            settings
        )
        configured = self.context.config["device"]
        self.device_name = str(device or configured["default"]).lower()
        if self.device_name not in set(configured["allowed"]):
            raise ValueError(f"Unsupported inference device: {self.device_name}")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested but is unavailable")
        torch.set_num_threads(max(1, int(configured["cpu_threads"])))
        self.device = torch.device(self.device_name)

        recurrent_version = str(
            self.context.config["upstream"]["recurrent_version"]
        )
        (
            self.recurrent_model,
            self.target_scaler,
            self.recurrent_card,
            self.recurrent_summary,
        ) = load_recurrent_volume_model(
            settings,
            version=recurrent_version,
            device=self.device_name,
        )
        expected_recurrent = str(
            self.context.config["active_routing"]["volume"]["model_version"]
        )
        if self.recurrent_summary["version"] != expected_recurrent:
            raise RuntimeError("Active recurrent routing version changed")
        self.recurrent_preprocessor = load_preprocessor(settings, "recurrent")
        self.router = FrozenModelRouter(
            settings,
            self.context.registry,
            self.context.config,
        )
        self.initialization_seconds = time.perf_counter() - started

    def build_request(
        self,
        *,
        road_ids: Sequence[str] | None = None,
        origin_timestamp: str | None = None,
        horizons: Sequence[int] | None = None,
    ) -> PredictionRequest:
        """Build a data-aware request, defaulting to latest full-corridor data."""

        frame = self.context.processed.frame
        all_roads = available_roads(frame)
        selected_roads = tuple(road_ids) if road_ids is not None else all_roads
        unknown = sorted(set(selected_roads) - set(all_roads))
        if unknown:
            raise ValueError(f"Unknown road_ids: {unknown}")
        if origin_timestamp is None:
            origin = latest_common_origin(frame, selected_roads)
        else:
            origin = normalize_origin(origin_timestamp, self.settings.timezone)
        selected_horizons = (
            tuple(horizons)
            if horizons is not None
            else tuple(self.context.config["request"]["horizons"])
        )
        return PredictionRequest.from_values(
            selected_roads,
            origin.isoformat(),
            selected_horizons,
            device=self.device_name,
        )

    def _validate_request(
        self,
        request: PredictionRequest,
    ) -> tuple[PredictionRequest, pd.Timestamp]:
        config = self.context.config
        allowed_horizons = set(int(value) for value in config["request"]["horizons"])
        if not set(request.horizons).issubset(allowed_horizons):
            raise ValueError(
                f"Horizons must be within {sorted(allowed_horizons)}"
            )
        if request.device != self.device_name:
            raise ValueError("Request device differs from initialized Predictor")
        maximum = int(config["request"]["maximum_roads"])
        if len(request.road_ids) > maximum:
            raise ValueError(f"A request cannot exceed {maximum} roads")
        known = set(available_roads(self.context.processed.frame))
        unknown = sorted(set(request.road_ids) - known)
        if unknown:
            raise ValueError(f"Unknown road_ids: {unknown}")
        origin = normalize_origin(
            request.origin_timestamp,
            self.settings.timezone,
        )
        normalized = PredictionRequest.from_values(
            request.road_ids,
            origin.isoformat(),
            request.horizons,
            device=request.device,
        )
        return normalized, origin

    def _recurrent_predictions(
        self,
        request: PredictionRequest,
        origin: pd.Timestamp,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        input_features = self.recurrent_card["features"]["input_features"]
        request_config = self.context.config["request"]
        sequences, endpoints = recurrent_feature_sequences(
            self.context.processed.frame,
            request.road_ids,
            origin,
            int(request_config["recurrent_sequence_length"]),
            int(request_config["cadence_minutes"]),
            input_features,
            self.recurrent_preprocessor,
        )
        tensor = torch.from_numpy(sequences).to(self.device)
        self.recurrent_model.eval()
        with torch.inference_mode():
            scaled = self.recurrent_model(tensor).detach().cpu().numpy()
        predictions = self.target_scaler.inverse_transform(scaled)
        if predictions.shape != (len(request.road_ids), 4):
            raise RuntimeError("Recurrent output shape changed")
        if not np.isfinite(predictions).all():
            raise RuntimeError("Recurrent predictions are non-finite")
        return predictions, endpoints

    @staticmethod
    def _regression_values(
        estimator: Any,
        card: dict[str, Any],
        origins: pd.DataFrame,
    ) -> np.ndarray:
        features = card["features"]["input_features"]
        values = np.asarray(estimator.predict(origins[features]), dtype=float)
        if values.shape != (len(origins),) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid regression output for {card['job_id']}")
        return values

    @staticmethod
    def _classification_values(
        estimator: Any,
        card: dict[str, Any],
        origins: pd.DataFrame,
    ) -> np.ndarray:
        features = card["features"]["input_features"]
        values = np.asarray(estimator.predict_proba(origins[features]), dtype=float)
        classes = len(card["target"]["class_order"])
        if values.shape != (len(origins), classes):
            raise RuntimeError(f"Invalid probability output for {card['job_id']}")
        return values

    def _horizon_outputs(
        self,
        horizon: int,
        origins: pd.DataFrame,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for target in ("volume", "speed", "travel_time"):
            estimator, card, _ = self.router.load(target, horizon)
            outputs[target] = self._regression_values(estimator, card, origins)
            outputs[f"{target}_card"] = card
        for target in ("congestion", "accident"):
            estimator, card, _ = self.router.load(target, horizon)
            outputs[target] = self._classification_values(estimator, card, origins)
            outputs[f"{target}_card"] = card
        return outputs

    def _row(
        self,
        request_id: str,
        origin: pd.Timestamp,
        source: pd.Series,
        horizon: int,
        recurrent_value: float,
        outputs: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        calibration = self.context.confidence.interval_calibration
        confidence_config = self.context.confidence.config
        recurrent_version = self.recurrent_summary["version"]
        volume_width, volume_level = interval_width(
            calibration, recurrent_version, "volume", horizon
        )
        speed_card = outputs["speed_card"]
        travel_card = outputs["travel_time_card"]
        speed = float(outputs["speed"][index])
        travel_time = float(outputs["travel_time"][index])
        speed_width, speed_level = interval_width(
            calibration, speed_card["model_version"], "speed", horizon
        )
        travel_width, travel_level = interval_width(
            calibration,
            travel_card["model_version"],
            "travel_time",
            horizon,
        )
        volume_lower, volume_upper = regression_interval(
            recurrent_value, volume_width
        )
        speed_lower, speed_upper = regression_interval(speed, speed_width)
        travel_lower, travel_upper = regression_interval(
            travel_time, travel_width
        )

        congestion_values = outputs["congestion"][index]
        congestion_confidence = probability_confidence(
            congestion_values, confidence_config
        )
        congestion_index = int(np.argmax(congestion_values))
        accident_values = outputs["accident"][index]
        accident_confidence = probability_confidence(
            accident_values, confidence_config
        )
        accident_card = outputs["accident_card"]
        threshold = float(accident_card["probability"]["operating_threshold"])
        accident_probability = float(accident_values[1])
        target_timestamp = origin + pd.Timedelta(minutes=30 * int(horizon))
        processed_sha = self.context.upstream_records["processed_dataset"]["sha256"]
        return {
            "request_id": request_id,
            "road_id": str(source["road_id"]),
            "road_name": str(source["road_name"]),
            "latitude": float(source["latitude"]),
            "longitude": float(source["longitude"]),
            "origin_timestamp": origin,
            "target_timestamp": target_timestamp,
            "horizon_windows": int(horizon),
            "horizon_minutes": int(horizon) * 30,
            "volume_prediction": float(recurrent_value),
            "volume_interval_lower": volume_lower,
            "volume_interval_upper": volume_upper,
            "volume_confidence_level": volume_level,
            "volume_model_version": recurrent_version,
            "volume_classical_comparator": float(outputs["volume"][index]),
            "volume_classical_model_version": outputs["volume_card"]["model_version"],
            "speed_prediction": speed,
            "speed_interval_lower": speed_lower,
            "speed_interval_upper": speed_upper,
            "speed_confidence_level": speed_level,
            "speed_model_version": speed_card["model_version"],
            "travel_time_prediction": travel_time,
            "travel_time_interval_lower": travel_lower,
            "travel_time_interval_upper": travel_upper,
            "travel_time_confidence_level": travel_level,
            "travel_time_model_version": travel_card["model_version"],
            "congestion_prediction": CONGESTION_LABELS[congestion_index],
            "congestion_probability_free_flow": float(congestion_values[0]),
            "congestion_probability_moderate": float(congestion_values[1]),
            "congestion_probability_heavy": float(congestion_values[2]),
            "congestion_probability_severe": float(congestion_values[3]),
            "congestion_max_probability": congestion_confidence[0],
            "congestion_entropy": congestion_confidence[1],
            "congestion_normalized_entropy": congestion_confidence[2],
            "congestion_confidence_band": congestion_confidence[3],
            "congestion_model_version": outputs["congestion_card"]["model_version"],
            "accident_probability": accident_probability,
            "no_accident_probability": float(accident_values[0]),
            "accident_prediction": bool(accident_probability >= threshold),
            "accident_operating_threshold": threshold,
            "accident_risk_band": accident_risk_band(
                accident_probability, threshold, confidence_config
            ),
            "accident_max_probability": accident_confidence[0],
            "accident_entropy": accident_confidence[1],
            "accident_normalized_entropy": accident_confidence[2],
            "accident_confidence_band": accident_confidence[3],
            "accident_model_version": accident_card["model_version"],
            "data_version": self.context.processed.summary["processed_version"],
            "feature_version": self.settings.feature_version,
            "preprocessing_version": self.context.modeling.summary["version"],
            "registry_version": self.context.registry["version"],
            "confidence_version": self.context.confidence.summary["version"],
            "inference_version": self.context.config["version"],
            "processed_data_sha256": processed_sha,
        }

    def predict(self, request: PredictionRequest) -> PredictionResult:
        """Produce all requested targets and confidence without fitting anything."""

        started = time.perf_counter()
        normalized, origin = self._validate_request(request)
        recurrent, _ = self._recurrent_predictions(normalized, origin)
        origins = select_origin_rows(
            self.context.processed.frame,
            normalized.road_ids,
            origin,
        )
        request_id = normalized.identifier(self.context.config["version"])
        rows: list[dict[str, Any]] = []
        for horizon in normalized.horizons:
            outputs = self._horizon_outputs(horizon, origins)
            for index, source in origins.iterrows():
                rows.append(
                    self._row(
                        request_id,
                        origin,
                        source,
                        horizon,
                        float(recurrent[index, horizon - 1]),
                        outputs,
                        index,
                    )
                )
        frame = validate_prediction_frame(pd.DataFrame(rows), normalized)
        lineage = {
            "configuration": {
                "path": str(self.context.config_path.relative_to(self.settings.root)),
            },
            "upstream": self.context.upstream_records,
            "models": {
                "recurrent_volume": {
                    "model_version": self.recurrent_summary["version"],
                    **self.recurrent_summary["model"],
                },
                **self.router.artifact_lineage(),
            },
        }
        return PredictionResult(
            request=normalized,
            frame=frame,
            lineage=lineage,
            initialization_seconds=self.initialization_seconds,
            prediction_seconds=time.perf_counter() - started,
        )
