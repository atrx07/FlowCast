"""Seeded training, early stopping, and inference for recurrent candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from flowcast.evaluation.regression import regression_metrics
from flowcast.modelling.recurrent_config import RecurrentCandidate
from flowcast.modelling.recurrent_model import (
    RecurrentVolumeForecaster,
    architecture_metadata,
)
from flowcast.modelling.sequence_data import (
    PreparedPartition,
    RecurrentSequenceDataset,
    TargetScaler,
)


@dataclass(frozen=True)
class CandidateTrainingResult:
    """Best restored state and validation evidence for one candidate."""

    candidate: RecurrentCandidate
    architecture: dict[str, Any]
    history: list[dict[str, Any]]
    best_epoch: int
    stopped_epoch: int
    early_stopped: bool
    best_validation_mean_rmse: float
    validation_metrics: list[dict[str, Any]]
    validation_predictions: np.ndarray
    best_state: dict[str, torch.Tensor]
    fit_seconds: float
    prediction_seconds: float
    device: str


def seed_torch(seed: int, deterministic: bool, cpu_threads: int) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic execution."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(cpu_threads)))
    torch.use_deterministic_algorithms(bool(deterministic))


def select_device(policy: str) -> torch.device:
    """Resolve the configured CPU/CUDA policy without promising unavailable CUDA."""

    normalized = str(policy).lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported recurrent device policy: {policy}")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not available to PyTorch")
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(
    dataset: RecurrentSequenceDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )


def _predict_scaled(
    model: RecurrentVolumeForecaster,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for features, _ in loader:
            outputs = model(features.to(device, non_blocking=device.type == "cuda"))
            batches.append(outputs.detach().cpu().numpy())
    return np.concatenate(batches, axis=0), time.perf_counter() - started


def horizon_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    """Return per-horizon regression metrics and their mean RMSE."""

    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Multi-horizon actual and predicted matrices must align")
    records = []
    for offset in range(actual.shape[1]):
        metrics = regression_metrics(actual[:, offset], predicted[:, offset])
        records.append({"horizon_windows": offset + 1, **metrics})
    return records, float(np.mean([record["rmse"] for record in records]))


def train_candidate(
    candidate: RecurrentCandidate,
    training: PreparedPartition,
    validation: PreparedPartition,
    training_endpoints: np.ndarray,
    validation_endpoints: np.ndarray,
    scaler: TargetScaler,
    config: dict[str, Any],
    seed: int,
) -> CandidateTrainingResult:
    """Train one candidate and restore its validation-best state."""

    device_config = config["device"]
    seed_torch(
        seed,
        bool(device_config["deterministic_algorithms"]),
        int(device_config["cpu_threads"]),
    )
    device = select_device(str(device_config["policy"]))
    scaled_training = scaler.transform(
        training.frame[list(training.target_columns)].to_numpy(dtype=float)
    )
    scaled_validation = scaler.transform(
        validation.frame[list(validation.target_columns)].to_numpy(dtype=float)
    )
    training_dataset = RecurrentSequenceDataset(
        training.features,
        scaled_training,
        training_endpoints,
        candidate.sequence_length,
    )
    validation_dataset = RecurrentSequenceDataset(
        validation.features,
        scaled_validation,
        validation_endpoints,
        candidate.sequence_length,
    )
    workers = int(device_config["dataloader_workers"])
    training_loader = _loader(
        training_dataset,
        candidate.batch_size,
        workers,
        shuffle=True,
        seed=seed,
        device=device,
    )
    validation_loader = _loader(
        validation_dataset,
        candidate.batch_size,
        workers,
        shuffle=False,
        seed=seed,
        device=device,
    )
    model = RecurrentVolumeForecaster(
        training.features.shape[1],
        candidate,
    ).to(device)
    architecture = {
        "candidate_id": candidate.candidate_id,
        **asdict(candidate),
        **architecture_metadata(model, candidate, training.features.shape[1]),
    }
    training_config = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    scheduler_config = training_config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["minimum_learning_rate"]),
    )
    criterion = nn.MSELoss()
    maximum_epochs = int(training_config["maximum_epochs"])
    minimum_epochs = int(training_config["minimum_epochs"])
    stopping = training_config["early_stopping"]
    patience = int(stopping["patience"])
    minimum_improvement = float(stopping["minimum_improvement"])
    gradient_clip = float(training_config["gradient_clip_norm"])
    best_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    validation_actual = validation.frame.iloc[validation_endpoints][
        list(validation.target_columns)
    ].to_numpy(dtype=float)
    latest_predictions = np.empty_like(validation_actual)
    prediction_seconds = 0.0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for features, targets in training_loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        scaled_predictions, elapsed = _predict_scaled(
            model,
            validation_loader,
            device,
        )
        prediction_seconds += elapsed
        latest_predictions = scaler.inverse_transform(scaled_predictions)
        metrics, mean_rmse = horizon_metrics(
            validation_actual,
            latest_predictions,
        )
        scheduler.step(mean_rmse)
        improved = mean_rmse < best_metric - minimum_improvement
        if improved:
            best_metric = mean_rmse
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        history.append(
            {
                "candidate_id": candidate.candidate_id,
                "epoch": epoch,
                "training_scaled_mse": float(np.mean(batch_losses)),
                "validation_mean_rmse": mean_rmse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_epoch_so_far": best_epoch,
                "improved": improved,
            }
        )
        if epoch >= minimum_epochs and stale_epochs >= patience:
            break
    if not best_state:
        raise RuntimeError("Recurrent training never produced a best checkpoint")
    model.load_state_dict(best_state)
    best_scaled, elapsed = _predict_scaled(model, validation_loader, device)
    prediction_seconds += elapsed
    best_predictions = scaler.inverse_transform(best_scaled)
    validation_metrics, restored_metric = horizon_metrics(
        validation_actual,
        best_predictions,
    )
    if not np.isclose(restored_metric, best_metric, rtol=1e-7, atol=1e-7):
        raise RuntimeError("Best-weight restoration changed validation score")
    return CandidateTrainingResult(
        candidate=candidate,
        architecture=architecture,
        history=history,
        best_epoch=best_epoch,
        stopped_epoch=len(history),
        early_stopped=len(history) < maximum_epochs,
        best_validation_mean_rmse=best_metric,
        validation_metrics=validation_metrics,
        validation_predictions=best_predictions,
        best_state=best_state,
        fit_seconds=time.perf_counter() - started,
        prediction_seconds=prediction_seconds,
        device=str(device),
    )


def predict_partition(
    model: RecurrentVolumeForecaster,
    partition: PreparedPartition,
    endpoints: np.ndarray,
    sequence_length: int,
    scaler: TargetScaler,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Run deterministic restored-checkpoint inference for one partition."""

    scaled = scaler.transform(
        partition.frame[list(partition.target_columns)].to_numpy(dtype=float)
    )
    dataset = RecurrentSequenceDataset(
        partition.features,
        scaled,
        endpoints,
        sequence_length,
    )
    loader = _loader(
        dataset,
        batch_size,
        workers,
        shuffle=False,
        seed=seed,
        device=device,
    )
    predictions, elapsed = _predict_scaled(model, loader, device)
    return scaler.inverse_transform(predictions), elapsed
