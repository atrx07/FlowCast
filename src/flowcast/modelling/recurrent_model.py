"""From-scratch PyTorch recurrent volume forecaster."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from flowcast.modelling.recurrent_config import RecurrentCandidate


class RecurrentVolumeForecaster(nn.Module):
    """LSTM/GRU encoder with dropout and a four-horizon regression head."""

    def __init__(
        self,
        input_size: int,
        candidate: RecurrentCandidate,
        output_size: int = 4,
    ) -> None:
        super().__init__()
        recurrent_class: type[nn.RNNBase]
        if candidate.recurrent_type == "lstm":
            recurrent_class = nn.LSTM
        elif candidate.recurrent_type == "gru":
            recurrent_class = nn.GRU
        else:
            raise ValueError(
                f"Unsupported recurrent type: {candidate.recurrent_type}"
            )
        self.recurrent = recurrent_class(
            input_size=int(input_size),
            hidden_size=candidate.hidden_size,
            num_layers=candidate.layer_count,
            dropout=(
                candidate.recurrent_dropout
                if candidate.layer_count > 1
                else 0.0
            ),
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(candidate.head_dropout),
            nn.Linear(candidate.hidden_size, candidate.head_hidden_size),
            nn.ReLU(),
            nn.Linear(candidate.head_hidden_size, int(output_size)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forecast four standardized volume targets from one sequence batch."""

        if inputs.ndim != 3:
            raise ValueError("Recurrent input must have [batch, sequence, feature]")
        encoded, _ = self.recurrent(inputs)
        return self.head(encoded[:, -1, :])


def architecture_metadata(
    model: RecurrentVolumeForecaster,
    candidate: RecurrentCandidate,
    input_size: int,
) -> dict[str, Any]:
    """Return reconstructable architecture and parameter-count evidence."""

    return {
        "recurrent_type": candidate.recurrent_type,
        "input_size": int(input_size),
        "sequence_length": candidate.sequence_length,
        "hidden_size": candidate.hidden_size,
        "layer_count": candidate.layer_count,
        "recurrent_dropout": candidate.recurrent_dropout,
        "head_hidden_size": candidate.head_hidden_size,
        "head_dropout": candidate.head_dropout,
        "output_size": 4,
        "bidirectional": False,
        "pretrained_weights": False,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }


def candidate_from_architecture(record: dict[str, Any]) -> RecurrentCandidate:
    """Reconstruct the selected candidate from checkpoint metadata."""

    return RecurrentCandidate(
        candidate_id=str(record["candidate_id"]),
        recurrent_type=str(record["recurrent_type"]),
        sequence_length=int(record["sequence_length"]),
        hidden_size=int(record["hidden_size"]),
        layer_count=int(record["layer_count"]),
        recurrent_dropout=float(record["recurrent_dropout"]),
        head_hidden_size=int(record["head_hidden_size"]),
        head_dropout=float(record["head_dropout"]),
        batch_size=int(record["batch_size"]),
        learning_rate=float(record["learning_rate"]),
        weight_decay=float(record["weight_decay"]),
    )
