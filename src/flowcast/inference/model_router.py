"""Frozen classical registry routing with verified source-model loading."""

from __future__ import annotations

from typing import Any

from flowcast.modelling.classical_artifacts import (
    load_classical_regression_model,
)
from flowcast.modelling.classification_artifacts import (
    load_classification_model,
)
from flowcast.settings import Settings


class FrozenModelRouter:
    """Resolve configured target/horizon routes without changing selections."""

    def __init__(
        self,
        settings: Settings,
        registry: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.config = config
        self._cache: dict[
            tuple[str, int],
            tuple[Any, dict[str, Any], dict[str, Any]],
        ] = {}

    def entry(self, target: str, horizon: int) -> dict[str, Any]:
        """Return the unique frozen registry entry for a target and horizon."""

        matches = [
            record
            for record in self.registry["entries"]
            if record["target"] == target
            and int(record["horizon_windows"]) == int(horizon)
        ]
        if len(matches) != 1:
            raise KeyError(f"Expected one registry entry for {target}_h{horizon}")
        return matches[0]

    def load(
        self,
        target: str,
        horizon: int,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """Load one model through its verified source loader and cache it."""

        key = (str(target), int(horizon))
        if key in self._cache:
            return self._cache[key]
        entry = self.entry(*key)
        source_version = str(entry["model_version"])
        expected_version = str(
            self.config["active_routing"][
                "volume" if target == "volume" else target
            ].get("model_version", source_version)
        )
        if target == "volume":
            expected_version = str(
                self.config["active_routing"]["volume"]["fallback"][
                    "model_version"
                ]
            )
        if source_version != expected_version:
            raise RuntimeError(f"Configured model version changed for {target}")
        if entry["source"] == "regression":
            estimator, card, _ = load_classical_regression_model(
                self.settings,
                target,
                horizon,
                version=source_version,
            )
        elif entry["source"] == "classification":
            estimator, card, _ = load_classification_model(
                self.settings,
                target,
                horizon,
                version=source_version,
            )
        else:
            raise RuntimeError(f"Unsupported registry source: {entry['source']}")
        if card["job_id"] != entry["job_id"]:
            raise RuntimeError("Loaded model identity differs from registry routing")
        if card["artifacts"]["model"] != entry["artifacts"]["model"]:
            raise RuntimeError("Loaded model artifact differs from registry routing")
        self._cache[key] = (estimator, card, entry)
        return self._cache[key]

    def artifact_lineage(self) -> dict[str, dict[str, Any]]:
        """Return model and card records for every model used so far."""

        records: dict[str, dict[str, Any]] = {}
        for (target, horizon), (_, card, entry) in sorted(self._cache.items()):
            records[f"{target}_h{horizon}"] = {
                "registry_key": entry["registry_key"],
                "model": entry["artifacts"]["model"],
                "model_card": entry["artifacts"]["model_card"],
                "predictions": entry["artifacts"]["predictions"],
                "model_version": card["model_version"],
            }
        return records
