"""Configuration-driven endpoint predictor registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .local_models import LocalLabRegressionPredictor
from .predictors import (
    ConservativeHergConsensusPredictor,
    GovernedSkopsClassifierPredictor,
    Predictor,
    PrivateQuickHergEnsemblePredictor,
    PublicHergSkopsPredictor,
    PublicMeninSkopsPredictor,
    StructuralAlertProxyPredictor,
)
from .schemas import PropertyEstimate


class PredictorRegistry:
    def __init__(self, predictors: Mapping[str, Predictor]) -> None:
        self.predictors = dict(predictors)
        for key, predictor in self.predictors.items():
            if key != predictor.endpoint:
                raise ValueError(f"Predictor key {key!r} does not match endpoint {predictor.endpoint!r}")
        self._cache: dict[tuple[str, str], PropertyEstimate] = {}

    @classmethod
    def from_config(
        cls,
        model_configs: Mapping[str, Mapping[str, Any]],
        *,
        repository_root: str | Path,
    ) -> PredictorRegistry:
        predictors: dict[str, Predictor] = {}
        consensus_definitions: list[tuple[str, dict[str, Any]]] = []
        predictor: Predictor
        for key, raw_definition in model_configs.items():
            definition = dict(raw_definition)
            kind = str(definition.get("kind", "")).strip()
            if kind == "private_herg_ensemble":
                # Private artifacts are a governed opt-in: only a literal YAML
                # boolean ``true`` may activate them.  Missing values and
                # truthy strings such as ``"false"`` must fail closed.
                if definition.get("enabled") is not True:
                    continue
            elif not bool(definition.get("enabled", True)):
                continue
            if kind == "public_menin_skops":
                predictor = PublicMeninSkopsPredictor(
                    artifact=definition.get("artifact"),
                    manifest=definition.get("manifest"),
                    metrics=definition.get("metrics"),
                    domain_reference=definition.get("domain_reference"),
                    domain_threshold=definition.get("domain_threshold"),
                    interval_half_width=definition.get("interval_half_width"),
                    repository_root=repository_root,
                )
            elif kind == "public_herg_skops":
                predictor = PublicHergSkopsPredictor(
                    artifact=definition.get("artifact"),
                    manifest=definition.get("manifest"),
                    metrics=definition.get("metrics"),
                    domain_reference=definition.get("domain_reference"),
                    domain_threshold=definition.get("domain_threshold"),
                    repository_root=repository_root,
                )
            elif kind == "private_herg_ensemble":
                predictor = PrivateQuickHergEnsemblePredictor(
                    benchmark_root=definition.get("benchmark_root", "research/benchmarks/herg/quick"),
                    regime=definition.get("regime", "equal_importance"),
                    domain_reference=definition.get("domain_reference"),
                    domain_threshold=definition.get("domain_threshold"),
                    domain_quantile=float(definition.get("domain_quantile", 0.05)),
                    repository_root=repository_root,
                )
            elif kind == "structural_alert_proxy":
                predictor = StructuralAlertProxyPredictor(
                    catalogs=tuple(definition.get("catalogs", ("PAINS", "BRENK", "NIH")))
                )
            elif kind == "local_lab_regression":
                local_predictor = LocalLabRegressionPredictor(
                    definition["artifact"],
                    manifest=definition.get("manifest"),
                    verify_hash=bool(definition.get("verify_hash", True)),
                )
                if not local_predictor.recommended_for_optimization and not bool(
                    definition.get("allow_not_recommended", False)
                ):
                    raise ValueError(
                        f"Local model {key!r} did not beat its scaffold-fold baseline; "
                        "set allow_not_recommended only for explicit sensitivity analysis"
                    )
                predictor = local_predictor
            elif kind == "conservative_consensus":
                consensus_definitions.append((key, definition))
                continue
            elif kind in {"external_classifier", "governed_skops_classifier"}:
                required = ("artifact", "manifest", "domain_reference", "domain_threshold")
                missing = [name for name in required if definition.get(name) in {None, ""}]
                if missing:
                    raise ValueError(f"Governed classifier {key!r} is missing configuration: {missing}")
                predictor = GovernedSkopsClassifierPredictor(
                    endpoint=key,
                    artifact=definition["artifact"],
                    manifest=definition["manifest"],
                    metrics=definition.get("metrics"),
                    domain_reference=definition["domain_reference"],
                    domain_threshold=float(definition["domain_threshold"]),
                    positive_class_index=int(definition.get("positive_class_index", 1)),
                    endpoint_semantics=str(
                        definition.get(
                            "endpoint_semantics",
                            "project-defined positive-class probability",
                        )
                    ),
                    verify_hash=bool(definition.get("verify_hash", True)),
                    repository_root=repository_root,
                )
            else:
                raise ValueError(f"Unsupported predictor kind for {key!r}: {kind!r}")
            if predictor.endpoint != key:
                raise ValueError(
                    f"Configured endpoint {key!r} does not match predictor endpoint {predictor.endpoint!r}"
                )
            predictors[key] = predictor

        for key, definition in consensus_definitions:
            members = [predictors[name] for name in definition.get("members", []) if name in predictors]
            missing = [name for name in definition.get("members", []) if name not in predictors]
            if missing:
                raise KeyError(f"Consensus endpoint {key!r} has unavailable members: {missing}")
            consensus_predictor = ConservativeHergConsensusPredictor(members)
            if consensus_predictor.endpoint != key:
                raise ValueError(
                    f"Configured consensus {key!r} does not match {consensus_predictor.endpoint!r}"
                )
            predictors[key] = consensus_predictor
        return cls(predictors)

    def predict(self, smiles: str, endpoint: str) -> PropertyEstimate:
        key = (endpoint, smiles)
        if key not in self._cache:
            if endpoint not in self.predictors:
                raise KeyError(f"No predictor is registered for endpoint {endpoint!r}")
            self._cache[key] = self.predictors[endpoint].predict(smiles)
        return self._cache[key]

    def predict_all(self, smiles: str) -> dict[str, PropertyEstimate]:
        return {endpoint: self.predict(smiles, endpoint) for endpoint in self.predictors}

    def predict_all_many(self, smiles: Sequence[str]) -> list[dict[str, PropertyEstimate]]:
        """Score a batch once per endpoint and populate the shared molecule cache.

        Molecular editing creates tens of candidates at each depth.  Calling a
        predictor once per candidate is needlessly expensive, especially for
        fingerprint generation and the isolated native hERG ensemble member.
        This method preserves the same cache contract while vectorizing every
        endpoint over the unique, previously unseen structures.
        """

        ordered = [str(value) for value in smiles]
        if not ordered:
            return []
        unique = tuple(dict.fromkeys(ordered))
        for endpoint, predictor in self.predictors.items():
            missing = [value for value in unique if (endpoint, value) not in self._cache]
            if not missing:
                continue
            estimates = predictor.predict_many(missing)
            if len(estimates) != len(missing):
                raise RuntimeError(
                    f"Predictor {endpoint!r} returned {len(estimates)} estimates for {len(missing)} molecules"
                )
            for value, estimate in zip(missing, estimates, strict=True):
                if estimate.endpoint != endpoint:
                    raise RuntimeError(f"Predictor {endpoint!r} returned endpoint {estimate.endpoint!r}")
                self._cache[(endpoint, value)] = estimate
        return [
            {endpoint: self._cache[(endpoint, value)] for endpoint in self.predictors} for value in ordered
        ]

    def model_versions(self) -> dict[str, str]:
        return {endpoint: predictor.model_version for endpoint, predictor in self.predictors.items()}

    @property
    def cache_size(self) -> int:
        return len(self._cache)
