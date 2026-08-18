from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest

from menin_edit.predictors import (
    ArtifactVerificationError,
    ConservativeHergConsensusPredictor,
    GovernedSkopsClassifierPredictor,
    Predictor,
    PrivateQuickHergEnsemblePredictor,
    PublicHergSkopsPredictor,
    PublicMeninSkopsPredictor,
    StructuralAlertProxyPredictor,
)
from menin_edit.schemas import PropertyEstimate

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _estimate(
    endpoint: str,
    mean: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    inside_domain: bool = True,
) -> PropertyEstimate:
    kwargs = {
        "endpoint": endpoint,
        "mean": mean,
        "lower": mean if lower is None else lower,
        "upper": mean if upper is None else upper,
        "inside_domain": inside_domain,
        "model_version": f"{endpoint}-v1",
        "evidence_status": "test",
    }
    if "metadata" in {field.name for field in fields(PropertyEstimate)}:
        kwargs["metadata"] = {"fixture": True}
    return PropertyEstimate(**kwargs)


class _FixedPredictor:
    def __init__(self, estimate: PropertyEstimate):
        self.estimate = estimate
        self.endpoint = estimate.endpoint

    @property
    def model_version(self) -> str:
        return self.estimate.model_version

    def predict(self, smiles: str) -> PropertyEstimate:
        return self.estimate

    def predict_many(self, smiles):
        return [self.estimate for _ in smiles]


def test_predictor_protocol_and_lazy_public_loading():
    predictor = PublicMeninSkopsPredictor(repository_root=REPOSITORY_ROOT)
    assert predictor._model is None
    # Runtime protocol inspection can evaluate the model_version property on
    # some Python releases, so verify laziness before performing that check.
    assert isinstance(predictor, Predictor)


def test_public_menin_and_herg_artifacts_predict_with_lineage():
    menin = PublicMeninSkopsPredictor(repository_root=REPOSITORY_ROOT)
    herg = PublicHergSkopsPredictor(repository_root=REPOSITORY_ROOT)

    menin_estimate = menin.predict("CCN1CCCCC1")
    herg_estimate = herg.predict("CCN1CCCCC1")

    assert menin_estimate.endpoint == "menin_biochemical_pIC50"
    assert menin_estimate.lower <= menin_estimate.mean <= menin_estimate.upper
    assert menin_estimate.model_version.startswith("sha256:")
    assert herg_estimate.endpoint == "herg_public_blocker_probability"
    assert 0 <= herg_estimate.mean <= 1
    assert herg_estimate.lower == herg_estimate.mean == herg_estimate.upper
    if hasattr(menin_estimate, "metadata"):
        assert menin_estimate.metadata["artifact_hash_verified"] is True
        assert herg_estimate.metadata["not_clinical_cardiotoxicity_probability"] is True


def test_public_adapter_rejects_artifact_hash_mismatch(tmp_path):
    artifact = tmp_path / "model.skops"
    artifact.write_bytes(b"not a model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact": {
                    "path": str(artifact),
                    "format": "skops",
                    "sha256": hashlib.sha256(b"different").hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "reference.csv"
    reference.write_text("smiles\nCC\n", encoding="utf-8")
    predictor = PublicMeninSkopsPredictor(
        repository_root=tmp_path,
        artifact=artifact,
        manifest=manifest,
        domain_reference=reference,
        domain_threshold=0.3,
    )

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        predictor.predict("CC")


def test_private_quick_ensemble_uses_saved_models_and_reports_disagreement():
    benchmark_root = REPOSITORY_ROOT / "research/benchmarks/herg/quick"
    if not (benchmark_root / "run_manifest.json").is_file():
        pytest.skip("Private Menin-series hERG benchmark is intentionally absent from the public release")
    predictor = PrivateQuickHergEnsemblePredictor(repository_root=REPOSITORY_ROOT)
    assert predictor._members is None
    estimate = predictor.predict("CCN1CCCCC1")

    assert estimate.endpoint == "herg_private_ensemble_probability"
    assert 0 <= estimate.lower <= estimate.mean <= estimate.upper <= 1
    assert estimate.model_version.startswith("quick-equal_importance-sha256:")
    if hasattr(estimate, "metadata"):
        assert len(estimate.metadata["members"]) == 3
        assert len(estimate.metadata["member_probabilities"]) == 3
        assert estimate.metadata["quick_benchmark_is_development_evidence"] is True


def test_conservative_consensus_uses_worst_probability_and_domain_conjunction():
    public = _FixedPredictor(
        _estimate(
            "herg_public_blocker_probability",
            0.35,
            lower=0.30,
            upper=0.40,
            inside_domain=True,
        )
    )
    private = _FixedPredictor(
        _estimate(
            "herg_private_ensemble_probability",
            0.70,
            lower=0.60,
            upper=0.80,
            inside_domain=False,
        )
    )
    predictor = ConservativeHergConsensusPredictor([public, private])
    estimate = predictor.predict("CC")

    assert estimate.endpoint == "herg_consensus_probability"
    assert estimate.mean == pytest.approx(0.70)
    assert estimate.lower == pytest.approx(0.30)
    assert estimate.upper == pytest.approx(0.80)
    assert estimate.inside_domain is False
    assert "contains_out_of_domain" in estimate.evidence_status


def test_structural_alert_output_is_explicitly_a_non_toxicity_proxy():
    predictor = StructuralAlertProxyPredictor(catalogs=("PAINS", "BRENK"))
    estimate = predictor.predict("O=C1NC(=S)SC1")

    assert estimate.endpoint == "structural_alert_count"
    assert estimate.mean >= 1
    assert estimate.lower == estimate.mean == estimate.upper
    assert estimate.inside_domain is True
    assert "not_toxicity" in estimate.evidence_status
    if hasattr(estimate, "metadata"):
        assert estimate.metadata["proxy_only"] is True
        assert estimate.metadata["not_a_toxicity_prediction"] is True


def test_predictors_reject_invalid_smiles():
    with pytest.raises(ValueError, match="Invalid molecular structure"):
        StructuralAlertProxyPredictor().predict("not a smiles")


def test_governed_classifier_adapter_supports_named_future_endpoints():
    predictor = GovernedSkopsClassifierPredictor(
        endpoint="test_named_toxicity_probability",
        endpoint_semantics="test fixture positive class",
        artifact=REPOSITORY_ROOT / "research" / "models" / "herg_liability_extra_trees_calibrated.skops",
        manifest=REPOSITORY_ROOT / "research" / "models" / "herg_classifier_manifest.json",
        domain_reference=REPOSITORY_ROOT / "research" / "reports" / "herg_classifier_split_assignments.csv",
        domain_threshold=0.2535,
        repository_root=REPOSITORY_ROOT,
    )

    estimate = predictor.predict("CCN1CCCCC1")

    assert estimate.endpoint == "test_named_toxicity_probability"
    assert 0 <= estimate.mean <= 1
    assert estimate.lower == estimate.mean == estimate.upper
    assert estimate.metadata["endpoint_semantics"] == "test fixture positive class"
    assert estimate.metadata["requires_endpoint_specific_validation"] is True
