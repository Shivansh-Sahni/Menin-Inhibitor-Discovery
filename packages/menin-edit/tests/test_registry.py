import pytest

from menin_edit.registry import PredictorRegistry
from menin_edit.schemas import PropertyEstimate


class _CountingPredictor:
    def __init__(self, endpoint: str, offset: float = 0.0) -> None:
        self.endpoint = endpoint
        self.offset = offset
        self.single_calls: list[str] = []
        self.batch_calls: list[tuple[str, ...]] = []

    @property
    def model_version(self) -> str:
        return f"{self.endpoint}-fixture-v1"

    def _estimate(self, smiles: str) -> PropertyEstimate:
        mean = self.offset + len(smiles) / 100
        return PropertyEstimate(
            endpoint=self.endpoint,
            mean=mean,
            lower=mean,
            upper=mean,
            inside_domain=True,
            model_version=self.model_version,
        )

    def predict(self, smiles: str) -> PropertyEstimate:
        self.single_calls.append(smiles)
        return self._estimate(smiles)

    def predict_many(self, smiles):
        values = tuple(smiles)
        self.batch_calls.append(values)
        return [self._estimate(value) for value in values]


@pytest.mark.parametrize(
    ("enabled_configuration", "expected_registered"),
    (
        ({}, False),
        ({"enabled": "false"}, False),
        ({"enabled": False}, False),
        ({"enabled": True}, True),
    ),
    ids=("missing", "string-false", "boolean-false", "explicit-true"),
)
def test_private_ensemble_requires_explicit_boolean_true(
    enabled_configuration,
    expected_registered,
):
    definition = {
        "kind": "private_herg_ensemble",
        "benchmark_root": "/governed/private/herg",
        **enabled_configuration,
    }

    registry = PredictorRegistry.from_config(
        {"herg_private_ensemble_probability": definition},
        repository_root=".",
    )

    assert ("herg_private_ensemble_probability" in registry.predictors) is expected_registered


def test_batch_registry_deduplicates_preserves_order_and_reuses_cache():
    potency = _CountingPredictor("potency")
    herg = _CountingPredictor("herg", offset=0.2)
    registry = PredictorRegistry({"potency": potency, "herg": herg})

    rows = registry.predict_all_many(["CC", "CCC", "CC", "CCO"])

    assert potency.batch_calls == [("CC", "CCC", "CCO")]
    assert herg.batch_calls == [("CC", "CCC", "CCO")]
    assert rows[0] == rows[2]
    assert [set(row) for row in rows] == [{"potency", "herg"}] * 4
    assert registry.cache_size == 6

    assert registry.predict("CC", "potency") is rows[0]["potency"]
    cached_rows = registry.predict_all_many(["CCO", "CCC"])
    assert cached_rows[0]["herg"] is rows[3]["herg"]
    assert potency.batch_calls == [("CC", "CCC", "CCO")]
    assert herg.batch_calls == [("CC", "CCC", "CCO")]
    assert not potency.single_calls and not herg.single_calls


def test_batch_registry_only_scores_new_structures_on_later_calls():
    predictor = _CountingPredictor("potency")
    registry = PredictorRegistry({"potency": predictor})

    registry.predict_all_many(["CC", "CCC"])
    registry.predict_all_many(["CCC", "CCCC", "CC"])

    assert predictor.batch_calls == [("CC", "CCC"), ("CCCC",)]
    assert registry.model_versions() == {"potency": "potency-fixture-v1"}
