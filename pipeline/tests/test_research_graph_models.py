from __future__ import annotations

import pandas as pd
import pytest
from menin_discovery.research_graph_models import (
    DirectedMPNN,
    grouped_conformer_mil_benchmark,
    grouped_dmpnn_herg_benchmark,
    molecule_graph,
)


def test_directed_graph_has_reverse_edges() -> None:
    graph = molecule_graph("CCO")
    assert len(graph.edge_sources) == 4
    assert graph.reverse_edges.tolist() == [1, 0, 3, 2]
    assert DirectedMPNN(hidden_size=16, depth=2)([graph]).shape == (1,)


def test_dmpnn_grouped_smoke() -> None:
    smiles = ["CC", "CCC", "CCCC", "CCO", "CCCO", "CCCCO", "CCN", "CCCN"]
    frame = pd.DataFrame(
        {
            "compound_id": [f"C{i}" for i in range(8)],
            "standardized_smiles": smiles,
            "scaffold": [f"S{i // 2}" for i in range(8)],
            "pic50_lower": [4.0, 4.2, 4.4, 4.6, 5.0, 5.2, 5.4, 5.6],
            "pic50_upper": [4.0, 4.2, 4.4, 4.6, 5.0, 5.2, 5.4, 5.6],
        }
    )
    metrics, predictions = grouped_dmpnn_herg_benchmark(frame, folds=2, epochs=2, hidden_size=16, depth=2)
    assert metrics["promotion_status"] == "discovery-track"
    assert len(predictions) == 8


def test_conformer_multiple_instance_smoke() -> None:
    compounds = pd.DataFrame(
        {
            "compound_id": [f"C{i}" for i in range(8)],
            "scaffold": [f"S{i // 2}" for i in range(8)],
            "pic50_lower": [4.0, 4.2, 4.4, 4.6, 5.0, 5.2, 5.4, 5.6],
            "pic50_upper": [4.0, 4.2, 4.4, 4.6, 5.0, 5.2, 5.4, 5.6],
        }
    )
    conformers = pd.DataFrame(
        [
            {
                "compound_id": compound_id,
                "ensemble_weight": 0.5,
                "radius_of_gyration_angstrom": 1.0 + conformer,
                "polar_sasa_ang2": 2.0 + conformer,
            }
            for compound_id in compounds["compound_id"]
            for conformer in (0, 1)
        ]
    )
    metrics, predictions = grouped_conformer_mil_benchmark(
        compounds,
        conformers,
        feature_columns=["radius_of_gyration_angstrom", "polar_sasa_ang2"],
        folds=2,
        epochs=2,
        hidden_size=8,
    )
    assert metrics["promotion_status"] == "discovery-track"
    assert len(predictions) == 8

    with pytest.raises(ValueError, match="not ontology-approved"):
        grouped_conformer_mil_benchmark(
            compounds,
            conformers.assign(unreviewed_numeric=1.0),
            feature_columns=["unreviewed_numeric"],
            folds=2,
            epochs=1,
            hidden_size=8,
        )
