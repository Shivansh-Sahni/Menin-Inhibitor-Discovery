import tempfile
from pathlib import Path

from menin_discovery.pubchem import load_pubchem_assay_csvs


def test_pubchem_loader_does_not_invent_missing_standard_value_units():
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory) / "assay_data"
        data_dir.mkdir()
        (data_dir / "AID_1.csv").write_text(
            "PUBCHEM_RESULT_TAG,PubChem Standard Value\n1,2.5\n",
            encoding="utf-8",
        )
        loaded = load_pubchem_assay_csvs(Path(directory))
        assert loaded.loc[0, "PubChem Standard Value Units"] == ""


def test_pubchem_loader_preserves_explicit_result_units():
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory) / "assay_data"
        data_dir.mkdir()
        (data_dir / "AID_2.csv").write_text(
            "PUBCHEM_RESULT_TAG,PubChem Standard Value\nRESULT_UNIT,MICROMOLAR\n1,2.5\n",
            encoding="utf-8",
        )
        loaded = load_pubchem_assay_csvs(Path(directory))
        assert loaded.loc[0, "PubChem Standard Value Units"] == "MICROMOLAR"
