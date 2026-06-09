"""Project constants for public menin inhibitor data collection."""

from __future__ import annotations

from dataclasses import dataclass

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBCHEM_PUG_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

MENIN_TARGET = {
    "name": "Menin",
    "gene": "MEN1",
    "chembl_id": "CHEMBL1615381",
    "uniprot": "O00255",
}

HERG_TARGET = {
    "name": "hERG / KCNH2",
    "gene": "KCNH2",
    "chembl_id": "CHEMBL240",
    "uniprot": "Q12809",
}

BIOACTIVITY_ENDPOINTS = ("IC50", "Ki", "Kd", "EC50")

PUBCHEM_SEARCH_TERMS = (
    "menin MLL inhibitor",
    "MEN1 menin inhibitor",
    "Menin-MLL Interaction Inhibitory Activity",
    '"menin/MLL" inhibitor',
)

PK_ADMET_KEYWORDS = (
    "adme",
    "auc",
    "bioavailability",
    "caco",
    "clearance",
    "clint",
    "cmax",
    "efflux",
    "half-life",
    "hepatocyte",
    "logd",
    "logp",
    "mdck",
    "metabolic stability",
    "microsomal",
    "pampa",
    "permeability",
    "pk",
    "plasma protein",
    "ppb",
    "solubility",
    "t1/2",
)

PK_ADMET_STANDARD_TYPES = (
    "AUC",
    "CL",
    "CLint",
    "Cmax",
    "F",
    "Fu",
    "HLM",
    "LogD",
    "LogP",
    "PPB",
    "Permeability",
    "Solubility",
    "T1/2",
    "Tmax",
)


@dataclass(frozen=True)
class BindingDBSource:
    name: str
    url: str
    target_hint: str


BINDINGDB_SOURCES = (
    BindingDBSource(
        name="bindingdb_menin",
        url="https://www.bindingdb.org/rwd/data/downloads/ptenK0/BDBpoly_6958.tsv",
        target_hint="Menin",
    ),
    BindingDBSource(
        name="bindingdb_menin_kmt2a_complex",
        url="https://www.bindingdb.org/rwd/data/downloads/ctenK0/BDBcomp_315.tsv",
        target_hint="Menin/KMT2A",
    ),
)
