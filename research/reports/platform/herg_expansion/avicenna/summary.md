# PubChem hERG expansion: Avicenna workstream

## Result

The priority official assay, **PubChem AID 588834**, was acquired as its complete depositor assay
CSV plus description, summary, SID list, and CID list. It is the NCATS/NCGC **HERG01** confirmatory
qHTS thallium-flux assay targeting human KCNH2/hERG.

- Full result rows: **5,381**
- Unique SIDs: **5,381**
- Unique mapped CIDs: **4,743**
- Unmapped-CID rows: **18**
- Outcome counts: **{"Active": 664, "Inactive": 4107, "Inconclusive": 610}**
- Direct SID/CID endpoints reconcile with the full CSV: **True / True**

Official Entrez same-project links define the separate KCNH2 3.1 family as summary **AID 720544**,
wildtype **AID 720551**, and mutant **AID 720553**. Concise outcome exports were retained for the
two experiment assays: **343,909** wildtype rows and
**342,311** mutant rows. They are not interchangeable with
AID 588834 and were not pooled.

## Provenance and rights

All artifacts came from official PubChem PUG-REST or NCBI Entrez endpoints. The manifest records
URL, HTTP date/status/type, retrieval time, byte size, and SHA-256. PubChem preserves the assay's
depositor identity and external ID. NCBI states that it adds no restriction to molecular data but
cannot transfer contributor rights; no explicit depositor licence appeared in the retained NCATS/
NCGC source page. Therefore these remain **pre-canonical conditional-use artifacts**: zero canonical
rows and zero training labels were written.

## Overlap readiness

AID 588834 is ready for SID- and CID-level overlap analysis against ChEMBL or other inventories.
The source fields also preserve PubChem activity outcome/score and assay-specific quantitative
columns. Structure standardization, duplicate adjudication, cross-assay aggregation, and model use
remain deliberately deferred.

Exact pairwise SID/CID and active-subset intersections among the three experimental assays are
reported in `within_pubchem_overlap.json`; these are identity-overlap diagnostics, not pooled labels.
