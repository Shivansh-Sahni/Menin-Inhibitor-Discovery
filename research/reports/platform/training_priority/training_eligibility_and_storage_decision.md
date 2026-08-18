# Training eligibility, hERG/PK priorities, and storage decision

Date: 2026-08-07  
Status: lead-reconciled; no substantive training authorized or started

## Why real observations can still be ineligible

Training eligibility is task- and claim-specific. A record can be scientifically useful but unsafe as a supervised label when any of the following applies:

- it is assay metadata, a well, control, concentration, normalized response, curve component, document section, or derived view rather than one resolved endpoint measurement;
- molecule structure, parent identity, target/protein/construct, endpoint, unit, direction, relation, or provenance is unresolved;
- assay quantities are incompatible: patch-clamp inhibition, radioligand binding, thallium flux, IC50, AC50, and binary hit calls are related but not interchangeable;
- a value is censored or ranged (`>`, `<`, interval); converting the boundary to an exact value creates a false label;
- essential context is missing. IC50 depends on protocol and concentration conditions; PK depends on species, route, dose, formulation, analyte, matrix, population, and time basis;
- biology differs: wild-type versus mutant hERG, blocker versus activator/protective direction, in-vitro hERG versus QT/QTc or torsade outcomes;
- a value is computed rather than measured. pActivity and standard free energy are useful transformations, not independent evidence;
- source rights or record-level provenance are insufficient;
- chemical-neighbor, protein-homology, assay, source, or temporal leakage prevents the intended evaluation claim;
- the training population is severely selected or imbalanced. This can support auxiliary learning but not an unqualified primary endpoint claim.

## Strict hERG versus EPA hERG

| Property | Strict canonical hERG | EPA hERG extraction |
| --- | ---: | ---: |
| Scale | 137 rows / 104 molecules | 11,030 result rows / 7,959 DTXSIDs |
| Label meaning | Exact functional IC50 in nM under the canonical task contract | Mixed MC/SC screening output from three assays |
| Modality | Narrow functional potency | Tox21 thallium-flux proxy plus NVS/ERF radioligand binding |
| Independence | ChEMBL-derived canonical task | 9,670 rows are the same Tox21 experiment already represented by PubChem |
| Actually new EPA physical results | Not applicable | 1,219 NVS + 141 ERF = 1,360 rows |
| Chemicals outside EPA Tox21 | Not applicable | 88 DTXSIDs; cross-source novelty still unresolved |
| Correct role | Engineering smoke test | Auxiliary binding/proxy tasks and QC after identity reconciliation |

The strict set is small because it is the intersection of exact relation, functional IC50, nM units, accepted source status, wild-type target, resolved molecule/protein, and the declared task contract. EPA is small because the >500-million-record database covers thousands of unrelated assays and stores multiple relational processing layers. Only three endpoints map to human KCNH2; the rows are not patch-clamp IC50 measurements.

## What is actually helpful

| Priority | Evidence | Correct model role | Decision |
| ---: | --- | --- | --- |
| 1 | Quantitative three-channel hERG, 22,969 pIC50 rows | hERG-specific quantitative regression | Highest-priority hERG admission candidate; reconcile provenance, technology, units, and overlap first |
| 2 | ChEMBL hERG, 41,078 rows | Separate patch-clamp, functional, binding, and fixed-dose tasks | Retain as provenance backbone; do not pool modalities |
| 3 | PubChem AID 720551, 343,909 rows | Fixed-dose/high-throughput auxiliary task | Useful chemical coverage and negative-screen evidence, not IC50 truth |
| 4 | HERGAI, 299,927 rows | Auxiliary binary/ranking pretraining | Secondary only because of extreme imbalance, sparse potency, and provenance/rights limitations |
| 5 | EPA NVS/ERF, 1,360 rows | Separate hERG-binding auxiliary tasks | Retain; do not merge directly into functional liability labels |
| 6 | Mutant hERG, 342,311 rows | Variant-aware structure/function task | Separate from general wild-type liability |
| 1 affinity | Exact Ki, 407,926 rows | Protein-conditioned cross-protein affinity | Best first powered affinity model |
| Pilot affinity | Exact Kd, 62,334 rows | Controlled protein-conditioned endpoint pilot | Use to establish a rigorous protocol before scaling |
| Large affinity | Exact IC50, 1,107,075 rows | Protein- and assay-conditioned potency | Best million-row benchmark; never relabel as Kd/Ki |
| PK curation | DailyMed, 43,162 candidate sections | Human endpoint/context extraction | Highest-priority PK curation source; currently zero admitted values |
| PK/ADME now | 274,168 upstream rows + 330,261 ChEMBL candidates | Separate solubility, permeability, PPB, clearance, CYP/transporter, and stability tasks | Useful after endpoint-specific normalization; do not call all of it systemic PK |
| Structured PK | PK-DB | Context-rich systemic PK | Potentially very valuable, but access/rights remain blocked |

### Model scope

- **hERG:** primarily one-protein chemical modeling. Protein sequence adds little for wild-type-only data; assay modality matters greatly. Mutants require variant/sequence input and a separate task.
- **PK:** not one protein and not one endpoint. Build separate CL, half-life, Vd/Vdss, F, AUC, Cmax, and Tmax models conditioned on route, dose, formulation, matrix, analyte, and population. CYP/transporter tasks can be protein-specific.
- **Affinity:** cross-protein models require protein/target input and endpoint identity. Per-protein models are appropriate where target support is strong. IC50 additionally requires assay context.

## Storage action

### Deleted immediately

- `invitrodb_v4_3.sql.gz.partial`: 5,518,020,608 bytes. Invalid interrupted prefix, superseded by the checksum-verified full ToxCast archive.
- `.git/objects/1e/tmp_obj_7R47Y3`: 3,451,240,448 bytes. Sole Git garbage temporary object; no lock/open handle and the Git index remained readable.
- **Logical bytes removed:** 8,969,261,056 (~8.35 GiB).

These deletions are not recoverable locally. The partial could only be re-downloaded; the Git temporary object was unreferenced and requires no recovery.

### Retain

- extracted ChEMBL 37 database: active canonical query source;
- DailyMed archives: essential for PK value-level extraction;
- complete verified ToxCast archive: required for broader toxicity/ADME extraction;
- model-ready Kd/Ki/IC50 partitions: immediate training inputs;
- BindingDB archive and overlap SQLite: required for admission, deduplication, and external validation;
- primary canonical and split trees.

### Do not delete without explicitly retiring audit replay

- two failed, never-promoted canonical build payloads: about 7.1 GiB. They are excluded from every training input and can be rebuilt, but historical audit documents declare them retained as forensic evidence;
- deterministic Build-B canonical/split trees: about 1.62 GiB. Their large files are byte-identical to the primary trees, but they preserve physical A/B determinism evidence;
- ChEMBL compressed source archive: about 5.76 GB. The extracted database is active, but deleting the archive trades local provenance/recovery for space;
- manifest-bound legacy ChEMBL snapshot duplicates: about 43 MiB.

Deleting the first two groups would save substantial space but invalidate existing path-based audit/replay guarantees. It should be treated as a formal retention-policy change, not ordinary duplicate cleanup.
