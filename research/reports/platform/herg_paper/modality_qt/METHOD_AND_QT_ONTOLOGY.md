# Wild-Type hERG Measurement Modality and QT/QTc Axis

## Delivered result

The additive index covers **407,698 wild-type-scope hERG observations** while
physically excluding all **258 explicitly mutant/variant records**. It does not
modify the upstream hERG hierarchy. Target evidence is kept honest:

- **343,909** observations are confirmed wild type (PubChem AID 720551).
- **63,789** are wild type or target variant unspecified.
- Unspecified records remain usable, but are never renamed confirmed wild type.

The method layer makes technology, automation, and dose design separate model
covariates. Missing method metadata is represented as `unresolved`; no generic
assay is silently labeled manual or automated.

## Measurement technology inventory

| Measurement modality | Observations | Unique structures | Intended use |
|---|---:|---:|---|
| High-throughput thallium flux | 344,029 | 340,212 | Large weak-label pretraining and flux-specific model |
| Patch-clamp electrophysiology | 16,200 | 8,943 | Higher-value electrophysiology head, fine-tuning, and evaluation |
| Binding, method unspecified | 10,094 | 7,981 | Binding-specific auxiliary head |
| Radioligand binding | 9,787 | 7,017 | Mechanistic binding head; do not equate with current block |
| Functional electrophysiology | 1,667 | 1,216 | Functional auxiliary stratum when patch clamp is not explicit |
| Functional method unspecified | 816 | 715 | Lower-confidence auxiliary stratum |
| Functional ion flux | 444 | 397 | Separate surrogate-flux stratum |
| Curated clinical QT/QTc phenotype assay | 7 | 5 | Clinical phenotype context, not hERG potency |
| Unresolved technology | 24,654 | 23,347 | Usable with a missing-method indicator; sensitivity analysis required |

Automation evidence is available at scale: **354,057 automated**, **579
explicitly manual/conventional**, **53,055 unresolved**, and **7 not
applicable** observations. The automated total is dominated by the 343,909-row
qHTS backbone, so an unadjusted automated-versus-manual comparison would mostly
measure source and assay differences. Within-source or matched-structure
comparisons should be the primary analysis.

## Classification rules

1. PubChem AID 720551 is assigned high-throughput FluxOR thallium-flux qHTS
   from its frozen source contract and fixed two-concentration categorical
   design.
2. IonWorks, QPatch/Q-Patch, Patchliner, SyncroPatch, explicit automated patch,
   plate-based electrophysiology, and planar patch phrases support automated
   patch-clamp classification.
3. Only explicit `manual` or `conventional patch` language supports manual
   classification. Generic `patch clamp` remains automation-unresolved.
4. Radioligand/tracer/displacement/scintillation phrases support radioligand
   binding. Generic ChEMBL binding assays remain binding-method-unspecified.
5. Thallium/FluxOR, ion-efflux, and voltage/current phrases are classified on
   explicit text only. Clinical QT/QTc classification requires membership in
   the curated phenotype-assay registry or an explicit native QT/QTc-interval
   endpoint; generic QT background language in an in-vitro hERG assay is not
   enough. Unreported details remain unresolved.
6. IC50/EC50/AC50/Ki/Kd and their log forms are indexed as
   concentration-response summaries; explicit test concentrations become
   fixed-dose observations; kinetic endpoints remain separate.

Dose-design counts are **344,147 fixed-dose categorical**, **6,725 fixed-dose
quantitative**, **48,946 concentration-response summaries**, **5,943 kinetic
measurements**, and **1,937 unresolved**. These are study-design categories,
not interchangeable endpoint units.

## QT/QTc interpretation

QT prolongation is incorporated as a separate clinical phenotype axis because
hERG/IKr inhibition is one important mechanism of delayed repolarization, but a
QT result is not an in-vitro hERG potency value. The exact structure-linked
clinical inventory contains:

- **221** posted QT/QTc endpoints;
- **95** linked molecular structures;
- **143** clinical trials;
- **3,828** reported result records, including **3,819 numeric values**; and
- **1,777** denominator records.

Phenotype classes comprise **148 event/threshold endpoints** and **73 interval
measurements**. The latter include 52 change-from-baseline and 21 interval-value
endpoints. Correction-method evidence includes 92 QTcF-only, 10 QTcB-only, 44
QTcF+QTcB, one QTcI, and 74 unresolved endpoints.

Every row retains its native result and denominator JSON, NCT identifier,
endpoint identifier, unit, time frame, source page, and JSON pointer. No QT
value is converted to IC50/pIC50 and no QT row is admitted as a hERG training
label. QT can instead support a secondary clinical concordance analysis:

1. train hERG models only on assay evidence;
2. generate blinded hERG predictions for the 95 trial-linked structures;
3. compare predictions with interval-change and threshold/event strata;
4. stratify by QT correction method and trial context; and
5. describe discordance as multi-factor cardiac biology, exposure, population,
   or measurement differences—not automatically as model error.

## Paper analysis enabled now

- Train a high-coverage thallium-flux model, a patch-clamp model, and
  assay-aware multitask model with a shared molecular encoder.
- Compare identical scaffold splits and report performance separately by
  technology, automation evidence, and dose design.
- Use patch-clamp observations as the highest-value model comparison stratum,
  while retaining the much larger flux corpus for representation learning.
- Add method and automation indicators to expose whether performance gains are
  data-source shortcuts or transferable chemistry.
- Run ablations: molecule-only; molecule plus modality; source-balanced;
  modality-specific heads; and quality-level curriculum/fine-tuning.
- Use the 24,654 unresolved-method rows rather than discarding them, but report
  a sensitivity analysis with and without them.

## Limitations to state explicitly

- Text mining recovers what the source reports; it cannot reconstruct missing
  voltage protocol, temperature, incubation, replicate, or automation details.
- The 579 manual records are an explicit-evidence lower bound, not the total
  number of manual patch-clamp experiments.
- ChEMBL assay-type metadata and descriptions can disagree; the index favors
  explicit method phrases and preserves the original assay description.
- Only seven source rows satisfy the curated clinical-QT rule. Earlier
  keyword-only screening overcalled 37 rows because 30 ordinary in-vitro hERG
  assays merely mentioned QT prolongation in their background text; those 30
  are now restored to their actual assay modalities.
- Method, source, endpoint, chemical space, and publication year are confounded.
  Claims about method impact require matched or adjusted analyses.
- Seventy-four trial endpoints do not name a QT correction method, and a small
  number have unusual reported units. Keep these records for breadth, with the
  supplied semantic class as a review/sensitivity covariate.

## Reproduction

```bash
env PYTHONPATH=pipeline/src .venv/bin/python -m menin_discovery.platform_herg_modality_qt \
  --hierarchy-root research/data/platform/processed/herg_hierarchy/v1 \
  --clinical-links-root research/data/platform/processed/herg_hierarchy/v1_clinical_links \
  --output-root research/data/platform/processed/herg_hierarchy/v1_2_modality_qt

env PYTHONPATH=pipeline/src .venv/bin/python -m menin_discovery.platform_herg_modality_qt \
  --output-root research/data/platform/processed/herg_hierarchy/v1_2_modality_qt \
  --validate-only
```

The manifest binds every upstream input and generated artifact by SHA-256,
Arrow schema, byte size, and row count. The builder is deterministic and
fail-closed on mutant admission, unknown ontology values, weakened QT/hERG
separation, count drift, schema drift, or file tampering.
