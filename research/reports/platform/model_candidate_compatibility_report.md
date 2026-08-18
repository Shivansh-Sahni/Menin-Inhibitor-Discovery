# Model candidate compatibility report

Evidence checked: 2026-08-04. Status: architecture and preflight registry prepared; no large model downloaded, fine-tuned, or evaluated.

The platform is protein-agnostic and does not center Menin or any single architecture. Candidate selection must follow the intended generalization claim and measured input coverage. Conventional descriptor, fingerprint, and graph baselines precede any claim of value from a large pretrained or structure-conditioned model.

## Candidate families

The machine-readable registry includes a supervised D-MPNN, a molecular transformer, a protein language model, a molecule/protein dual encoder, a generic structure-conditioned complex model, and three named frozen external comparators. Generic candidates intentionally remain blocked until an exact checkpoint, license, training cutoff, maximum input length, and overlap policy are selected.

The named comparators were checked against their official sources:

| Candidate | Verified interface and role | Reported license | Mandatory caution |
|---|---|---|---|
| [Chai-1](https://github.com/chaidiscovery/chai-lab) | Protein/ligand and other biomolecular structure prediction; frozen pose/structure comparator | Apache-2.0 for code and model weights, per official repository | Pin package, weight revision, and hashes; audit ligand/structure overlap; it is not an endpoint-specific affinity model |
| [Boltz-2](https://github.com/jwohlwend/boltz) | Cofolding plus documented binder probability and `log10(IC50)` in micromolar outputs | MIT for code and weights, per official repository | Never reinterpret its documented affinity output as Kd or standard binding free energy; pin release and audit overlap |
| [Nesso-1](https://huggingface.co/recursionpharma/nesso) | Protein sequence plus ligand to affinity scalar and binder/non-binder score; frozen external comparator | Apache-2.0, per official model card | Pin checkpoint and hashes; verify endpoint semantics from the technical report; never relabel mixed potency/affinity supervision as endpoint-specific Kd or free energy |

Reported upstream licenses are inventory facts, not project legal approval. The exact license text and immutable checkpoint distribution used in a future run must be archived and reviewed.

## Selection order

1. Fixed dummy, linear, fingerprint, and restrained tree diagnostics on immutable splits.
2. D-MPNN or frozen molecular/protein embeddings, with ablations against the simple features.
3. Dual-encoder fusion for protein-level and double-cold claims.
4. Structure-conditioned models only after coordinate/pose validity and training-overlap audits, and only if they add value beyond simpler models.
5. Chai-1, Boltz-2, and Nesso-1 as frozen external comparators under endpoint-correct protocols; they are not substitutes for the canonical observed labels.

## Blocking preflight

Every candidate remains blocked from substantive training until the dataset, official split manifest, tokenizer/representation artifact, exact checkpoint revision/hash, software environment, license snapshot/review, training-data overlap audit, hardware inventory, and task-specific loss/model-selection contract are frozen. These blockers are intentional.

## Materialized artifacts

- `research/models/platform/model_candidate_registry.json`
- `research/models/platform/model_candidate_registry.csv`
- `research/models/platform/pretraining_static_manifest.json`
