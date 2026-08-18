# Structure-linked adjacent training surfaces — 2026-08-09

PRISM is now indexed as a structure-linked drug-by-cell response surface with 8,372,603 eligible finite log-fold-change targets. Reported component sets remain explicit disconnected structures rather than being silently reduced to one component. Cell contexts that failed or lack STR metadata are retained but excluded from eligible counts.

LINCS is now indexed at the compound-instance grain with 976,325 structure-resolved compound profiles and 954,845,850 metadata-derived, addressable landmark-gene positions. These positions are not claimed as scanned finite values. The compressed GCTX matrices remain source-bound rather than being wastefully expanded to long format; training requires a GCTX-capable loader or deterministic staging.

Exact structures, Bemis–Murcko scaffolds, and normalized Broad compound IDs are joined into cross-modality connected leakage groups before deterministic train, validation, and test assignment. PRISM viability and LINCS expression remain separate context-dependent objectives. No molecular features, model fitting, HPC work, figure, or presentation table was generated.
