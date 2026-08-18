# PK-DB public candidate audit

## Outcome

**Fail closed: 0 candidate observations, 0 canonical observations, and 0 training labels.**

The official statistics endpoint reported **138,411 outputs**, while the anonymous
outputs probe returned **0** records. The official OpenAPI document also declares
API-wide authentication, and the retained public study probe demonstrates that `access=public`
can coexist with a closed licence. Therefore no numeric PK result, compound link, or model label
was admitted.

The public count surfaces are also not interchangeable: statistics reported **819**
studies and **2392** interventions, versus **803**
and **4762** from the corresponding anonymous list queries. This may reflect
different definitions or access filters and must be resolved before any coverage claim.

## What was retained

- Exact official statistics, OpenAPI, anonymous output probe, and ten PK ontology nodes.
- HTTP status/date/version/body-size/SHA-256 receipts and a complete artifact manifest.
- Privacy-minimized study/intervention probes; publication text and numeric dose values were discarded.
- Separate semantics for AUC-end, AUC-infinity, clearance, half-life, distribution volume,
  steady-state volume, Cmax, Tmax, bioavailability, and fraction unbound.

## Gate for future use

Obtain documented reproducible output access, complete a record-level/terms licence review, and
then measure study/species/route/matrix/dose/time/unit context completeness. Unknown fields must
not be interpreted as absent, and endpoint families must not be pooled without an explicit model.
