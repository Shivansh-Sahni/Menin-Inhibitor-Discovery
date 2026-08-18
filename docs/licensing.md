# Licensing decision record

## Current status

No `LICENSE` file is present and no software license has been selected. This is deliberate: the project owner must choose the license after confirming institutional, sponsor, collaborator, patent, and third-party-data obligations. This documentation does not grant permission to copy, modify, distribute, or use the repository.

In many jurisdictions, code and documentation without an explicit license remain protected by default copyright rules. Public visibility of a repository is not the same as an open-source license. Obtain permission or wait for an approved license before reuse.

This is not legal advice. The rights holder should use institutional counsel or technology-transfer guidance when needed.

## Separate rights surfaces

A licensing decision must treat these separately:

| Surface | Decision needed |
| --- | --- |
| Original source code | Select a software license and confirm every contributor can grant it. |
| Documentation and figures | Decide whether the software license applies or a content license is needed. |
| ChEMBL-derived data | Follow current ChEMBL/EMBL-EBI terms and CC BY-SA obligations; preserve attribution and release metadata. |
| BindingDB-derived data | Follow row provenance; BindingDB-curated and ChEMBL-imported records can have different Creative Commons terms. |
| PubChem/depositor data | Review PubChem and original depositor/source terms and citation requirements. |
| Patent/literature content | Database access does not remove third-party patent, copyright, or other rights. |
| Trained models | Determine whether weights/artifacts are covered by the software license, a model license, source-data terms, or confidentiality restrictions. |
| Proprietary lab data and derivatives | Keep private unless the data owner explicitly approves a defined release. |

The repository's future code license cannot relicense upstream data or confidential inputs.

## Owner decisions required

Before adding a license:

- confirm the code copyright holder(s) and whether the institution claims ownership;
- review collaboration, funding, sponsor, employment, and invention-assignment terms;
- confirm contributor history and obtain any required contributor agreement;
- decide whether commercial use, patent grants, copyleft, attribution, warranty disclaimer, and redistribution conditions match project goals;
- decide whether documentation, model artifacts, generated analysis/prioritization outputs, reports, and example data need separate terms;
- inventory third-party code/assets and verify compatibility; and
- define the process for accepting future contributions.

Use a recognized standard license rather than drafting custom terms without counsel. [Choose a License](https://choosealicense.com/) can explain common options, but the final choice remains the rights holder's decision.

## Candidate categories for owner review

This table is informational, not a recommendation or license grant.

| Category | Typical property to review |
| --- | --- |
| Permissive software license (for example BSD-3-Clause or Apache-2.0) | Broad reuse; Apache-2.0 includes an express patent license, while exact notice obligations differ. |
| Reciprocal/copyleft license (for example GPLv3) | Distributed derivatives generally carry reciprocal source/license obligations. |
| Source-available/custom research terms | May restrict uses and may not meet open-source definitions; custom drafting and compatibility create legal/operational costs. |
| No public license | Preserves default restrictions but prevents normal open-source reuse and external contribution. |

Do not add a Creative Commons license to software merely because source data use Creative Commons terms; software and database/content licenses serve different purposes.

## Release action after approval

Once the owner approves a license:

1. Add the exact canonical license text as `LICENSE`.
2. Add the license identifier to `pyproject.toml` and `CITATION.cff` if accurate.
3. Add copyright/notices files if required.
4. Define whether generated models, analysis outputs, docs, and reports are included.
5. Update the README and contribution policy.
6. Review every bundled data file and artifact separately; remove anything not permitted.
7. Record the approval and effective release/tag in this file.

## Approval record

| Item | Value |
| --- | --- |
| Rights holder(s) | Pending |
| Institution/technology-transfer review | Pending |
| Selected code license | Pending |
| Documentation license | Pending |
| Model artifact license | Pending |
| Data redistribution decision | Pending by source/build |
| Approved by/date | Pending |
| First licensed release | Pending |
