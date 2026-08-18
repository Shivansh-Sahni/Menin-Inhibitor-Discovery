from __future__ import annotations

import collections
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/Users/shivanshsahni/Downloads/Personal/Wang/Menin")
PARTS = [Path(f"/private/tmp/dm_pk_part{n}") for n in range(1, 7)]
OUT = ROOT / "research/data/platform/raw/external_public/pk_expansion/avicenna/dailymed_pk_candidate_evidence"
DAILYMED = ROOT / "research/data/platform/raw/external_public/dailymed_spl_v2_human_rx"
DAILYMED_MANIFEST = DAILYMED / "dailymed_spl_v2_human_rx_manifest.json"
FDA_MANIFEST = ROOT / "research/data/platform/raw/external_public/drugs_at_fda_bulk/drugs_at_fda_bulk_manifest.json"
SCHEMA = "pk-expansion-dailymed-candidates/1.0"
GENERATED = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def json_lines(path: Path):
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            try:
                yield line_number, json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def doc_key(row):
    return (row.get("archive"), row.get("outer_member"), row.get("inner_xml_member"))


def version_order(row):
    effective = row.get("effective_time") or ""
    version = row.get("version_number") or ""
    try:
        version_n = int(version)
    except (TypeError, ValueError):
        version_n = -1
    return (effective, version_n, row.get("outer_member") or "", row.get("document_id") or "")


OUT.mkdir(parents=True, exist_ok=True)
for name in ["document_inventory.jsonl", "section_candidates.jsonl", "table_candidates.jsonl"]:
    with (OUT / name).open("wb") as dst:
        for part in PARTS:
            with (part / name).open("rb") as src:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)

part_summaries = [json.loads((part / "scan_summary.json").read_text()) for part in PARTS]

doc_keys = set()
candidate_doc_keys = set()
candidate_docs = []
document_ids = set()
set_versions = collections.defaultdict(set)
candidate_set_versions = collections.defaultdict(set)
latest_candidate_by_set = {}
doc_counts = collections.Counter()
candidate_provenance = collections.Counter()
parse_error_rows = []
for _, row in json_lines(OUT / "document_inventory.jsonl"):
    doc_counts["rows"] += 1
    if row.get("parse_status") == "error":
        parse_error_rows.append(row)
        doc_counts["errors"] += 1
        continue
    key = doc_key(row)
    if key in doc_keys:
        doc_counts["duplicate_locator_rows"] += 1
    doc_keys.add(key)
    doc_counts["parsed"] += 1
    if row.get("has_clinical_pharmacology_section"):
        doc_counts["clinical_pharmacology"] += 1
    if row.get("has_explicit_pharmacokinetics_section"):
        doc_counts["explicit_pk"] += 1
    if row.get("document_id"):
        document_ids.add(row["document_id"])
    if row.get("set_id"):
        set_versions[row["set_id"]].add(row.get("version_number") or "<NULL>")
    if row.get("drugsfda_matches"):
        doc_counts["fda_match"] += 1
    if row.get("candidate_section_count", 0) > 0:
        doc_counts["candidate_docs"] += 1
        candidate_doc_keys.add(key)
        candidate_docs.append(row)
        if row.get("set_id"):
            candidate_set_versions[row["set_id"]].add(row.get("version_number") or "<NULL>")
            current = latest_candidate_by_set.get(row["set_id"])
            if current is None or version_order(row) > version_order(current):
                latest_candidate_by_set[row["set_id"]] = row
        if row.get("drugsfda_matches"):
            doc_counts["candidate_fda_match"] += 1
        for field in ["document_id", "set_id", "version_number", "effective_time", "document_title"]:
            if row.get(field):
                candidate_provenance[field + "_present"] += 1
        for field in ["author_organizations", "product_names", "product_codes", "active_ingredients", "approval_ids", "drugsfda_matches"]:
            if row.get(field):
                candidate_provenance[field + "_present"] += 1

latest_keys = {doc_key(row) for row in latest_candidate_by_set.values()}
with (OUT / "latest_available_candidate_documents.jsonl").open("w") as f:
    for set_id, row in sorted(latest_candidate_by_set.items()):
        compact = {
            "schema_version": SCHEMA,
            "view_status": "latest_available_version_in_frozen_bulk_candidate_only",
            "set_id": set_id,
            "version_number": row.get("version_number"),
            "effective_time": row.get("effective_time"),
            "document_id": row.get("document_id"),
            "archive": row.get("archive"),
            "outer_member": row.get("outer_member"),
            "inner_xml_member": row.get("inner_xml_member"),
            "inner_xml_sha256": row.get("inner_xml_sha256"),
            "document_title": row.get("document_title"),
            "author_organizations": row.get("author_organizations"),
            "product_names": row.get("product_names"),
            "product_codes": row.get("product_codes"),
            "active_ingredients": row.get("active_ingredients"),
            "approval_ids": row.get("approval_ids"),
            "drugsfda_matches": row.get("drugsfda_matches"),
            "candidate_section_count": row.get("candidate_section_count"),
            "admission_status": "candidate_inventory_only_not_a_label",
        }
        f.write(json.dumps(compact, sort_keys=True, ensure_ascii=False) + "\n")

candidate_ids = set()
section_doc_keys = set()
selection = collections.Counter()
endpoints = collections.Counter()
contexts = collections.Counter()
context_tiers = collections.Counter()
section_code_titles = collections.Counter()
section_provenance = collections.Counter()
effective_years = collections.Counter()
endpoint_cooccurrence = collections.Counter()
section_counts = collections.Counter()
latest_section_counts = collections.Counter()
candidate_docs_with_tier_a = set()
evidence_span_count = 0
for _, row in json_lines(OUT / "section_candidates.jsonl"):
    section_counts["rows"] += 1
    cid = row["candidate_id"]
    if cid in candidate_ids:
        section_counts["duplicate_candidate_ids"] += 1
    candidate_ids.add(cid)
    key = doc_key(row)
    section_doc_keys.add(key)
    if key not in doc_keys:
        section_counts["orphan_document_locator"] += 1
    if key not in candidate_doc_keys:
        section_counts["noncandidate_document_locator"] += 1
    if key in latest_keys:
        latest_section_counts["rows"] += 1
    selection[row["selection_reason"]] += 1
    hits = tuple(sorted(row.get("endpoint_hits") or []))
    for hit in hits:
        endpoints[hit] += 1
    endpoint_cooccurrence["+".join(hits)] += 1
    evidence_span_count += len(row.get("evidence_spans") or [])
    flags = row["context_completeness_flags"]
    for name, value in flags.items():
        if value:
            contexts[name] += 1
    if flags.get("all_core_context_flags_in_same_section"):
        tier = "A_all_core_machine_detected_unverified"
        candidate_docs_with_tier_a.add(key)
    elif flags.get("unit_like_mentioned") and flags.get("matrix_mentioned") and flags.get("human_population_mentioned") and (flags.get("route_mentioned") or flags.get("dose_mentioned")):
        tier = "B_endpoint_unit_matrix_human_and_partial_administration_context"
    else:
        tier = "C_quantitative_endpoint_candidate_with_major_context_gap"
    context_tiers[tier] += 1
    if key in latest_keys:
        latest_section_counts[tier] += 1
    section_code_titles[(row.get("section_code") or "<NULL>", row.get("section_title") or "")] += 1
    if row.get("effective_time") and len(row["effective_time"]) >= 4:
        effective_years[row["effective_time"][:4]] += 1
    for field in ["document_id", "set_id", "version_number", "effective_time", "inner_xml_sha256", "section_id", "normalized_section_text_sha256"]:
        if row.get(field):
            section_provenance[field + "_present"] += 1
    section_counts["declared_tables"] += row.get("table_count", 0)

table_counts = collections.Counter()
table_keys = set()
for _, row in json_lines(OUT / "table_candidates.jsonl"):
    table_counts["rows"] += 1
    cid = row.get("candidate_id")
    if cid not in candidate_ids:
        table_counts["orphan_candidate_id"] += 1
    key = (cid, row.get("table_index_in_section"))
    if key in table_keys:
        table_counts["duplicate_table_key"] += 1
    table_keys.add(key)
    if row.get("table_xml_sha256"):
        table_counts["xml_hash_present"] += 1
    if row.get("normalized_text_sha256"):
        table_counts["text_hash_present"] += 1
    if row.get("numeric_token_count", 0) > 0:
        table_counts["numeric_token_present"] += 1
    if row.get("endpoint_hits"):
        table_counts["pk_endpoint_hit_present"] += 1

dailymed_manifest = json.loads(DAILYMED_MANIFEST.read_text())
archive_entries = {x.get("local_path"): x for x in dailymed_manifest["files"] if x.get("artifact_role") == "human_prescription_release_part"}
archive_validation = []
for part_summary in part_summaries:
    archive_name = part_summary["by_archive"][0]["archive"]
    path = DAILYMED / archive_name
    entry = archive_entries[archive_name]
    with zipfile.ZipFile(path) as z:
        actual_members = len(z.infolist())
    actual_sha = file_sha(path)
    archive_validation.append({
        "archive": archive_name,
        "expected_bytes": entry["acquired_bytes"],
        "actual_bytes": path.stat().st_size,
        "expected_sha256": entry["acquired_sha256"],
        "actual_sha256": actual_sha,
        "expected_outer_members": entry["expected_file_member_count"],
        "actual_outer_members": actual_members,
        "scanned_outer_members": part_summary["counts"]["outer_members_scanned"],
        "parsed_inner_xml_members": part_summary["counts"]["inner_xml_members_parsed"],
        "parse_errors": part_summary["counts"]["parse_errors"],
        "outer_member_crc_policy": "Every outer member was fully read by Python zipfile; CRC mismatch would have raised and been counted as a scan error.",
        "inner_xml_crc_policy": "Every selected inner XML member was fully read by Python zipfile; CRC mismatch would have raised and been counted as a scan error.",
        "passed": path.stat().st_size == entry["acquired_bytes"] and actual_sha == entry["acquired_sha256"] and actual_members == entry["expected_file_member_count"] == part_summary["counts"]["outer_members_scanned"] == part_summary["counts"]["inner_xml_members_parsed"] and part_summary["counts"]["parse_errors"] == 0,
    })

counts = {
    "frozen_outer_archives": len(archive_validation),
    "outer_members_scanned": doc_counts["rows"],
    "inner_xml_members_parsed": doc_counts["parsed"],
    "parse_errors": doc_counts["errors"],
    "documents_with_clinical_pharmacology_section": doc_counts["clinical_pharmacology"],
    "documents_with_explicit_pharmacokinetics_section": doc_counts["explicit_pk"],
    "candidate_document_versions": doc_counts["candidate_docs"],
    "candidate_sections": section_counts["rows"],
    "candidate_tables": table_counts["rows"],
    "unique_document_ids": len(document_ids),
    "unique_set_ids_all_documents": len(set_versions),
    "unique_set_ids_candidate_documents": len(candidate_set_versions),
    "set_ids_with_multiple_versions_all_documents": sum(len(x) > 1 for x in set_versions.values()),
    "set_ids_with_multiple_candidate_versions": sum(len(x) > 1 for x in candidate_set_versions.values()),
    "latest_available_candidate_documents": len(latest_candidate_by_set),
    "latest_available_candidate_sections": latest_section_counts["rows"],
    "documents_with_exact_drugsfda_application_match": doc_counts["fda_match"],
    "candidate_documents_with_exact_drugsfda_application_match": doc_counts["candidate_fda_match"],
    "candidate_documents_with_at_least_one_tier_A_section": len(candidate_docs_with_tier_a),
    "bounded_evidence_spans": evidence_span_count,
    "canonical_rows": 0,
    "training_labels": 0,
}

summary = {
    "schema_version": SCHEMA,
    "generated_at_utc": GENERATED,
    "scope": "Full six-archive DailyMed human-prescription SPL candidate evidence scan with exact Drugs@FDA application-number joins; no measurement extraction.",
    "candidate_definition": part_summaries[0]["candidate_definition"],
    "counts": counts,
    "context_completeness_tiers": dict(context_tiers),
    "latest_available_context_completeness_tiers": {k: latest_section_counts[k] for k in context_tiers},
    "context_tier_definitions": {
        "A_all_core_machine_detected_unverified": "Human/population, route, dose, matrix, and unit-like terms all detected in the same candidate section. This is not human validation.",
        "B_endpoint_unit_matrix_human_and_partial_administration_context": "Endpoint, units, matrix, and human/population detected, with route or dose but not both.",
        "C_quantitative_endpoint_candidate_with_major_context_gap": "Quantitative endpoint candidate that fails the stricter A/B context rules.",
    },
    "candidate_sections_by_selection_reason": dict(selection),
    "candidate_sections_by_endpoint_hit_nonexclusive": dict(endpoints),
    "candidate_sections_by_endpoint_combination": dict(endpoint_cooccurrence.most_common()),
    "candidate_sections_by_context_flag_nonexclusive": dict(contexts),
    "candidate_sections_by_effective_year": dict(sorted(effective_years.items())),
    "candidate_document_provenance_presence": dict(candidate_provenance),
    "candidate_section_provenance_presence": dict(section_provenance),
    "table_inventory": dict(table_counts),
    "top_section_code_title_pairs": [
        {"section_code": key[0], "section_title": key[1], "candidate_sections": count}
        for key, count in section_code_titles.most_common(100)
    ],
    "by_archive": [x["by_archive"][0] for x in part_summaries],
    "source_bindings": {
        "dailymed_manifest": {"path": str(DAILYMED_MANIFEST.relative_to(ROOT)), "bytes": DAILYMED_MANIFEST.stat().st_size, "sha256": file_sha(DAILYMED_MANIFEST)},
        "drugsfda_manifest": {"path": str(FDA_MANIFEST.relative_to(ROOT)), "bytes": FDA_MANIFEST.stat().st_size, "sha256": file_sha(FDA_MANIFEST)},
        "archives": archive_validation,
    },
    "admission_boundary": "Candidate sections, evidence spans, table hashes, and application joins are review aids only. No numeric value was normalized into CL/t1/2/AUC/Cmax/Tmax/Vd/F; canonical rows and training labels remain zero.",
    "important_limitations": [
        "Regex selection can miss values expressed without recognized endpoint terms and can retain unrelated numbers inside a selected section.",
        "Context flags report term presence, not that every context term applies to every quantitative value.",
        "Latest-available view is based only on the frozen bulk corpus and is not asserted to be the currently marketed or regulator-preferred label.",
        "Drugs@FDA links require an exact numeric approval ID in SPL; absence of a match is not evidence of no approval.",
        "Multiple label versions and multiple products may share ingredients; no molecular deduplication or structure mapping was attempted.",
    ],
}
write_json(OUT / "scan_summary.json", summary)

validation_checks = {
    "all_six_archives_pass_size_sha_member_parse_validation": all(x["passed"] for x in archive_validation),
    "outer_member_count_matches_frozen_manifest": doc_counts["rows"] == dailymed_manifest["expected_and_verified_file_member_count"] == 54672,
    "all_outer_members_have_parsed_inner_xml": doc_counts["rows"] == doc_counts["parsed"],
    "parse_errors_zero": doc_counts["errors"] == 0 and not parse_error_rows,
    "document_locators_unique": doc_counts["duplicate_locator_rows"] == 0 and len(doc_keys) == doc_counts["rows"],
    "candidate_ids_unique": section_counts["duplicate_candidate_ids"] == 0 and len(candidate_ids) == section_counts["rows"],
    "section_document_links_complete": section_counts["orphan_document_locator"] == 0 and section_counts["noncandidate_document_locator"] == 0,
    "table_keys_unique": table_counts["duplicate_table_key"] == 0 and len(table_keys) == table_counts["rows"],
    "table_section_links_complete": table_counts["orphan_candidate_id"] == 0,
    "declared_table_count_matches_table_rows": section_counts["declared_tables"] == table_counts["rows"],
    "all_table_hashes_present": table_counts["xml_hash_present"] == table_counts["text_hash_present"] == table_counts["rows"],
    "canonical_rows_zero": counts["canonical_rows"] == 0,
    "training_labels_zero": counts["training_labels"] == 0,
}
validation = {
    "schema_version": SCHEMA,
    "generated_at_utc": GENERATED,
    "checks": validation_checks,
    "all_passed": all(validation_checks.values()),
    "archive_validation": archive_validation,
    "internal_counters": {"documents": dict(doc_counts), "sections": dict(section_counts), "tables": dict(table_counts)},
}
write_json(OUT / "validation.json", validation)

artifacts = []
for path in sorted(OUT.iterdir()):
    if not path.is_file() or path.name == "manifest.json":
        continue
    artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)})
manifest = {
    "schema_version": SCHEMA,
    "generated_at_utc": GENERATED,
    "artifact_count": len(artifacts),
    "total_bytes": sum(x["bytes"] for x in artifacts),
    "artifacts": artifacts,
    "manifest_excluded_as_self_referential": "manifest.json",
    "canonical_rows": 0,
    "training_labels": 0,
}
write_json(OUT / "manifest.json", manifest)
print(json.dumps({"counts": counts, "context_tiers": dict(context_tiers), "validation_all_passed": validation["all_passed"], "artifact_count": len(artifacts), "total_bytes": manifest["total_bytes"]}, indent=2))
