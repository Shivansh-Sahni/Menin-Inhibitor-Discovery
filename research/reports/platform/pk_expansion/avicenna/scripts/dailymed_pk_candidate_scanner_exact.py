from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/Users/shivanshsahni/Downloads/Personal/Wang/Menin")
DAILYMED = PROJECT / "research/data/platform/raw/external_public/dailymed_spl_v2_human_rx"
FDA = PROJECT / "research/data/platform/raw/external_public/drugs_at_fda_bulk"
DEFAULT_OUT = PROJECT / "research/data/platform/raw/external_public/pk_expansion/avicenna/dailymed_pk_candidate_evidence"
NS = "{urn:hl7-org:v3}"
SCHEMA = "pk-expansion-dailymed-candidates/1.0"

ENDPOINTS = {
    "auc": re.compile(r"\bAUC(?:\s|\b|[_0-9\-–∞])|area\s+under\s+(?:the\s+)?(?:plasma\s+|serum\s+|blood\s+)?(?:concentration[- ]time\s+)?curve", re.I),
    "cmax": re.compile(r"\bC\s*max\b|maximum\s+(?:plasma\s+|serum\s+|blood\s+)?concentration|peak\s+(?:plasma\s+|serum\s+|blood\s+)?concentration", re.I),
    "tmax": re.compile(r"\bT\s*max\b|time\s+to\s+(?:reach\s+)?(?:maximum|peak)\s+(?:plasma\s+|serum\s+|blood\s+)?concentration", re.I),
    "half_life": re.compile(r"half[- ]?li(?:fe|ves)|\bt\s*[½1]\s*[/⁄]\s*2\b|terminal\s+elimination\s+half", re.I),
    "clearance": re.compile(r"\bclearance\b|\bCL(?:/F)?\b", re.I),
    "volume_distribution": re.compile(r"volume\s+of\s+distribution|\bVd(?:ss)?\b|\bVss\b", re.I),
    "bioavailability": re.compile(r"\bbioavailability\b|absolute\s+bioavailability|relative\s+bioavailability", re.I),
}
PK_TITLE = re.compile(r"pharmaco\s*kinetic", re.I)
CP_TITLE = re.compile(r"clinical\s+pharmacology", re.I)
NUMBER = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z])")
UNIT = re.compile(r"(?:ng|pg|µg|μg|mcg|mg|g|nmol|µmol|μmol|mmol|nM|µM|μM|mM|mL|L)\s*(?:/|per\s+)?\s*(?:mL|L|kg|g|h|hr|hour|min|day)?", re.I)
DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ng|µg|μg|mcg|mg|g)(?:\s*/\s*kg)?\b", re.I)
ROUTE = re.compile(r"\b(?:oral(?:ly)?|intravenous(?:ly)?|IV|subcutaneous(?:ly)?|intramuscular(?:ly)?|inhal(?:ed|ation)|intranasal(?:ly)?|topical(?:ly)?|transdermal(?:ly)?|rectal(?:ly)?|sublingual(?:ly)?)\b", re.I)
MATRIX = re.compile(r"\b(?:plasma|serum|whole\s+blood|blood|urine|cerebrospinal\s+fluid|CSF|saliva|feces|faeces)\b", re.I)
HUMAN = re.compile(r"\b(?:human|patient|subject|participant|volunteer|adult|pediatric|paediatric|child|children|infant|neonate|elderly|geriatric|men|women)\b", re.I)
ANIMAL = re.compile(r"\b(?:rat|rats|mouse|mice|dog|dogs|monkey|monkeys|rabbit|rabbits|animal|animals)\b", re.I)
POPULATION = re.compile(r"\b(?:healthy\s+(?:subjects?|volunteers?)|patients?\s+with\s+[^.;:]{1,100}|renal(?:ly)?\s+impair(?:ed|ment)|hepatic(?:ally)?\s+impair(?:ed|ment)|pediatric|paediatric|geriatric|elderly)\b", re.I)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def norm_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def unique(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fda_lookup():
    archive = FDA / "datdaf20260804.zip"
    apps = {}
    products = collections.defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        with z.open("Applications.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"), delimiter="\t")
            for row in reader:
                apps[row["ApplNo"].strip()] = {k: (v or "").strip() for k, v in row.items()}
        with z.open("Products.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"), delimiter="\t")
            for row in reader:
                products[row["ApplNo"].strip()].append({k: (v or "").strip() for k, v in row.items()})
    return apps, products, {
        "path": str(archive.relative_to(PROJECT)),
        "bytes": archive.stat().st_size,
        "sha256": file_sha(archive),
        "release_id": "drugs_at_fda_2026_08_04",
    }


def document_metadata(root: ET.Element):
    def attrs(path, attr):
        return unique(e.get(attr) for e in root.findall(path))

    def texts(path):
        return unique(norm_text(e) for e in root.findall(path))

    ingredients = []
    for ingredient in root.findall(".//" + NS + "ingredient"):
        if ingredient.get("classCode") != "ACTIM":
            continue
        names = texts_from(ingredient, ".//" + NS + "ingredientSubstance/" + NS + "name")
        ingredients.extend(names)
    approvals = []
    for e in root.findall(".//" + NS + "approval/" + NS + "id"):
        ext = e.get("extension")
        if ext:
            approvals.append(ext)
    return {
        "document_id": first(attrs("./" + NS + "id", "root")),
        "set_id": first(attrs("./" + NS + "setId", "root")),
        "version_number": first(attrs("./" + NS + "versionNumber", "value")),
        "effective_time": first(attrs("./" + NS + "effectiveTime", "value")),
        "document_title": first(texts("./" + NS + "title")),
        "author_organizations": texts(".//" + NS + "author//" + NS + "representedOrganization/" + NS + "name"),
        "product_names": texts(".//" + NS + "manufacturedProduct/" + NS + "name"),
        "product_codes": unique(
            (e.get("code"), e.get("codeSystem"))
            for e in root.findall(".//" + NS + "manufacturedProduct/" + NS + "code")
        ),
        "active_ingredients": unique(ingredients),
        "approval_ids": unique(approvals),
    }


def texts_from(element: ET.Element, path: str):
    return unique(norm_text(e) for e in element.findall(path))


def first(values):
    return values[0] if values else None


def endpoint_hits(text: str):
    return [name for name, pattern in ENDPOINTS.items() if pattern.search(text)]


def evidence_spans(text: str, hits: list[str]):
    spans = []
    for endpoint in hits:
        seen = set()
        for match in list(ENDPOINTS[endpoint].finditer(text))[:2]:
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 220)
            excerpt = text[start:end]
            key = sha(excerpt.encode("utf-8"))
            if key not in seen:
                spans.append({"endpoint": endpoint, "start": start, "end": end, "text": excerpt, "sha256": key})
                seen.add(key)
    return spans


def table_record(table: ET.Element, index: int):
    text = norm_text(table)
    xml = ET.tostring(table, encoding="utf-8")
    rows = [e for e in table.iter() if e.tag.split("}")[-1].lower() == "tr"]
    cells = [e for e in table.iter() if e.tag.split("}")[-1].lower() in {"td", "th"}]
    return {
        "table_index_in_section": index,
        "table_xml_sha256": sha(xml),
        "normalized_text_sha256": sha(text.encode("utf-8")),
        "normalized_text_chars": len(text),
        "row_count": len(rows),
        "cell_count": len(cells),
        "endpoint_hits": endpoint_hits(text),
        "numeric_token_count": len(NUMBER.findall(text)),
    }


def section_features(section: ET.Element):
    title = norm_text(section.find(NS + "title"))
    text = norm_text(section)
    code_e = section.find(NS + "code")
    section_id = section.find(NS + "id")
    tables = [e for e in section.iter() if e.tag.split("}")[-1].lower() == "table"]
    hits = endpoint_hits(text)
    numeric = len(NUMBER.findall(text))
    explicit = bool(PK_TITLE.search(title)) or (code_e is not None and code_e.get("code") == "43682-4")
    clinical = bool(CP_TITLE.search(title)) or (code_e is not None and code_e.get("code") == "34090-1")
    quantitative = bool(hits) and (numeric >= 2 or bool(tables))
    return {
        "element": section,
        "section_id": section_id.get("root") if section_id is not None else None,
        "code": code_e.get("code") if code_e is not None else None,
        "code_display_name": code_e.get("displayName") if code_e is not None else None,
        "title": title,
        "text": text,
        "text_chars": len(text),
        "text_sha256": sha(text.encode("utf-8")),
        "hits": hits,
        "numeric": numeric,
        "tables": tables,
        "explicit": explicit,
        "clinical": clinical,
        "quantitative": quantitative,
    }


def selected_sections(root: ET.Element):
    sections = root.findall(".//" + NS + "section")
    parent = {child: p for p in root.iter() for child in p}
    features = [section_features(s) for s in sections]
    by_element = {f["element"]: f for f in features}

    explicit_q = {f["element"] for f in features if f["explicit"] and f["quantitative"]}
    selected = []
    for f in features:
        if f["element"] not in explicit_q:
            continue
        ancestor = parent.get(f["element"])
        has_explicit_ancestor = False
        while ancestor is not None:
            if ancestor in explicit_q:
                has_explicit_ancestor = True
                break
            ancestor = parent.get(ancestor)
        if not has_explicit_ancestor:
            selected.append((f, "explicit_pharmacokinetics_section"))

    selected_elements = {f["element"] for f, _ in selected}
    for f in features:
        if not (f["clinical"] and f["quantitative"]):
            continue
        descendants = set(f["element"].iter())
        if descendants & selected_elements:
            continue
        ancestor = parent.get(f["element"])
        has_clinical_q_ancestor = False
        while ancestor is not None:
            af = by_element.get(ancestor)
            if af and af["clinical"] and af["quantitative"]:
                has_clinical_q_ancestor = True
                break
            ancestor = parent.get(ancestor)
        if not has_clinical_q_ancestor:
            selected.append((f, "clinical_pharmacology_quantitative_fallback"))
    return selected, features


def application_matches(meta, apps, products):
    out = []
    seen = set()
    for approval in meta["approval_ids"]:
        match = re.search(r"(?:NDA|ANDA|BLA)?\s*0*(\d{4,6})\b", approval, re.I)
        if not match:
            continue
        applno = str(int(match.group(1)))
        if applno in seen:
            continue
        seen.add(applno)
        if applno in apps:
            out.append({
                "approval_id": approval,
                "appl_no": applno,
                "application": apps[applno],
                "product_row_count": len(products[applno]),
                "drug_names": unique(p.get("DrugName") for p in products[applno]),
                "active_ingredients": unique(p.get("ActiveIngredient") for p in products[applno]),
                "forms": unique(p.get("Form") for p in products[applno]),
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--archive")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    generated = now()
    apps, products, fda_binding = fda_lookup()

    archive_paths = ([DAILYMED / args.archive] if args.archive else sorted(DAILYMED.glob("dm_spl_release_human_rx_part*.zip")))
    archive_summary = []
    endpoints = collections.Counter()
    selection_reasons = collections.Counter()
    effective_years = collections.Counter()
    context_flags = collections.Counter()
    section_codes = collections.Counter()
    set_versions = collections.defaultdict(set)
    set_candidate_versions = collections.defaultdict(set)
    errors = []
    doc_count = xml_count = candidate_docs = candidate_sections = candidate_tables = 0
    explicit_pk_docs = clinical_pharm_docs = 0
    fda_matched_docs = fda_matched_candidate_docs = 0
    t0 = time.time()

    doc_path = out / "document_inventory.jsonl"
    sec_path = out / "section_candidates.jsonl"
    table_path = out / "table_candidates.jsonl"
    with doc_path.open("w") as doc_f, sec_path.open("w") as sec_f, table_path.open("w") as table_f:
        stop = False
        for archive in archive_paths:
            part = collections.Counter()
            part["outer_members_declared"] = 0
            with zipfile.ZipFile(archive) as outer:
                part["outer_members_declared"] = len(outer.infolist())
                for oi in outer.infolist():
                    if args.limit is not None and doc_count >= args.limit:
                        stop = True
                        break
                    doc_count += 1
                    part["outer_members_scanned"] += 1
                    locator = {
                        "archive": archive.name,
                        "outer_member": oi.filename,
                        "outer_member_crc32": f"{oi.CRC:08x}",
                        "outer_member_bytes": oi.file_size,
                        "outer_member_timestamp": "%04d-%02d-%02dT%02d:%02d:%02d" % oi.date_time,
                        "outer_member_path_date": (re.search(r"/(\d{8})_", oi.filename).group(1) if re.search(r"/(\d{8})_", oi.filename) else None),
                    }
                    try:
                        inner_bytes = outer.read(oi)
                        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                            xml_infos = [x for x in inner.infolist() if x.filename.lower().endswith(".xml")]
                            if not xml_infos:
                                raise ValueError("inner archive contains no XML member")
                            if len(xml_infos) != 1:
                                part["inner_archives_with_multiple_xml"] += 1
                            xi = xml_infos[0]
                            xml_bytes = inner.read(xi)
                        xml_count += 1
                        part["xml_parsed"] += 1
                        root = ET.fromstring(xml_bytes)
                        meta = document_metadata(root)
                        meta["drugsfda_matches"] = application_matches(meta, apps, products)
                        if meta["drugsfda_matches"]:
                            fda_matched_docs += 1
                        if meta["set_id"]:
                            set_versions[meta["set_id"]].add(meta["version_number"] or "<NULL>")
                        selected, all_features = selected_sections(root)
                        has_explicit = any(f["explicit"] for f in all_features)
                        has_clinical = any(f["clinical"] for f in all_features)
                        if has_explicit:
                            explicit_pk_docs += 1
                            part["documents_with_explicit_pk_section"] += 1
                        if has_clinical:
                            clinical_pharm_docs += 1
                            part["documents_with_clinical_pharmacology_section"] += 1
                        if selected:
                            candidate_docs += 1
                            part["candidate_documents"] += 1
                            if meta["drugsfda_matches"]:
                                fda_matched_candidate_docs += 1
                            if meta["set_id"]:
                                set_candidate_versions[meta["set_id"]].add(meta["version_number"] or "<NULL>")
                            if meta["effective_time"] and len(meta["effective_time"]) >= 4:
                                effective_years[meta["effective_time"][:4]] += 1
                        doc_row = {
                            "schema_version": SCHEMA,
                            "source": "DailyMed human prescription SPL bulk",
                            **locator,
                            "inner_xml_member": xi.filename,
                            "inner_xml_crc32": f"{xi.CRC:08x}",
                            "inner_xml_bytes": xi.file_size,
                            "inner_xml_sha256": sha(xml_bytes),
                            **meta,
                            "has_clinical_pharmacology_section": has_clinical,
                            "has_explicit_pharmacokinetics_section": has_explicit,
                            "candidate_section_count": len(selected),
                            "admission_status": "candidate_inventory_only",
                        }
                        doc_f.write(json.dumps(doc_row, sort_keys=True, ensure_ascii=False) + "\n")

                        for f, reason in selected:
                            candidate_sections += 1
                            part["candidate_sections"] += 1
                            selection_reasons[reason] += 1
                            for endpoint in f["hits"]:
                                endpoints[endpoint] += 1
                            section_codes[(f["code"] or "<NULL>", f["title"])] += 1
                            text = f["text"]
                            flags = {
                                "human_population_mentioned": bool(HUMAN.search(text)),
                                "animal_species_mentioned": bool(ANIMAL.search(text)),
                                "route_mentioned": bool(ROUTE.search(text)),
                                "dose_mentioned": bool(DOSE.search(text)),
                                "matrix_mentioned": bool(MATRIX.search(text)),
                                "unit_like_mentioned": bool(UNIT.search(text)),
                                "population_subgroup_mentioned": bool(POPULATION.search(text)),
                            }
                            flags["all_core_context_flags_in_same_section"] = all(
                                flags[x] for x in ["human_population_mentioned", "route_mentioned", "dose_mentioned", "matrix_mentioned", "unit_like_mentioned"]
                            )
                            for key, value in flags.items():
                                if value:
                                    context_flags[key] += 1
                            tables = [table_record(t, n) for n, t in enumerate(f["tables"], 1)]
                            candidate_tables += len(tables)
                            part["candidate_tables"] += len(tables)
                            section_row = {
                                "schema_version": SCHEMA,
                                "source": "DailyMed human prescription SPL bulk",
                                "candidate_id": sha(f"{archive.name}|{oi.filename}|{xi.filename}|{f['section_id']}|{f['text_sha256']}".encode("utf-8")),
                                **locator,
                                "inner_xml_member": xi.filename,
                                "inner_xml_sha256": sha(xml_bytes),
                                "document_id": meta["document_id"],
                                "set_id": meta["set_id"],
                                "version_number": meta["version_number"],
                                "effective_time": meta["effective_time"],
                                "document_title": meta["document_title"],
                                "author_organizations": meta["author_organizations"],
                                "product_names": meta["product_names"],
                                "product_codes": meta["product_codes"],
                                "active_ingredients": meta["active_ingredients"],
                                "approval_ids": meta["approval_ids"],
                                "drugsfda_matches": meta["drugsfda_matches"],
                                "section_id": f["section_id"],
                                "section_code": f["code"],
                                "section_code_display_name": f["code_display_name"],
                                "section_title": f["title"],
                                "selection_reason": reason,
                                "normalized_section_text_chars": f["text_chars"],
                                "normalized_section_text_sha256": f["text_sha256"],
                                "numeric_token_count": f["numeric"],
                                "endpoint_hits": f["hits"],
                                "evidence_spans": evidence_spans(text, f["hits"]),
                                "table_count": len(tables),
                                "context_completeness_flags": flags,
                                "context_completeness_status": "all_core_flags_detected_unverified" if flags["all_core_context_flags_in_same_section"] else "incomplete_or_not_machine_detected",
                                "admission_status": "candidate_evidence_only_not_a_pk_label",
                            }
                            sec_f.write(json.dumps(section_row, sort_keys=True, ensure_ascii=False) + "\n")
                            for table in tables:
                                table_f.write(json.dumps({
                                    "schema_version": SCHEMA,
                                    "candidate_id": section_row["candidate_id"],
                                    "archive": archive.name,
                                    "outer_member": oi.filename,
                                    "inner_xml_member": xi.filename,
                                    "set_id": meta["set_id"],
                                    "version_number": meta["version_number"],
                                    "effective_time": meta["effective_time"],
                                    "section_id": f["section_id"],
                                    "section_title": f["title"],
                                    **table,
                                    "admission_status": "candidate_table_inventory_only",
                                }, sort_keys=True, ensure_ascii=False) + "\n")
                    except Exception as exc:
                        part["errors"] += 1
                        errors.append({**locator, "error_type": type(exc).__name__, "error": str(exc)})
                        doc_f.write(json.dumps({
                            "schema_version": SCHEMA,
                            **locator,
                            "parse_status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "admission_status": "not_parsed",
                        }, sort_keys=True) + "\n")
                    if doc_count % 2000 == 0:
                        print(f"progress documents={doc_count} candidates={candidate_docs} sections={candidate_sections} errors={len(errors)} elapsed_s={time.time()-t0:.1f}", flush=True)
            archive_summary.append({"archive": archive.name, **dict(part)})
            if stop:
                break

    summary = {
        "schema_version": SCHEMA,
        "generated_at_utc": generated,
        "scan_scope": (f"single frozen archive {args.archive}" if args.archive else "all frozen DailyMed human-prescription SPL release members") if args.limit is None else f"test limit {args.limit}",
        "candidate_definition": {
            "explicit": "Top-most section with pharmacokinetics title or LOINC 43682-4, at least one PK endpoint term, and >=2 numeric tokens or a table.",
            "fallback": "Clinical Pharmacology title or LOINC 34090-1 meeting the quantitative endpoint test only when no qualifying explicit PK descendant was selected.",
            "pk_endpoints": list(ENDPOINTS),
            "important_limit": "Keyword/regex inventory only. Numeric values are not parsed into PK labels and context flags are not human-verified.",
        },
        "counts": {
            "outer_members_scanned": doc_count,
            "inner_xml_members_parsed": xml_count,
            "parse_errors": len(errors),
            "documents_with_clinical_pharmacology_section": clinical_pharm_docs,
            "documents_with_explicit_pharmacokinetics_section": explicit_pk_docs,
            "candidate_documents": candidate_docs,
            "candidate_sections": candidate_sections,
            "candidate_tables": candidate_tables,
            "unique_set_ids_all_documents": len(set_versions),
            "unique_set_ids_candidate_documents": len(set_candidate_versions),
            "set_ids_with_multiple_versions_all_documents": sum(len(v) > 1 for v in set_versions.values()),
            "set_ids_with_multiple_candidate_versions": sum(len(v) > 1 for v in set_candidate_versions.values()),
            "documents_with_exact_drugsfda_application_match": fda_matched_docs,
            "candidate_documents_with_exact_drugsfda_application_match": fda_matched_candidate_docs,
            "canonical_rows": 0,
            "training_labels": 0,
        },
        "by_archive": archive_summary,
        "candidate_sections_by_selection_reason": dict(selection_reasons),
        "candidate_sections_by_endpoint_hit_nonexclusive": dict(endpoints),
        "candidate_documents_by_effective_year": dict(sorted(effective_years.items())),
        "candidate_sections_by_context_flag_nonexclusive": dict(context_flags),
        "top_section_code_title_pairs": [
            {"section_code": key[0], "section_title": key[1], "candidate_sections": count}
            for key, count in section_codes.most_common(100)
        ],
        "parse_errors": errors,
        "source_bindings": {
            "dailymed_manifest": "research/data/platform/raw/external_public/dailymed_spl_v2_human_rx/dailymed_spl_v2_human_rx_manifest.json",
            "drugsfda": fda_binding,
        },
        "rights_and_admission": "Evidence excerpts are bounded review aids. Preserve DailyMed SETID/version/effectiveTime, SPL author, XML locator/hash, and FDA/NLM terms. No candidate is a normalized or validated PK label.",
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (out / "scan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    artifacts = []
    for path in sorted(out.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)})
    manifest = {
        "schema_version": SCHEMA,
        "generated_at_utc": generated,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_bytes": sum(x["bytes"] for x in artifacts),
        "manifest_excluded_as_self_referential": "manifest.json",
        "canonical_rows": 0,
        "training_labels": 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
