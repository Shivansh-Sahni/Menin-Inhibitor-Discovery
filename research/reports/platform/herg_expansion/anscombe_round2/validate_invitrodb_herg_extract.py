#!/usr/bin/env python3
"""Independent arithmetic/content validation for the HREG-5 extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sqlite3
from pathlib import Path


AEIDS = (686, 3184, 3210)
TABLE_EXPORTS = {
    "src_mc4": ("herg_mc4.csv", "CAST(aeid AS INTEGER), CAST(m4id AS INTEGER)"),
    "src_mc5": ("herg_mc5.csv", "CAST(m4id AS INTEGER)"),
    "src_sc2": ("herg_sc2.csv", "CAST(aeid AS INTEGER), CAST(s2id AS INTEGER)"),
    "src_sample": ("herg_sample.csv", "spid"),
    "src_chemical": ("herg_chemical.csv", "CAST(chid AS INTEGER)"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    root = arguments.extract_dir.resolve()
    summary = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))

    listed_hashes = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        listed_hashes[name] = digest
    hash_mismatches = {
        name: {"expected": digest, "observed": sha256(root / name)}
        for name, digest in listed_hashes.items()
        if sha256(root / name) != digest
    }
    if hash_mismatches:
        raise RuntimeError(f"output hash mismatch: {hash_mismatches}")

    parser_path = Path(summary["parser"]["path"])
    parser_hash_matches = sha256(parser_path) == summary["parser"]["sha256"]
    if not parser_hash_matches:
        raise RuntimeError("extractor source changed after the recorded run")

    connection = sqlite3.connect(f"file:{root / 'invitrodb_v4_3_herg_audit.sqlite'}?mode=ro", uri=True)
    csv_checks = {}
    for table, (filename, order) in TABLE_EXPORTS.items():
        cursor = connection.execute(f"SELECT * FROM {table} ORDER BY {order}")
        expected_header = [column[0] for column in cursor.description]
        csv_path = root / filename
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            observed_header = next(reader)
            if observed_header != expected_header:
                raise RuntimeError(f"header mismatch for {filename}")
            rows_checked = 0
            for number, (db_row, csv_row) in enumerate(
                itertools.zip_longest(cursor, reader), start=1
            ):
                if db_row is None or csv_row is None:
                    raise RuntimeError(f"row-count mismatch for {filename} at {number}")
                normalized = ["" if value is None else str(value) for value in db_row]
                if normalized != csv_row:
                    raise RuntimeError(f"content mismatch for {filename} at data row {number}")
                rows_checked += 1
        if rows_checked != summary["csv_data_rows"][filename]:
            raise RuntimeError(f"summary row-count mismatch for {filename}")
        csv_checks[filename] = {"rows": rows_checked, "matches_sqlite": True}

    endpoint_counts = {}
    for aeid in AEIDS:
        mc = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT m.m4id), COUNT(DISTINCT m.spid),
                   COUNT(DISTINCT s.chid), COUNT(DISTINCT c.dsstox_substance_id),
                   SUM(CASE WHEN CAST(f.hitc AS REAL) >= 0.9 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(f.hitc AS REAL) >= 0 AND CAST(f.hitc AS REAL) < 0.9
                            THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(f.hitc AS REAL) < 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN s.spid IS NULL THEN 1 ELSE 0 END)
            FROM src_mc4 m JOIN src_mc5 f USING (m4id)
            LEFT JOIN src_sample s USING (spid) LEFT JOIN src_chemical c USING (chid)
            WHERE CAST(m.aeid AS INTEGER)=?
            """,
            (aeid,),
        ).fetchone()
        mc = tuple(0 if value is None and index >= 5 else value for index, value in enumerate(mc))
        sc = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT sc.s2id), COUNT(DISTINCT sc.spid),
                   COUNT(DISTINCT s.chid), COUNT(DISTINCT c.dsstox_substance_id),
                   SUM(CASE WHEN CAST(sc.hitc AS REAL)=1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(sc.hitc AS REAL)=0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(sc.hitc AS REAL)<0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN sc.hitc IS NULL OR CAST(sc.hitc AS REAL)>1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN s.spid IS NULL THEN 1 ELSE 0 END)
            FROM src_sc2 sc LEFT JOIN src_sample s USING (spid)
            LEFT JOIN src_chemical c USING (chid)
            WHERE CAST(sc.aeid AS INTEGER)=?
            """,
            (aeid,),
        ).fetchone()
        sc = tuple(0 if value is None and index >= 5 else value for index, value in enumerate(sc))
        mc_resolved = connection.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN CAST(f.hitc AS REAL)>=0.9 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(f.hitc AS REAL)>=0 AND CAST(f.hitc AS REAL)<0.9
                            THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(f.hitc AS REAL)<0 THEN 1 ELSE 0 END)
            FROM src_mc4 m JOIN src_mc5 f USING (m4id) JOIN src_sample s USING (spid)
            WHERE CAST(m.aeid AS INTEGER)=?
            """,
            (aeid,),
        ).fetchone()
        mc_resolved = tuple(0 if value is None else value for value in mc_resolved)
        sc_resolved = connection.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN CAST(sc.hitc AS REAL)=1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(sc.hitc AS REAL)=0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(sc.hitc AS REAL)<0 THEN 1 ELSE 0 END)
            FROM src_sc2 sc JOIN src_sample s USING (spid)
            WHERE CAST(sc.aeid AS INTEGER)=?
            """,
            (aeid,),
        ).fetchone()
        sc_resolved = tuple(0 if value is None else value for value in sc_resolved)
        union_dtxsid = connection.execute(
            "SELECT COUNT(*) FROM endpoint_identity WHERE CAST(aeid AS INTEGER)=?", (aeid,)
        ).fetchone()[0]
        mc_sc_shared = connection.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT DISTINCT c.dsstox_substance_id
              FROM src_mc4 m JOIN src_sample s USING (spid) JOIN src_chemical c USING (chid)
              WHERE CAST(m.aeid AS INTEGER)=?
              INTERSECT
              SELECT DISTINCT c.dsstox_substance_id
              FROM src_sc2 sc JOIN src_sample s USING (spid) JOIN src_chemical c USING (chid)
              WHERE CAST(sc.aeid AS INTEGER)=?
            )
            """,
            (aeid, aeid),
        ).fetchone()[0]
        if mc[5] + mc[6] + mc[7] != mc[0]:
            raise RuntimeError(f"mc hit-call arithmetic failed for {aeid}: {mc}")
        if sc[5] + sc[6] + sc[7] + sc[8] != sc[0]:
            raise RuntimeError(f"sc hit-call arithmetic failed for {aeid}: {sc}")
        if sum(mc_resolved[1:]) != mc_resolved[0]:
            raise RuntimeError(f"resolved mc hit-call arithmetic failed for {aeid}")
        if sum(sc_resolved[1:]) != sc_resolved[0]:
            raise RuntimeError(f"resolved sc hit-call arithmetic failed for {aeid}")
        if union_dtxsid != mc[4] + sc[4] - mc_sc_shared:
            raise RuntimeError(f"within-endpoint identity union failed for {aeid}")
        endpoint_counts[str(aeid)] = {
            "mc_rows": mc[0],
            "mc_distinct_m4ids": mc[1],
            "mc_distinct_spids": mc[2],
            "mc_distinct_chids": mc[3],
            "mc_distinct_dtxsids": mc[4],
            "mc_hitc_active_ge_0_90": mc[5],
            "mc_hitc_inactive_0_to_lt_0_90": mc[6],
            "mc_hitc_negative_unintended_direction": mc[7],
            "mc_rows_missing_sample_identity": mc[8],
            "mc_resolved_sample_rows": mc_resolved[0],
            "mc_resolved_hitc_active_ge_0_90": mc_resolved[1],
            "mc_resolved_hitc_inactive_0_to_lt_0_90": mc_resolved[2],
            "mc_resolved_hitc_negative_unintended_direction": mc_resolved[3],
            "sc_rows": sc[0],
            "sc_distinct_s2ids": sc[1],
            "sc_distinct_spids": sc[2],
            "sc_distinct_chids": sc[3],
            "sc_distinct_dtxsids": sc[4],
            "sc_hitc_active_1": sc[5],
            "sc_hitc_inactive_0": sc[6],
            "sc_hitc_negative_unintended_direction": sc[7],
            "sc_hitc_other_or_null": sc[8],
            "sc_rows_missing_sample_identity": sc[9],
            "sc_resolved_sample_rows": sc_resolved[0],
            "sc_resolved_hitc_active_1": sc_resolved[1],
            "sc_resolved_hitc_inactive_0": sc_resolved[2],
            "sc_resolved_hitc_negative_unintended_direction": sc_resolved[3],
            "mc_sc_shared_dtxsids": mc_sc_shared,
            "union_distinct_dtxsids": union_dtxsid,
        }

    pairwise = []
    for left_index, left in enumerate(AEIDS):
        for right in AEIDS[left_index + 1:]:
            shared = connection.execute(
                """
                SELECT COUNT(*) FROM endpoint_identity l JOIN endpoint_identity r
                  USING (dsstox_substance_id)
                WHERE CAST(l.aeid AS INTEGER)=? AND CAST(r.aeid AS INTEGER)=?
                """,
                (left, right),
            ).fetchone()[0]
            pairwise.append({"aeid_left": left, "aeid_right": right, "shared_dtxsids": shared})
    all_three = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT dsstox_substance_id FROM endpoint_identity
          GROUP BY dsstox_substance_id HAVING COUNT(DISTINCT aeid)=3
        )
        """
    ).fetchone()[0]
    union_all = connection.execute(
        "SELECT COUNT(DISTINCT dsstox_substance_id) FROM endpoint_identity"
    ).fetchone()[0]
    inclusion_exclusion = (
        sum(endpoint_counts[str(aeid)]["union_distinct_dtxsids"] for aeid in AEIDS)
        - sum(item["shared_dtxsids"] for item in pairwise)
        + all_three
    )
    if union_all != inclusion_exclusion:
        raise RuntimeError(f"three-way inclusion-exclusion failed: {union_all} != {inclusion_exclusion}")

    membership_path = root / "endpoint_dtxsid_membership.csv"
    coverage_classes: dict[str, int] = {}
    membership_rows = 0
    membership_cursor = connection.execute(
        """
        SELECT dsstox_substance_id,
               MAX(CASE WHEN CAST(aeid AS INTEGER)=686 THEN 1 ELSE 0 END) AS in_686,
               MAX(CASE WHEN CAST(aeid AS INTEGER)=3184 THEN 1 ELSE 0 END) AS in_3184,
               MAX(CASE WHEN CAST(aeid AS INTEGER)=3210 THEN 1 ELSE 0 END) AS in_3210
        FROM endpoint_identity GROUP BY dsstox_substance_id ORDER BY dsstox_substance_id
        """
    )
    with membership_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("dsstox_substance_id", "in_aeid_686", "in_aeid_3184", "in_aeid_3210", "coverage_class"))
        for dtxsid, in_686, in_3184, in_3210 in membership_cursor:
            members = []
            if in_686:
                members.append("686")
            if in_3184:
                members.append("3184")
            if in_3210:
                members.append("3210")
            coverage_class = "+".join(members)
            coverage_classes[coverage_class] = coverage_classes.get(coverage_class, 0) + 1
            writer.writerow((dtxsid, in_686, in_3184, in_3210, coverage_class))
            membership_rows += 1
    if membership_rows != union_all:
        raise RuntimeError("membership export row count does not match three-endpoint union")

    unresolved_rows = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT m.spid FROM src_mc4 m LEFT JOIN src_sample s USING (spid) WHERE s.spid IS NULL
          UNION ALL
          SELECT sc.spid FROM src_sc2 sc LEFT JOIN src_sample s USING (spid) WHERE s.spid IS NULL
        )
        """
    ).fetchone()[0]
    unresolved_spids = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT m.spid FROM src_mc4 m LEFT JOIN src_sample s USING (spid) WHERE s.spid IS NULL
          UNION
          SELECT sc.spid FROM src_sc2 sc LEFT JOIN src_sample s USING (spid) WHERE s.spid IS NULL
        )
        """
    ).fetchone()[0]
    connection.close()

    result = {
        "status": "pass",
        "extractor_source_hash_matches_run": parser_hash_matches,
        "listed_output_hashes_verified": len(listed_hashes),
        "hash_mismatches": hash_mismatches,
        "csv_sqlite_content_checks": csv_checks,
        "endpoint_counts": endpoint_counts,
        "pairwise_dtxsid_overlap": pairwise,
        "dtxsids_shared_by_all_three": all_three,
        "union_dtxsids_all_three_endpoints": union_all,
        "endpoint_membership_file": "endpoint_dtxsid_membership.csv",
        "endpoint_membership_rows": membership_rows,
        "coverage_class_counts": dict(sorted(coverage_classes.items())),
        "inclusion_exclusion_reconciled": union_all == inclusion_exclusion,
        "unresolved_endpoint_rows": unresolved_rows,
        "unresolved_unique_spids": unresolved_spids,
        "disposition": "validation only; zero canonical/training admission",
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
