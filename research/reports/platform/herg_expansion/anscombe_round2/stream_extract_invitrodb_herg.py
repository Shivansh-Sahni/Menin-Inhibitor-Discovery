#!/usr/bin/env python3
"""Fail-closed streaming extraction of three hERG endpoints from invitrodb v4.3.

The parser reads the verified mysqldump gzip sequentially. It parses only the
chemical, sample, mc4, mc5 and sc2 INSERT statements, binds fields to column
order recovered from the dump's own CREATE TABLE statements, and retains mc4
or sc2 rows for AEIDs 686, 3184 and 3210 plus matching mc5 rows. The source
archive is never modified and the full database is never materialized.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path


TARGET_AEIDS = (686, 3184, 3210)
EXPECTED_BYTES = 17_651_807_605
EXPECTED_SHA256 = "ee159e1cdd28996f85db13e742700d8d76ef9d5baf31e3b5e00d249899529c7b"
SELECTED_TABLES = ("chemical", "sample", "mc4", "mc5", "sc2")
INSERT_RE = re.compile(br"^INSERT INTO `([^`]+)` VALUES ")
DROP_RE = re.compile(br"^DROP TABLE IF EXISTS `([^`]+)`;")
COLUMN_RE = re.compile(br"^  `([^`]+)` ")


class HashingReader:
    """Sequential reader that hashes the compressed bytes consumed by gzip."""

    def __init__(self, path: Path):
        self._stream = path.open("rb")
        self.sha256 = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        block = self._stream.read(size)
        if block:
            self.sha256.update(block)
            self.bytes_read += len(block)
        return block

    def tell(self) -> int:
        return self._stream.tell()

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._stream.close()


def quote_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return '"' + value + '"'


def statement_complete(data: bytes) -> bool:
    """mysqldump escapes embedded newlines; a statement ends in semicolon+EOL."""
    return data.rstrip().endswith(b";")


def iter_tuple_fields(values: bytes):
    """Yield lists of raw MySQL literal fields from an extended VALUES list."""
    index = 0
    length = len(values)
    while index < length and values[index] in b" \r\n\t":
        index += 1
    row_number = 0
    while index < length:
        if values[index] == ord(";"):
            index += 1
            break
        if values[index] != ord("("):
            raise ValueError(f"expected '(' at byte {index}, found {values[index:index+20]!r}")
        index += 1
        start = index
        fields: list[bytes] = []
        in_string = False
        escaped = False
        while index < length:
            char = values[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == ord("\\"):
                    escaped = True
                elif char == ord("'"):
                    in_string = False
            else:
                if char == ord("'"):
                    in_string = True
                elif char == ord(","):
                    fields.append(values[start:index])
                    start = index + 1
                elif char == ord(")"):
                    fields.append(values[start:index])
                    index += 1
                    row_number += 1
                    yield fields
                    while index < length and values[index] in b" \r\n\t":
                        index += 1
                    if index < length and values[index] == ord(","):
                        index += 1
                        while index < length and values[index] in b" \r\n\t":
                            index += 1
                        break
                    if index < length and values[index] == ord(";"):
                        index += 1
                        return
                    raise ValueError(
                        f"expected ',' or ';' after row {row_number}, found {values[index:index+20]!r}"
                    )
            index += 1
        else:
            raise ValueError(f"unterminated tuple/string after row {row_number}")
    if values[index:].strip():
        raise ValueError(f"unexpected bytes after VALUES statement: {values[index:index+80]!r}")


def decode_mysql_literal(token: bytes):
    token = token.strip()
    if token == b"NULL":
        return None
    if token.startswith(b"'") and token.endswith(b"'"):
        source = token[1:-1]
        decoded = bytearray()
        index = 0
        escapes = {
            ord("0"): 0,
            ord("b"): 8,
            ord("n"): 10,
            ord("r"): 13,
            ord("t"): 9,
            ord("Z"): 26,
            ord("\\"): 92,
            ord("'"): 39,
            ord('"'): 34,
        }
        while index < len(source):
            char = source[index]
            if char == ord("\\"):
                index += 1
                if index >= len(source):
                    raise ValueError("terminal backslash in SQL string literal")
                char = escapes.get(source[index], source[index])
            decoded.append(char)
            index += 1
        return decoded.decode("utf-8", errors="strict")
    if token.startswith((b"_binary'", b"X'", b"x'", b"B'", b"b'")):
        raise ValueError(f"unsupported typed/binary literal in selected table: {token[:40]!r}")
    return token.decode("ascii", errors="strict")


def create_source_table(connection: sqlite3.Connection, name: str, columns: list[str]) -> None:
    if not columns:
        raise ValueError(f"no columns recovered for {name}")
    column_sql = ", ".join(f"{quote_identifier(column)} TEXT" for column in columns)
    connection.execute(f"CREATE TABLE {quote_identifier('src_' + name)} ({column_sql})")


def insert_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: list[list[object]],
) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO {quote_identifier('src_' + table)} VALUES ({placeholders})",
        rows,
    )


def export_table(connection: sqlite3.Connection, table: str, output: Path, order: str) -> int:
    cursor = connection.execute(f"SELECT * FROM {quote_identifier(table)} ORDER BY {order}")
    columns = [description[0] for description in cursor.description]
    count = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in cursor:
            writer.writerow(row)
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(input_path: Path, output_dir: Path) -> None:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if input_path.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"input size mismatch: {input_path.stat().st_size} != {EXPECTED_BYTES}"
        )
    if "high_value_expansion/lead/toxcast_v4_3" not in input_path.as_posix():
        raise RuntimeError("refusing unexpected input path; expected verified lead/toxcast_v4_3 archive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir == input_path.parent or input_path.parent in output_dir.parents:
        raise RuntimeError("refusing to write inside the lead acquisition directory")

    partial_db = output_dir / "invitrodb_v4_3_herg_audit.sqlite.partial"
    final_db = output_dir / "invitrodb_v4_3_herg_audit.sqlite"
    connection = sqlite3.connect(partial_db)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE source_schema (table_name TEXT, ordinal INTEGER, column_name TEXT, "
        "PRIMARY KEY (table_name, ordinal))"
    )

    columns: dict[str, list[str]] = {}
    current_table: str | None = None
    tables_created: set[str] = set()
    target_m4ids: set[str] = set()
    seen_rows = Counter()
    kept_rows = Counter()
    insert_statements = Counter()
    table_order: list[str] = []
    parse_errors: list[str] = []
    started = time.time()
    next_progress = 256 * 1024 * 1024

    raw = HashingReader(input_path)
    try:
        with gzip.GzipFile(fileobj=raw, mode="rb") as compressed:
            stream = io.BufferedReader(compressed, buffer_size=4 * 1024 * 1024)
            line_number = 0
            iterator = iter(stream)
            for line in iterator:
                line_number += 1
                if raw.bytes_read >= next_progress:
                    elapsed = max(time.time() - started, 0.001)
                    print(
                        f"progress compressed={raw.bytes_read}/{EXPECTED_BYTES} "
                        f"({100 * raw.bytes_read / EXPECTED_BYTES:.1f}%) "
                        f"elapsed={elapsed / 60:.1f}m",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_progress += 256 * 1024 * 1024

                drop_match = DROP_RE.match(line)
                if drop_match:
                    current_table = drop_match.group(1).decode("ascii")
                    table_order.append(current_table)
                    columns.setdefault(current_table, [])
                    continue

                if current_table in SELECTED_TABLES:
                    column_match = COLUMN_RE.match(line)
                    if column_match and current_table not in tables_created:
                        column = column_match.group(1).decode("ascii")
                        columns[current_table].append(column)
                        continue
                    if line.startswith(b") ENGINE=") and current_table not in tables_created:
                        create_source_table(connection, current_table, columns[current_table])
                        connection.executemany(
                            "INSERT INTO source_schema VALUES (?,?,?)",
                            [
                                (current_table, ordinal, column)
                                for ordinal, column in enumerate(columns[current_table], start=1)
                            ],
                        )
                        tables_created.add(current_table)
                        connection.commit()
                        continue

                insert_match = INSERT_RE.match(line)
                if not insert_match:
                    continue
                table = insert_match.group(1).decode("ascii")
                if table not in SELECTED_TABLES:
                    continue
                if table not in tables_created:
                    raise RuntimeError(f"INSERT before schema for {table} at line {line_number}")

                statement = bytearray(line)
                while not statement_complete(statement):
                    try:
                        continuation = next(iterator)
                    except StopIteration as error:
                        raise RuntimeError(f"unterminated INSERT for {table}") from error
                    line_number += 1
                    statement.extend(continuation)
                insert_statements[table] += 1
                values = bytes(statement[insert_match.end():]).rstrip()
                if not values.endswith(b";"):
                    raise RuntimeError(f"missing semicolon for {table} at line {line_number}")

                table_columns = columns[table]
                index = {column: position for position, column in enumerate(table_columns)}
                required = {
                    "chemical": ("chid", "dsstox_substance_id"),
                    "sample": ("spid", "chid"),
                    "mc4": ("m4id", "aeid", "spid"),
                    "mc5": ("m4id", "hitc"),
                    "sc2": ("s2id", "aeid", "spid", "hitc"),
                }[table]
                missing = [field for field in required if field not in index]
                if missing:
                    raise RuntimeError(f"missing required {table} fields: {missing}")

                batch: list[list[object]] = []
                for raw_fields in iter_tuple_fields(values):
                    seen_rows[table] += 1
                    if len(raw_fields) != len(table_columns):
                        raise RuntimeError(
                            f"field-count mismatch in {table}: {len(raw_fields)} != "
                            f"{len(table_columns)} at source row {seen_rows[table]}"
                        )
                    keep = True
                    if table in ("mc4", "sc2"):
                        aeid = int(raw_fields[index["aeid"]])
                        keep = aeid in TARGET_AEIDS
                    elif table == "mc5":
                        m4id = raw_fields[index["m4id"]].decode("ascii")
                        keep = m4id in target_m4ids
                    if not keep:
                        continue
                    decoded = [decode_mysql_literal(field) for field in raw_fields]
                    if table == "mc4":
                        target_m4ids.add(str(decoded[index["m4id"]]))
                    batch.append(decoded)
                    kept_rows[table] += 1
                    if len(batch) >= 10_000:
                        insert_rows(connection, table, table_columns, batch)
                        batch.clear()
                insert_rows(connection, table, table_columns, batch)
                connection.commit()
    except Exception as error:
        parse_errors.append(f"{type(error).__name__}: {error}")
        connection.close()
        raise
    finally:
        raw.close()

    compressed_sha256 = raw.sha256.hexdigest()
    if raw.bytes_read != EXPECTED_BYTES:
        raise RuntimeError(f"compressed stream not fully consumed: {raw.bytes_read} != {EXPECTED_BYTES}")
    if compressed_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"input SHA-256 mismatch: {compressed_sha256} != {EXPECTED_SHA256}")
    if set(SELECTED_TABLES) != tables_created:
        raise RuntimeError(f"selected schemas missing: {set(SELECTED_TABLES) - tables_created}")
    if table_order != sorted(table_order, key=table_order.index):
        raise RuntimeError("internal table-order invariant failed")
    if table_order.index("mc4") > table_order.index("mc5"):
        raise RuntimeError("mc5 preceded mc4; one-pass m4id filtering is invalid")

    # Index, restrict identity tables to the retained endpoint samples, and prove
    # that every retained SPID/CHID resolves exactly once.
    connection.executescript(
        """
        CREATE UNIQUE INDEX mc4_m4id ON src_mc4(m4id);
        CREATE INDEX mc4_aeid_spid ON src_mc4(aeid, spid);
        CREATE INDEX mc5_m4id ON src_mc5(m4id);
        CREATE INDEX sc2_aeid_spid ON src_sc2(aeid, spid);
        CREATE UNIQUE INDEX sample_spid ON src_sample(spid);
        CREATE UNIQUE INDEX chemical_chid ON src_chemical(chid);

        CREATE TEMP TABLE relevant_spid AS
          SELECT DISTINCT spid FROM src_mc4
          UNION
          SELECT DISTINCT spid FROM src_sc2;
        DELETE FROM src_sample WHERE spid NOT IN (SELECT spid FROM relevant_spid);
        CREATE TEMP TABLE relevant_chid AS SELECT DISTINCT chid FROM src_sample;
        DELETE FROM src_chemical WHERE chid NOT IN (SELECT chid FROM relevant_chid);
        """
    )
    unresolved_spids = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT spid FROM src_mc4 UNION SELECT spid FROM src_sc2
        ) AS wanted LEFT JOIN src_sample USING (spid) WHERE src_sample.spid IS NULL
        """
    ).fetchone()[0]
    unresolved_chids = connection.execute(
        "SELECT COUNT(*) FROM src_sample LEFT JOIN src_chemical USING (chid) "
        "WHERE src_chemical.chid IS NULL"
    ).fetchone()[0]
    duplicate_mc5 = connection.execute(
        "SELECT COUNT(*) FROM (SELECT m4id FROM src_mc5 GROUP BY m4id HAVING COUNT(*) != 1)"
    ).fetchone()[0]
    missing_mc5 = connection.execute(
        "SELECT COUNT(*) FROM src_mc4 LEFT JOIN src_mc5 USING (m4id) WHERE src_mc5.m4id IS NULL"
    ).fetchone()[0]
    orphan_mc5 = connection.execute(
        "SELECT COUNT(*) FROM src_mc5 LEFT JOIN src_mc4 USING (m4id) WHERE src_mc4.m4id IS NULL"
    ).fetchone()[0]
    if any((unresolved_chids, duplicate_mc5, missing_mc5, orphan_mc5)):
        raise RuntimeError(
            "join integrity failure: "
            f"unresolved_spids={unresolved_spids}, unresolved_chids={unresolved_chids}, "
            f"duplicate_mc5={duplicate_mc5}, missing_mc5={missing_mc5}, orphan_mc5={orphan_mc5}"
        )

    # invitrodb can retain source SPIDs that are not mapped into its sample
    # table (the v4.3 release note explicitly discusses this for TOX21). Preserve
    # those assay rows and expose the gap; never invent chemical identities.
    unresolved_sample_rows = [
        {
            "source_table": row[0],
            "aeid": int(row[1]),
            "source_row_id": row[2],
            "spid": row[3],
        }
        for row in connection.execute(
            """
            SELECT 'mc4', CAST(m.aeid AS INTEGER), m.m4id, m.spid
            FROM src_mc4 AS m LEFT JOIN src_sample AS s USING (spid)
            WHERE s.spid IS NULL
            UNION ALL
            SELECT 'sc2', CAST(sc.aeid AS INTEGER), sc.s2id, sc.spid
            FROM src_sc2 AS sc LEFT JOIN src_sample AS s USING (spid)
            WHERE s.spid IS NULL
            ORDER BY 2, 1, 3
            """
        )
    ]

    connection.execute(
        """
        CREATE VIEW endpoint_identity AS
          SELECT DISTINCT m.aeid, c.dsstox_substance_id
          FROM src_mc4 AS m
          JOIN src_sample AS s USING (spid)
          JOIN src_chemical AS c USING (chid)
          WHERE c.dsstox_substance_id IS NOT NULL
        UNION
          SELECT DISTINCT sc.aeid, c.dsstox_substance_id
          FROM src_sc2 AS sc
          JOIN src_sample AS s USING (spid)
          JOIN src_chemical AS c USING (chid)
          WHERE c.dsstox_substance_id IS NOT NULL
        """
    )
    connection.commit()

    counts: dict[str, dict[str, object]] = {}
    for aeid in TARGET_AEIDS:
        mc = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT m4.m4id), COUNT(DISTINCT m4.spid),
                   COUNT(DISTINCT s.chid), COUNT(DISTINCT c.dsstox_substance_id),
                   SUM(CASE WHEN CAST(m5.hitc AS REAL) >= 0.90 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(m5.hitc AS REAL) >= 0 AND CAST(m5.hitc AS REAL) < 0.90
                            THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(m5.hitc AS REAL) < 0 THEN 1 ELSE 0 END)
            FROM src_mc4 AS m4
            JOIN src_mc5 AS m5 USING (m4id)
            LEFT JOIN src_sample AS s USING (spid)
            LEFT JOIN src_chemical AS c USING (chid)
            WHERE CAST(m4.aeid AS INTEGER) = ?
            """,
            (aeid,),
        ).fetchone()
        sc = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT sc.s2id), COUNT(DISTINCT sc.spid),
                   COUNT(DISTINCT s.chid), COUNT(DISTINCT c.dsstox_substance_id),
                   SUM(CASE WHEN CAST(sc.hitc AS REAL) = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(sc.hitc AS REAL) = 0 THEN 1 ELSE 0 END)
            FROM src_sc2 AS sc
            LEFT JOIN src_sample AS s USING (spid)
            LEFT JOIN src_chemical AS c USING (chid)
            WHERE CAST(sc.aeid AS INTEGER) = ?
            """,
            (aeid,),
        ).fetchone()
        identities = connection.execute(
            "SELECT COUNT(*) FROM endpoint_identity WHERE CAST(aeid AS INTEGER) = ?",
            (aeid,),
        ).fetchone()[0]
        counts[str(aeid)] = {
            "mc5_rows": mc[0],
            "distinct_m4ids": mc[1],
            "mc_distinct_spids": mc[2],
            "mc_distinct_chids": mc[3],
            "mc_distinct_dtxsids": mc[4],
            "mc_active_rows_hitc_ge_0_90": mc[5] or 0,
            "mc_inactive_rows_hitc_0_to_lt_0_90": mc[6] or 0,
            "mc_unintended_direction_rows_hitc_lt_0": mc[7] or 0,
            "mc_rows_missing_sample_identity": sum(
                1 for row in unresolved_sample_rows
                if row["source_table"] == "mc4" and row["aeid"] == aeid
            ),
            "sc2_rows": sc[0],
            "distinct_s2ids": sc[1],
            "sc_distinct_spids": sc[2],
            "sc_distinct_chids": sc[3],
            "sc_distinct_dtxsids": sc[4],
            "sc_active_rows_hitc_1": sc[5] or 0,
            "sc_inactive_rows_hitc_0": sc[6] or 0,
            "sc_rows_missing_sample_identity": sum(
                1 for row in unresolved_sample_rows
                if row["source_table"] == "sc2" and row["aeid"] == aeid
            ),
            "union_distinct_dtxsids": identities,
        }

    overlaps = []
    for left_index, left in enumerate(TARGET_AEIDS):
        for right in TARGET_AEIDS[left_index + 1:]:
            shared = connection.execute(
                """
                SELECT COUNT(*) FROM endpoint_identity AS l
                JOIN endpoint_identity AS r USING (dsstox_substance_id)
                WHERE CAST(l.aeid AS INTEGER) = ? AND CAST(r.aeid AS INTEGER) = ?
                """,
                (left, right),
            ).fetchone()[0]
            overlaps.append({"aeid_left": left, "aeid_right": right, "shared_dtxsids": shared})
    shared_all_three = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT dsstox_substance_id FROM endpoint_identity
          GROUP BY dsstox_substance_id HAVING COUNT(DISTINCT aeid) = 3
        )
        """
    ).fetchone()[0]

    csv_counts = {}
    csv_counts["herg_mc4.csv"] = export_table(
        connection, "src_mc4", output_dir / "herg_mc4.csv", "CAST(aeid AS INTEGER), CAST(m4id AS INTEGER)"
    )
    csv_counts["herg_mc5.csv"] = export_table(
        connection, "src_mc5", output_dir / "herg_mc5.csv", "CAST(m4id AS INTEGER)"
    )
    csv_counts["herg_sc2.csv"] = export_table(
        connection, "src_sc2", output_dir / "herg_sc2.csv", "CAST(aeid AS INTEGER), CAST(s2id AS INTEGER)"
    )
    csv_counts["herg_sample.csv"] = export_table(
        connection, "src_sample", output_dir / "herg_sample.csv", "spid"
    )
    csv_counts["herg_chemical.csv"] = export_table(
        connection, "src_chemical", output_dir / "herg_chemical.csv", "CAST(chid AS INTEGER)"
    )
    with (output_dir / "within_epa_overlap.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("aeid_left", "aeid_right", "shared_dtxsids"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(overlaps)
    csv_counts["within_epa_overlap.csv"] = len(overlaps)
    with (output_dir / "unresolved_sample_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("source_table", "aeid", "source_row_id", "spid"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(unresolved_sample_rows)
    csv_counts["unresolved_sample_rows.csv"] = len(unresolved_sample_rows)

    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("VACUUM")
    connection.close()
    os.replace(partial_db, final_db)

    script_path = Path(__file__).resolve()
    summary = {
        "status": (
            "pass_with_source_unmapped_samples" if unresolved_spids else "pass"
        ),
        "fail_closed": True,
        "input": {
            "path": str(input_path),
            "bytes": raw.bytes_read,
            "sha256": compressed_sha256,
            "gzip_crc": "validated by full Python gzip stream consumption",
        },
        "parser": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "selected_tables": list(SELECTED_TABLES),
            "target_aeids": list(TARGET_AEIDS),
            "table_order": table_order,
            "source_rows_seen_selected_tables": dict(seen_rows),
            "rows_retained_before_identity_restriction": dict(kept_rows),
            "insert_statements_parsed": dict(insert_statements),
            "parse_errors": parse_errors,
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "join_integrity": {
            "unresolved_spids": unresolved_spids,
            "unresolved_endpoint_rows": len(unresolved_sample_rows),
            "unresolved_sample_rows_file": "unresolved_sample_rows.csv",
            "identity_policy": (
                "preserved assay rows; no chemical identity imputed; excluded from DTXSID overlap"
            ),
            "unresolved_chids": unresolved_chids,
            "duplicate_mc5_m4ids": duplicate_mc5,
            "missing_mc5_for_target_mc4": missing_mc5,
            "orphan_target_mc5": orphan_mc5,
        },
        "endpoint_counts": counts,
        "within_epa_overlap": overlaps,
        "dtxsids_shared_by_all_three": shared_all_three,
        "csv_data_rows": csv_counts,
        "disposition": "audit-only; zero canonical/training admission",
    }
    summary_path = output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hash_targets = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS.txt"}
    )
    with (output_dir / "SHA256SUMS.txt").open("w", encoding="ascii", newline="") as stream:
        for path in hash_targets:
            stream.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.input, arguments.output_dir)


if __name__ == "__main__":
    main()
