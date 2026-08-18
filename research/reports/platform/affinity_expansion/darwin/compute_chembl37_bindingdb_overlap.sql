.bail on
.mode tabs
.headers off
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=FILE;

DROP TABLE IF EXISTS bdb_rows;
CREATE TABLE bdb_rows (
  bindingdb_reactant_set_id INTEGER,
  bindingdb_monomer_id INTEGER,
  chembl_ligand_id TEXT,
  ligand_inchikey TEXT,
  target_key TEXT,
  target_key_status TEXT,
  declared_chain_count INTEGER,
  resolved_accession_count INTEGER
);
.import --skip 1 __BDB_ROWS_TSV__ bdb_rows

DROP TABLE IF EXISTS bdb_measurements;
CREATE TABLE bdb_measurements (
  bindingdb_reactant_set_id INTEGER,
  bindingdb_monomer_id INTEGER,
  chembl_ligand_id TEXT,
  ligand_inchikey TEXT,
  target_key TEXT,
  target_key_status TEXT,
  endpoint TEXT,
  relation TEXT,
  value_nm REAL,
  value_raw TEXT
);
.import --skip 1 __BDB_MEASUREMENTS_TSV__ bdb_measurements

CREATE INDEX idx_bdb_rows_chembl ON bdb_rows(chembl_ligand_id);
CREATE INDEX idx_bdb_rows_target ON bdb_rows(target_key);
CREATE INDEX idx_bdb_measurement_key ON bdb_measurements(chembl_ligand_id, target_key, endpoint, relation, value_nm);

ATTACH DATABASE '__CHEMBL37_DB__' AS chembl;

DROP TABLE IF EXISTS chembl_target_keys;
CREATE TABLE chembl_target_keys AS
SELECT td.tid,
       td.chembl_id AS chembl_target_id,
       COALESCE((
         SELECT group_concat(accession, ';')
         FROM (
           SELECT DISTINCT cs.accession AS accession
           FROM chembl.target_components tc
           JOIN chembl.component_sequences cs ON cs.component_id = tc.component_id
           WHERE tc.tid = td.tid AND cs.accession IS NOT NULL AND cs.accession <> ''
           ORDER BY cs.accession
         )
       ), '') AS target_key,
       (SELECT COUNT(DISTINCT cs.accession)
        FROM chembl.target_components tc
        JOIN chembl.component_sequences cs ON cs.component_id = tc.component_id
        WHERE tc.tid = td.tid AND cs.accession IS NOT NULL AND cs.accession <> '') AS component_count
FROM chembl.target_dictionary td;
CREATE UNIQUE INDEX idx_chembl_target_tid ON chembl_target_keys(tid);
CREATE INDEX idx_chembl_target_key ON chembl_target_keys(target_key);

DROP TABLE IF EXISTS chembl_compounds;
CREATE TABLE chembl_compounds AS
SELECT md.molregno, md.chembl_id, COALESCE(cs.standard_inchi_key, '') AS standard_inchi_key
FROM chembl.molecule_dictionary md
LEFT JOIN chembl.compound_structures cs ON cs.molregno = md.molregno;
CREATE UNIQUE INDEX idx_chembl_compound_molregno ON chembl_compounds(molregno);
CREATE UNIQUE INDEX idx_chembl_compound_id ON chembl_compounds(chembl_id);

DROP TABLE IF EXISTS chembl_measurement_keys;
CREATE TABLE chembl_measurement_keys AS
SELECT DISTINCT cc.chembl_id,
       ctk.target_key,
       a.standard_type AS endpoint,
       COALESCE(NULLIF(a.standard_relation, ''), '=') AS relation,
       CAST(a.standard_value AS REAL) AS value_nm
FROM chembl.activities a
JOIN chembl_compounds cc ON cc.molregno = a.molregno
JOIN chembl.assays ass ON ass.assay_id = a.assay_id
JOIN chembl_target_keys ctk ON ctk.tid = ass.tid
WHERE a.standard_type IN ('Kd', 'Ki', 'IC50', 'EC50')
  AND a.standard_value IS NOT NULL
  AND a.standard_units = 'nM'
  AND ctk.target_key <> '';
CREATE UNIQUE INDEX idx_chembl_measurement_key
ON chembl_measurement_keys(chembl_id, target_key, endpoint, relation, value_nm);

DROP TABLE IF EXISTS bdb_key_matches;
CREATE TABLE bdb_key_matches AS
SELECT bm.rowid AS bdb_measurement_rowid,
       CASE WHEN cmk.chembl_id IS NULL THEN 0 ELSE 1 END AS matched
FROM bdb_measurements bm
LEFT JOIN chembl_measurement_keys cmk
  ON cmk.chembl_id = bm.chembl_ligand_id
 AND cmk.target_key = bm.target_key
 AND cmk.endpoint = bm.endpoint
 AND cmk.relation = bm.relation
 AND cmk.value_nm = bm.value_nm
WHERE bm.chembl_ligand_id <> ''
  AND bm.target_key <> ''
  AND bm.relation <> 'UNPARSEABLE'
  AND bm.value_nm IS NOT NULL;
CREATE UNIQUE INDEX idx_bdb_key_match_rowid ON bdb_key_matches(bdb_measurement_rowid);

DROP TABLE IF EXISTS overlap_metrics;
CREATE TABLE overlap_metrics(metric TEXT PRIMARY KEY, value INTEGER, note TEXT);

INSERT INTO overlap_metrics VALUES
('bindingdb_chembl_physical_rows', (SELECT COUNT(*) FROM bdb_rows), 'BindingDB physical rows with Curation/DataSource exactly ChEMBL'),
('bindingdb_chembl_unique_reactant_set_ids', (SELECT COUNT(DISTINCT bindingdb_reactant_set_id) FROM bdb_rows), 'Distinct BindingDB record identifiers'),
('bindingdb_chembl_measurement_rows', (SELECT COUNT(*) FROM bdb_measurements), 'One row per populated Kd/Ki/IC50/EC50 field'),
('bindingdb_chembl_unique_monomer_ids', (SELECT COUNT(DISTINCT bindingdb_monomer_id) FROM bdb_rows WHERE bindingdb_monomer_id IS NOT NULL), 'Distinct BindingDB ligand IDs'),
('bindingdb_chembl_unique_ligand_chembl_ids', (SELECT COUNT(DISTINCT chembl_ligand_id) FROM bdb_rows WHERE chembl_ligand_id <> ''), 'Distinct nonblank ChEMBL ligand IDs'),
('bindingdb_chembl_unique_inchikeys', (SELECT COUNT(DISTINCT ligand_inchikey) FROM bdb_rows WHERE ligand_inchikey <> ''), 'Distinct nonblank ligand InChIKeys'),
('bindingdb_chembl_unique_target_keys', (SELECT COUNT(DISTINCT target_key) FROM bdb_rows WHERE target_key <> ''), 'Distinct normalized UniProt accession sets'),
('bindingdb_chembl_rows_missing_ligand_chembl_id', (SELECT COUNT(*) FROM bdb_rows WHERE chembl_ligand_id = ''), 'Cannot use explicit ChEMBL ligand-ID key'),
('bindingdb_chembl_rows_target_complete', (SELECT COUNT(*) FROM bdb_rows WHERE target_key_status = 'complete'), 'Resolved accession count meets or exceeds declared chain count'),
('bindingdb_chembl_rows_target_partial_or_undeclared', (SELECT COUNT(*) FROM bdb_rows WHERE target_key_status = 'partial_or_undeclared'), 'Some accessions resolved but declared chain count not met/available'),
('bindingdb_chembl_rows_target_missing', (SELECT COUNT(*) FROM bdb_rows WHERE target_key_status = 'missing'), 'No Swiss-Prot/TrEMBL primary accession resolved'),
('bindingdb_chembl_measurements_unparseable_value', (SELECT COUNT(*) FROM bdb_measurements WHERE relation = 'UNPARSEABLE'), 'Value was populated but not a single relation plus numeric nM value'),
('bindingdb_ligand_ids_found_in_chembl37', (SELECT COUNT(DISTINCT br.chembl_ligand_id) FROM bdb_rows br JOIN chembl_compounds cc ON cc.chembl_id=br.chembl_ligand_id WHERE br.chembl_ligand_id <> ''), 'Explicit ligand-ID coverage'),
('bindingdb_ligand_ids_absent_from_chembl37', (SELECT COUNT(DISTINCT br.chembl_ligand_id) FROM bdb_rows br LEFT JOIN chembl_compounds cc ON cc.chembl_id=br.chembl_ligand_id WHERE br.chembl_ligand_id <> '' AND cc.chembl_id IS NULL), 'Likely removed/retired IDs or malformed identifiers'),
('bindingdb_ligand_id_inchikey_consistent_rows', (SELECT COUNT(*) FROM bdb_rows br JOIN chembl_compounds cc ON cc.chembl_id=br.chembl_ligand_id WHERE br.ligand_inchikey <> '' AND cc.standard_inchi_key <> '' AND br.ligand_inchikey=cc.standard_inchi_key), 'Same ChEMBL ID and exact standard InChIKey'),
('bindingdb_ligand_id_inchikey_mismatch_rows', (SELECT COUNT(*) FROM bdb_rows br JOIN chembl_compounds cc ON cc.chembl_id=br.chembl_ligand_id WHERE br.ligand_inchikey <> '' AND cc.standard_inchi_key <> '' AND br.ligand_inchikey<>cc.standard_inchi_key), 'Inspect salts, stereochemistry, or identifier drift'),
('bindingdb_ligand_id_inchikey_unresolved_rows', (SELECT COUNT(*) FROM bdb_rows br LEFT JOIN chembl_compounds cc ON cc.chembl_id=br.chembl_ligand_id WHERE br.chembl_ligand_id='' OR cc.chembl_id IS NULL OR br.ligand_inchikey='' OR cc.standard_inchi_key=''), 'One or both structure keys unavailable'),
('bindingdb_target_keys_found_in_chembl37', (SELECT COUNT(DISTINCT br.target_key) FROM bdb_rows br JOIN chembl_target_keys ctk ON ctk.target_key=br.target_key WHERE br.target_key<>''), 'Exact component-set coverage'),
('bindingdb_target_keys_absent_from_chembl37', (SELECT COUNT(DISTINCT br.target_key) FROM bdb_rows br LEFT JOIN chembl_target_keys ctk ON ctk.target_key=br.target_key WHERE br.target_key<>'' AND ctk.target_key IS NULL), 'Target set not represented exactly in ChEMBL 37'),
('bindingdb_measurement_rows_eligible_for_strict_key', (SELECT COUNT(*) FROM bdb_measurements WHERE chembl_ligand_id<>'' AND target_key<>'' AND relation<>'UNPARSEABLE' AND value_nm IS NOT NULL), 'Has ligand ID, target accession set, endpoint, relation, and numeric nM value'),
('bindingdb_measurement_rows_strict_key_matched', (SELECT COUNT(*) FROM bdb_key_matches WHERE matched=1), 'BindingDB measurement rows whose complete standardized key exists in ChEMBL 37'),
('bindingdb_measurement_rows_strict_key_unmatched', (SELECT COUNT(*) FROM bdb_key_matches WHERE matched=0), 'Eligible BindingDB measurement rows with no complete standardized-key hit'),
('bindingdb_unique_strict_keys', (SELECT COUNT(*) FROM (SELECT DISTINCT chembl_ligand_id,target_key,endpoint,relation,value_nm FROM bdb_measurements WHERE chembl_ligand_id<>'' AND target_key<>'' AND relation<>'UNPARSEABLE' AND value_nm IS NOT NULL)), 'Distinct eligible BindingDB measurement keys'),
('bindingdb_unique_strict_keys_matched', (SELECT COUNT(*) FROM (SELECT DISTINCT bm.chembl_ligand_id,bm.target_key,bm.endpoint,bm.relation,bm.value_nm FROM bdb_measurements bm JOIN bdb_key_matches bkm ON bkm.bdb_measurement_rowid=bm.rowid WHERE bkm.matched=1)), 'Distinct BindingDB measurement keys present in ChEMBL 37'),
('chembl37_endpoint_measurement_rows_nm', (SELECT COUNT(*) FROM chembl.activities WHERE standard_type IN ('Kd','Ki','IC50','EC50') AND standard_value IS NOT NULL AND standard_units='nM'), 'All ChEMBL 37 target types/assay families, used for overlap existence'),
('chembl37_unique_resolved_measurement_keys', (SELECT COUNT(*) FROM chembl_measurement_keys), 'Distinct ChEMBL 37 ligand-target-endpoint-relation-value keys with target accession sets');

DROP TABLE IF EXISTS endpoint_overlap;
CREATE TABLE endpoint_overlap AS
SELECT bm.endpoint,
       COUNT(*) AS bindingdb_measurement_rows,
       SUM(CASE WHEN bm.relation='=' THEN 1 ELSE 0 END) AS exact_relation_rows,
       SUM(CASE WHEN bm.relation NOT IN ('=', 'UNPARSEABLE') THEN 1 ELSE 0 END) AS censored_or_approx_rows,
       SUM(CASE WHEN bm.relation='UNPARSEABLE' THEN 1 ELSE 0 END) AS unparseable_rows,
       SUM(CASE WHEN bkm.matched=1 THEN 1 ELSE 0 END) AS strict_key_matched_rows,
       SUM(CASE WHEN bkm.matched=0 THEN 1 ELSE 0 END) AS strict_key_unmatched_eligible_rows
FROM bdb_measurements bm
LEFT JOIN bdb_key_matches bkm ON bkm.bdb_measurement_rowid=bm.rowid
GROUP BY bm.endpoint;

DROP TABLE IF EXISTS relation_overlap;
CREATE TABLE relation_overlap AS
SELECT bm.endpoint, bm.relation,
       COUNT(*) AS bindingdb_measurement_rows,
       SUM(CASE WHEN bkm.matched=1 THEN 1 ELSE 0 END) AS strict_key_matched_rows,
       SUM(CASE WHEN bkm.matched=0 THEN 1 ELSE 0 END) AS strict_key_unmatched_eligible_rows
FROM bdb_measurements bm
LEFT JOIN bdb_key_matches bkm ON bkm.bdb_measurement_rowid=bm.rowid
GROUP BY bm.endpoint,bm.relation;

ANALYZE;
PRAGMA wal_checkpoint(TRUNCATE);
