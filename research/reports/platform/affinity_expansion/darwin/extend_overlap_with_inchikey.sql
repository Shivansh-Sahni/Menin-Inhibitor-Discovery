.bail on
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=FILE;

DROP TABLE IF EXISTS chembl_measurement_keys_inchikey;
CREATE TABLE chembl_measurement_keys_inchikey AS
SELECT DISTINCT cc.standard_inchi_key,
       cmk.target_key,
       cmk.endpoint,
       cmk.relation,
       cmk.value_nm
FROM chembl_measurement_keys cmk
JOIN chembl_compounds cc ON cc.chembl_id = cmk.chembl_id
WHERE cc.standard_inchi_key <> '';
CREATE UNIQUE INDEX idx_chembl_measurement_inchikey
ON chembl_measurement_keys_inchikey(standard_inchi_key, target_key, endpoint, relation, value_nm);

DROP TABLE IF EXISTS bdb_inchikey_matches;
CREATE TABLE bdb_inchikey_matches AS
SELECT bm.rowid AS bdb_measurement_rowid,
       CASE WHEN cmk.standard_inchi_key IS NULL THEN 0 ELSE 1 END AS matched
FROM bdb_measurements bm
LEFT JOIN chembl_measurement_keys_inchikey cmk
  ON cmk.standard_inchi_key = bm.ligand_inchikey
 AND cmk.target_key = bm.target_key
 AND cmk.endpoint = bm.endpoint
 AND cmk.relation = bm.relation
 AND cmk.value_nm = bm.value_nm
WHERE bm.ligand_inchikey <> ''
  AND bm.target_key <> ''
  AND bm.relation <> 'UNPARSEABLE'
  AND bm.value_nm IS NOT NULL;
CREATE UNIQUE INDEX idx_bdb_inchikey_match_rowid ON bdb_inchikey_matches(bdb_measurement_rowid);

DROP TABLE IF EXISTS bdb_combined_matches;
CREATE TABLE bdb_combined_matches AS
SELECT bm.rowid AS bdb_measurement_rowid,
       COALESCE(idm.matched, 0) AS id_matched,
       COALESCE(ikm.matched, 0) AS inchikey_matched,
       CASE WHEN COALESCE(idm.matched,0)=1 OR COALESCE(ikm.matched,0)=1 THEN 1 ELSE 0 END AS combined_matched
FROM bdb_measurements bm
LEFT JOIN bdb_key_matches idm ON idm.bdb_measurement_rowid = bm.rowid
LEFT JOIN bdb_inchikey_matches ikm ON ikm.bdb_measurement_rowid = bm.rowid
WHERE bm.target_key <> ''
  AND bm.relation <> 'UNPARSEABLE'
  AND bm.value_nm IS NOT NULL
  AND (bm.chembl_ligand_id <> '' OR bm.ligand_inchikey <> '');
CREATE UNIQUE INDEX idx_bdb_combined_match_rowid ON bdb_combined_matches(bdb_measurement_rowid);

INSERT OR REPLACE INTO overlap_metrics VALUES
('bindingdb_chembl_rows_with_inchikey', (SELECT COUNT(*) FROM bdb_rows WHERE ligand_inchikey<>''), 'Physical ChEMBL-tagged rows with exact ligand InChIKey'),
('bindingdb_measurement_rows_eligible_for_inchikey_key', (SELECT COUNT(*) FROM bdb_measurements WHERE ligand_inchikey<>'' AND target_key<>'' AND relation<>'UNPARSEABLE' AND value_nm IS NOT NULL), 'Has exact InChIKey, target accession set, endpoint, relation, and numeric nM value'),
('bindingdb_measurement_rows_inchikey_key_matched', (SELECT COUNT(*) FROM bdb_inchikey_matches WHERE matched=1), 'Measurement rows whose exact InChIKey-based standardized key exists in ChEMBL 37'),
('bindingdb_measurement_rows_inchikey_key_unmatched', (SELECT COUNT(*) FROM bdb_inchikey_matches WHERE matched=0), 'Eligible exact InChIKey-based rows without a ChEMBL 37 key hit'),
('bindingdb_measurement_rows_eligible_for_combined_key', (SELECT COUNT(*) FROM bdb_combined_matches), 'Has target/value semantics and at least one ligand identity route'),
('bindingdb_measurement_rows_combined_key_matched', (SELECT COUNT(*) FROM bdb_combined_matches WHERE combined_matched=1), 'Matched through exact ChEMBL ID or exact InChIKey standardized key'),
('bindingdb_measurement_rows_combined_key_unmatched', (SELECT COUNT(*) FROM bdb_combined_matches WHERE combined_matched=0), 'Eligible through at least one identity route but no standardized key hit'),
('bindingdb_measurement_rows_matched_by_both_routes', (SELECT COUNT(*) FROM bdb_combined_matches WHERE id_matched=1 AND inchikey_matched=1), 'Concordant exact-ID and exact-InChIKey key hit'),
('bindingdb_measurement_rows_matched_by_id_only', (SELECT COUNT(*) FROM bdb_combined_matches WHERE id_matched=1 AND inchikey_matched=0), 'ID key matched; InChIKey key did not'),
('bindingdb_measurement_rows_matched_by_inchikey_only', (SELECT COUNT(*) FROM bdb_combined_matches WHERE id_matched=0 AND inchikey_matched=1), 'InChIKey key matched; ID key absent or did not match'),
('bindingdb_unique_combined_identity_keys', (SELECT COUNT(*) FROM (SELECT DISTINCT COALESCE(NULLIF(chembl_ligand_id,''), 'IK:'||ligand_inchikey) AS ligand_identity,target_key,endpoint,relation,value_nm FROM bdb_measurements WHERE target_key<>'' AND relation<>'UNPARSEABLE' AND value_nm IS NOT NULL AND (chembl_ligand_id<>'' OR ligand_inchikey<>''))), 'Distinct BindingDB keys using ChEMBL ID when present, otherwise exact InChIKey'),
('bindingdb_unique_combined_identity_keys_matched', (SELECT COUNT(*) FROM (SELECT DISTINCT COALESCE(NULLIF(bm.chembl_ligand_id,''), 'IK:'||bm.ligand_inchikey) AS ligand_identity,bm.target_key,bm.endpoint,bm.relation,bm.value_nm FROM bdb_measurements bm JOIN bdb_combined_matches bcm ON bcm.bdb_measurement_rowid=bm.rowid WHERE bcm.combined_matched=1)), 'Distinct combined-identity BindingDB keys with a ChEMBL 37 hit');

DROP TABLE IF EXISTS endpoint_overlap_combined;
CREATE TABLE endpoint_overlap_combined AS
SELECT bm.endpoint,
       COUNT(*) AS bindingdb_measurement_rows,
       SUM(CASE WHEN bcm.combined_matched=1 THEN 1 ELSE 0 END) AS combined_key_matched_rows,
       SUM(CASE WHEN bcm.combined_matched=0 THEN 1 ELSE 0 END) AS combined_key_unmatched_eligible_rows,
       SUM(CASE WHEN bcm.id_matched=1 AND bcm.inchikey_matched=1 THEN 1 ELSE 0 END) AS both_routes_rows,
       SUM(CASE WHEN bcm.id_matched=1 AND bcm.inchikey_matched=0 THEN 1 ELSE 0 END) AS id_only_rows,
       SUM(CASE WHEN bcm.id_matched=0 AND bcm.inchikey_matched=1 THEN 1 ELSE 0 END) AS inchikey_only_rows
FROM bdb_measurements bm
LEFT JOIN bdb_combined_matches bcm ON bcm.bdb_measurement_rowid=bm.rowid
GROUP BY bm.endpoint;

ANALYZE;
PRAGMA wal_checkpoint(TRUNCATE);
