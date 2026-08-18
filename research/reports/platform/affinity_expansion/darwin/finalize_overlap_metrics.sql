.bail on
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE INDEX IF NOT EXISTS idx_chembl_compound_inchikey
ON chembl_compounds(standard_inchi_key);

INSERT OR REPLACE INTO overlap_metrics VALUES
('bindingdb_inchikeys_found_in_chembl37',
 (SELECT COUNT(DISTINCT br.ligand_inchikey)
  FROM bdb_rows br
  JOIN chembl_compounds cc ON cc.standard_inchi_key=br.ligand_inchikey
  WHERE br.ligand_inchikey<>''),
 'Distinct exact BindingDB InChIKeys present in the ChEMBL 37 structure table'),
('bindingdb_inchikeys_absent_from_chembl37',
 (SELECT COUNT(DISTINCT br.ligand_inchikey)
  FROM bdb_rows br
  LEFT JOIN chembl_compounds cc ON cc.standard_inchi_key=br.ligand_inchikey
  WHERE br.ligand_inchikey<>'' AND cc.standard_inchi_key IS NULL),
 'Distinct exact BindingDB InChIKeys absent from the ChEMBL 37 structure table'),
('bindingdb_physical_rows_compound_resolved_by_id_or_inchikey',
 (SELECT COUNT(*)
  FROM bdb_rows br
  WHERE EXISTS (SELECT 1 FROM chembl_compounds cc WHERE cc.chembl_id=br.chembl_ligand_id)
     OR EXISTS (SELECT 1 FROM chembl_compounds cc WHERE cc.standard_inchi_key=br.ligand_inchikey)),
 'Physical ChEMBL-tagged rows with at least one exact compound identity route into ChEMBL 37'),
('bindingdb_physical_rows_compound_unresolved_by_id_and_inchikey',
 (SELECT COUNT(*)
  FROM bdb_rows br
  WHERE NOT EXISTS (SELECT 1 FROM chembl_compounds cc WHERE cc.chembl_id=br.chembl_ligand_id)
    AND NOT EXISTS (SELECT 1 FROM chembl_compounds cc WHERE cc.standard_inchi_key=br.ligand_inchikey)),
 'Physical ChEMBL-tagged rows unresolved by both explicit ChEMBL ligand ID and exact InChIKey'),
('explicit_id_key_union_observed',
 (SELECT (SELECT value FROM overlap_metrics WHERE metric='chembl37_unique_resolved_measurement_keys')
       + (SELECT value FROM overlap_metrics WHERE metric='bindingdb_unique_strict_keys')
       - (SELECT value FROM overlap_metrics WHERE metric='bindingdb_unique_strict_keys_matched')),
 'ChEMBL 37 resolved keys plus explicit-ID BindingDB keys minus exact explicit-ID key overlap'),
('combined_identity_key_union_upper_bound',
 (SELECT (SELECT value FROM overlap_metrics WHERE metric='chembl37_unique_resolved_measurement_keys')
       + (SELECT value FROM overlap_metrics WHERE metric='bindingdb_unique_combined_identity_keys')
       - (SELECT value FROM overlap_metrics WHERE metric='bindingdb_unique_combined_identity_keys_matched')),
 'Upper bound under exact ChEMBL-ID-or-InChIKey keying; unmatched keys may still be aliases or source-version drift'),
('combined_identity_key_union_lower_bound',
 (SELECT MAX(value) FROM overlap_metrics WHERE metric IN ('chembl37_unique_resolved_measurement_keys','bindingdb_unique_combined_identity_keys')),
 'Lower bound if every currently unmatched BindingDB key ultimately resolves to an existing ChEMBL key');

ANALYZE;
PRAGMA wal_checkpoint(TRUNCATE);
