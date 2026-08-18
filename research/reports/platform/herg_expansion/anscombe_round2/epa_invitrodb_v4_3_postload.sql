-- EPA invitrodb v4.3 hERG/KCNH2 audit.
-- Run only after the official dump has finished downloading, its gzip has
-- passed `gzip -t`, and it has been loaded into a disposable MySQL database.
-- This script is read-only. It does not admit records to a canonical corpus.

USE invitrodb_v4_3;

-- 0. Fail closed if the expected v4 schema is not present. Review this output
-- before running subsequent statements; do not silently adapt missing fields.
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = DATABASE()
  AND table_name IN (
    'assay', 'assay_component', 'assay_component_endpoint',
    'chemical', 'sample', 'mc4', 'mc5', 'sc2'
  )
ORDER BY table_name, ordinal_position;

-- Expected identity triplets from EPA's official v4.3 annotation workbook:
-- NVS  aid/acid/aeid = 237/423/686
-- ERF  aid/acid/aeid = 897/2914/3184
-- T21  aid/acid/aeid = 910/2939/3210
SELECT a.aid,
       ac.acid,
       ace.aeid,
       a.assay_name,
       ac.assay_component_name,
       ace.assay_component_endpoint_name
FROM assay_component_endpoint AS ace
JOIN assay_component AS ac ON ac.acid = ace.acid
JOIN assay AS a ON a.aid = ac.aid
WHERE ace.aeid IN (686, 3184, 3210)
ORDER BY ace.aeid;

-- 1. Multi-concentration endpoint-sample and substance counts. In v4, mc4
-- carries aeid/spid/m4id and mc5 carries the curve-fit/hit-call result.
SELECT m4.aeid,
       COUNT(*) AS mc5_rows,
       COUNT(DISTINCT m4.m4id) AS distinct_m4ids,
       COUNT(DISTINCT m4.spid) AS distinct_spids,
       COUNT(DISTINCT c.chid) AS distinct_chids,
       COUNT(DISTINCT c.dsstox_substance_id) AS distinct_dtxsids,
       SUM(CASE WHEN m5.hitc >= 0.90 THEN 1 ELSE 0 END) AS active_rows,
       SUM(CASE WHEN m5.hitc >= 0 AND m5.hitc < 0.90 THEN 1 ELSE 0 END) AS inactive_rows,
       SUM(CASE WHEN m5.hitc < 0 THEN 1 ELSE 0 END) AS unintended_direction_rows
FROM mc4 AS m4
JOIN mc5 AS m5 ON m5.m4id = m4.m4id
LEFT JOIN sample AS s ON s.spid = m4.spid
LEFT JOIN chemical AS c ON c.chid = s.chid
WHERE m4.aeid IN (686, 3184, 3210)
GROUP BY m4.aeid
ORDER BY m4.aeid;

-- 2. Single-concentration endpoint-sample and substance counts.
SELECT sc.aeid,
       COUNT(*) AS sc2_rows,
       COUNT(DISTINCT sc.s2id) AS distinct_s2ids,
       COUNT(DISTINCT sc.spid) AS distinct_spids,
       COUNT(DISTINCT c.chid) AS distinct_chids,
       COUNT(DISTINCT c.dsstox_substance_id) AS distinct_dtxsids,
       SUM(CASE WHEN sc.hitc = 1 THEN 1 ELSE 0 END) AS active_rows,
       SUM(CASE WHEN sc.hitc = 0 THEN 1 ELSE 0 END) AS inactive_rows
FROM sc2 AS sc
LEFT JOIN sample AS s ON s.spid = sc.spid
LEFT JOIN chemical AS c ON c.chid = s.chid
WHERE sc.aeid IN (686, 3184, 3210)
GROUP BY sc.aeid
ORDER BY sc.aeid;

-- 3. Exact identity exports. Keep all assay result fields so potency, censoring,
-- fit method, direction and QC can be reconciled before deduplication.
SELECT ace.aeid,
       ace.assay_component_endpoint_name,
       m4.spid,
       c.chid,
       c.dsstox_substance_id,
       c.casn,
       c.chnm,
       m4.*,
       m5.*
FROM assay_component_endpoint AS ace
JOIN mc4 AS m4 ON m4.aeid = ace.aeid
JOIN mc5 AS m5 ON m5.m4id = m4.m4id
LEFT JOIN sample AS s ON s.spid = m4.spid
LEFT JOIN chemical AS c ON c.chid = s.chid
WHERE ace.aeid IN (686, 3184, 3210)
ORDER BY ace.aeid, m4.spid, m4.m4id;

SELECT ace.aeid,
       ace.assay_component_endpoint_name,
       sc.spid,
       c.chid,
       c.dsstox_substance_id,
       c.casn,
       c.chnm,
       sc.*
FROM assay_component_endpoint AS ace
JOIN sc2 AS sc ON sc.aeid = ace.aeid
LEFT JOIN sample AS s ON s.spid = sc.spid
LEFT JOIN chemical AS c ON c.chid = s.chid
WHERE ace.aeid IN (686, 3184, 3210)
ORDER BY ace.aeid, sc.spid, sc.s2id;

-- 4. Within-EPA sample/substance overlap. This is not cross-source novelty;
-- it only quantifies whether the three EPA endpoints tested the same entities.
SELECT x.aeid AS aeid_left,
       y.aeid AS aeid_right,
       COUNT(DISTINCT x.dsstox_substance_id) AS shared_dtxsids
FROM (
  SELECT DISTINCT m4.aeid, c.dsstox_substance_id
  FROM mc4 AS m4
  JOIN sample AS s ON s.spid = m4.spid
  JOIN chemical AS c ON c.chid = s.chid
  WHERE m4.aeid IN (686, 3184, 3210)
    AND c.dsstox_substance_id IS NOT NULL
  UNION
  SELECT DISTINCT sc.aeid, c.dsstox_substance_id
  FROM sc2 AS sc
  JOIN sample AS s ON s.spid = sc.spid
  JOIN chemical AS c ON c.chid = s.chid
  WHERE sc.aeid IN (686, 3184, 3210)
    AND c.dsstox_substance_id IS NOT NULL
) AS x
JOIN (
  SELECT DISTINCT m4.aeid, c.dsstox_substance_id
  FROM mc4 AS m4
  JOIN sample AS s ON s.spid = m4.spid
  JOIN chemical AS c ON c.chid = s.chid
  WHERE m4.aeid IN (686, 3184, 3210)
    AND c.dsstox_substance_id IS NOT NULL
  UNION
  SELECT DISTINCT sc.aeid, c.dsstox_substance_id
  FROM sc2 AS sc
  JOIN sample AS s ON s.spid = sc.spid
  JOIN chemical AS c ON c.chid = s.chid
  WHERE sc.aeid IN (686, 3184, 3210)
    AND c.dsstox_substance_id IS NOT NULL
) AS y
  ON y.dsstox_substance_id = x.dsstox_substance_id
 AND y.aeid > x.aeid
GROUP BY x.aeid, y.aeid
ORDER BY x.aeid, y.aeid;

-- Cross-source novelty gate (outside MySQL): resolve each exported DTXSID to a
-- standardized parent structure, then compare parent InChIKey plus measurement
-- lineage against PubChem AID 1671200, ChEMBL 37 and BindingDB. CAS/name-only or
-- raw-SMILES-only comparisons are insufficient. Do not count the Tox21 rows as
-- new merely because EPA tcpl and PubChem NCATS use different fits/hit calls.
