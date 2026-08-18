import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error("Usage: node build_assay_workbook.mjs INPUT.json OUTPUT.xlsx");
}

const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();
const navy = "#17324D";
const teal = "#087E8B";
const paleBlue = "#EAF2F8";
const paleGreen = "#EAF7F0";
const paleAmber = "#FFF4D6";
const border = "#CAD5E0";

function columnName(index) {
  let name = "";
  let value = index + 1;
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

function normalizeRows(rows) {
  return rows.map((row) => Object.fromEntries(Object.entries(row).map(([key, value]) => [
    key,
    value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value)) ? null : value,
  ])));
}

function projectRows(rows, columns) {
  return (rows || []).map((row) => Object.fromEntries(columns.map((column) => [column, row[column] ?? null])));
}

function addTableSheet(name, title, rows, tableName, note) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const normalized = normalizeRows(rows || []);
  const headers = normalized.length ? Object.keys(normalized[0]) : ["status"];
  const matrix = normalized.length ? normalized.map((row) => headers.map((header) => row[header])) : [["No rows available"]];
  const endColumn = columnName(headers.length - 1);
  sheet.mergeCells(`A1:${endColumn}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
    rowHeight: 30,
    verticalAlignment: "center",
  };
  sheet.mergeCells(`A2:${endColumn}2`);
  sheet.getRange("A2").values = [[note]];
  sheet.getRange("A2").format = {
    fill: paleBlue,
    font: { color: "#31485C", italic: true },
    wrapText: true,
    rowHeight: 34,
  };
  sheet.getRange(`A4:${endColumn}4`).values = [headers];
  sheet.getRange(`A4:${endColumn}4`).format = {
    fill: teal,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: border },
  };
  const endRow = 4 + matrix.length;
  sheet.getRange(`A5:${endColumn}${endRow}`).values = matrix;
  sheet.getRange(`A5:${endColumn}${endRow}`).format = {
    borders: { preset: "all", style: "thin", color: border },
    verticalAlignment: "top",
    wrapText: true,
  };
  sheet.tables.add(`A4:${endColumn}${endRow}`, true, tableName);
  sheet.getRange(`A4:${endColumn}${endRow}`).format.autofitColumns();
  for (let column = 0; column < headers.length; column += 1) {
    const columnLetter = columnName(column);
    const range = sheet.getRange(`${columnLetter}1:${columnLetter}${endRow}`);
    const header = headers[column].toLowerCase();
    if (header.includes("smiles")) {
      range.format.columnWidth = 38;
    } else if (header.includes("conditions") || header.includes("rationale") || header.includes("reporting") || header.includes("local_path") || header === "title") {
      range.format.columnWidth = 40;
    } else if (header.includes("compound") || header.includes("matched_pair")) {
      range.format.columnWidth = 22;
    } else if (header.includes("requested") || header.includes("protocol")) {
      range.format.columnWidth = 24;
    } else if (header.includes("composite") || header.includes("physics")) {
      range.format.columnWidth = 24;
    } else if (header.includes("class") || header.includes("status") || header.includes("extreme")) {
      range.format.columnWidth = 20;
    } else {
      range.format.columnWidth = 16;
    }
    const dataRange = sheet.getRange(`${columnLetter}5:${columnLetter}${endRow}`);
    if (header.includes("priority") || header === "panel_priority") {
      dataRange.format.numberFormat = "0";
    } else if (header === "mw") {
      dataRange.format.numberFormat = "0.00";
    } else if (header.includes("score") || header.includes("pic50") || header.includes("composite") || header.includes("auc") || header.includes("vdss") || header.includes("clearance") || header.includes("rat_cl")) {
      dataRange.format.numberFormat = "0.000";
    }
  }
  sheet.getRange(`A4:${endColumn}${endRow}`).format.autofitRows();
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

const readme = workbook.worksheets.add("Read Me");
readme.showGridLines = false;
readme.mergeCells("A1:F1");
readme.getRange("A1").values = [["Mechanistic PK + hERG assay-request panel"]];
readme.getRange("A1").format = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 18 }, rowHeight: 34 };
const readmeRows = [
  ["Purpose", "Information-gain assay planning for the large-molecule internal series; not molecule ranking or generation."],
  ["Selection", "Quota-constrained across MW regimes, hERG evidence, PK extremes, data gaps, and matched pairs."],
  ["PK scope", "Rat IV/PO exposure. Full profiles require formulation, strain/sex, sampling, LLOQ, and per-animal data."],
  ["hERG scope", "Continuous concentration-response plus state-dependent onset/recovery/trapping for the selected kinetic subset."],
  ["Structure physics", payload.physics_execution_status === "deferred_to_hpc" ? "Deferred to HPC; no local chemical-state, conformer, membrane, or receptor observable is used in this workbook." : String(payload.physics_execution_status || "Not available")],
  ["Interpretation", "Priority orders assay information gain within this panel only; it is not an optimization score."],
  ["Generated", payload.generated_at || "2026-07-21"],
];
readme.getRange("A3:B9").values = readmeRows;
readme.getRange("A3:A9").format = { fill: teal, font: { bold: true, color: "#FFFFFF" }, borders: { preset: "all", style: "thin", color: border } };
readme.getRange("B3:B9").format = { fill: paleBlue, wrapText: true, borders: { preset: "all", style: "thin", color: border } };
readme.getRange("A1:A9").format.columnWidth = 19;
readme.getRange("B1:B9").format.columnWidth = 78;
readme.getRange("A3:B9").format.autofitRows();

const panelColumnCandidates = [
  "panel_priority",
  "compound_id",
  "display_name",
  "mw",
  "mw_bin",
  "herg_class",
  "herg_pic50",
  "rat_cl_ml_kg_min",
  "rat_po_auc_dose_normalized",
  "rat_vdss_l_kg",
  "pk_extreme",
  "matched_pair_ids",
  "information_gain_score",
  "physics_quality_status",
  "physics_model_eligible",
  "composite__environment_conditioned_polarity_response_surrogate",
  "composite__water_to_low_dielectric_folded_fraction_shift_surrogate",
  "composite__rare_state_transport_dominance_surrogate",
  "composite_pka_sensitivity_span__rare_state_transport_dominance_surrogate",
  "standardized_smiles",
];
const requiredPanelColumns = new Set([
  "panel_priority",
  "compound_id",
  "display_name",
  "mw",
  "mw_bin",
  "herg_class",
  "pk_extreme",
  "matched_pair_ids",
  "information_gain_score",
  "standardized_smiles",
]);
const panelColumns = panelColumnCandidates.filter((column) =>
  requiredPanelColumns.has(column)
  || (payload.panel || []).some((row) => row[column] !== null && row[column] !== undefined && row[column] !== ""),
);
const panelRows = projectRows(payload.panel, panelColumns);
addTableSheet("Panel", "16-compound mechanistic assay panel", panelRows, "PanelTable", "Priority is expected information gain for assay planning—not a molecule optimization rank. The canonical CSV retains the full feature matrix; this sheet shows decision-relevant fields.");
addTableSheet("Assay Requests", "Requested in-vitro measurements", payload.assay_requests, "AssayRequestTable", "All assay conditions, nominal/free concentrations, censoring, replicates, and uncertainty should be returned as study-level records.");
addTableSheet("Rat PK Profiles", "Complete rat IV/PO concentration–time subset", payload.pk_profiles, "RatProfileTable", "Request per-animal samples, formulation/vehicle, strain/sex, dose events, sampling times, LLOQ, and NCA derivation details.");
addTableSheet("hERG Kinetics", "State-dependent hERG subset", payload.herg_protocol, "HergKineticTable", "Request onset, recovery, use dependence, and trapping across a documented voltage protocol with free concentrations.");
addTableSheet("Matched Pairs", "Mechanistically informative matched pairs", payload.matched_pairs, "MatchedPairTable", "Pairs are selected for high structural similarity and outcome or evidence discordance; they are prioritized falsification units.");
addTableSheet("Sources", "Sources and provenance notes", payload.sources, "SourceTable", "URLs are plain text for auditability. Source conflicts remain visible and are not resolved by averaging.");

const checks = workbook.worksheets.add("Checks");
checks.showGridLines = false;
checks.mergeCells("A1:D1");
checks.getRange("A1").values = [["Workbook acceptance checks"]];
checks.getRange("A1").format = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 16 }, rowHeight: 30 };
checks.getRange("A3:C10").values = [
  ["Check", "Observed", "Requirement"],
  ["Panel size", null, 16],
  ["MW 650-699", null, payload.quotas?.["650-699"] ?? 3],
  ["MW 700-749", null, payload.quotas?.["700-749"] ?? 6],
  ["MW 750+", null, payload.quotas?.["750+"] ?? 3],
  ["Rat profile subset", null, 8],
  ["hERG kinetics subset", null, 6],
  ["Matched-pair units", null, payload.quotas?.matched_pairs ?? 4],
];
checks.getRange("B4:B10").formulas = [
  [`=COUNTA('Panel'!${columnName(panelColumns.indexOf("compound_id"))}5:${columnName(panelColumns.indexOf("compound_id"))}1000)`],
  [`=COUNTIF('Panel'!${columnName(panelColumns.indexOf("mw_bin"))}5:${columnName(panelColumns.indexOf("mw_bin"))}1000,"650-699")`],
  [`=COUNTIF('Panel'!${columnName(panelColumns.indexOf("mw_bin"))}5:${columnName(panelColumns.indexOf("mw_bin"))}1000,"700-749")`],
  [`=COUNTIF('Panel'!${columnName(panelColumns.indexOf("mw_bin"))}5:${columnName(panelColumns.indexOf("mw_bin"))}1000,"750+")`],
  ["=COUNTA('Rat PK Profiles'!A5:A1000)"],
  ["=COUNTA('hERG Kinetics'!A5:A1000)"],
  ["=COUNTA('Matched Pairs'!A5:A1000)"],
];
checks.getRange("D3:D10").values = [["Status"], [null], [null], [null], [null], [null], [null], [null]];
checks.getRange("D4:D10").formulas = [
  ["=IF(B4=C4,\"PASS\",\"REVIEW\")"],
  ["=IF(B5>=C5,\"PASS\",\"REVIEW\")"],
  ["=IF(B6>=C6,\"PASS\",\"REVIEW\")"],
  ["=IF(B7>=C7,\"PASS\",\"REVIEW\")"],
  ["=IF(B8=C8,\"PASS\",\"REVIEW\")"],
  ["=IF(B9=C9,\"PASS\",\"REVIEW\")"],
  ["=IF(B10>=C10,\"PASS\",\"REVIEW\")"],
];
checks.getRange("A3:D3").format = { fill: teal, font: { bold: true, color: "#FFFFFF" }, borders: { preset: "all", style: "thin", color: border } };
checks.getRange("A4:D10").format = { borders: { preset: "all", style: "thin", color: border } };
checks.getRange("D4:D10").conditionalFormats.add("containsText", { text: "PASS", format: { fill: paleGreen, font: { bold: true, color: "#176B3A" } } });
checks.getRange("D4:D10").conditionalFormats.add("containsText", { text: "REVIEW", format: { fill: paleAmber, font: { bold: true, color: "#8A5A00" } } });
checks.getRange("A1:D10").format.columnWidth = 22;
checks.freezePanes.freezeRows(3);

const outputDir = path.dirname(outputPath);
const renderDir = path.join(outputDir, "rendered_assay_workbook");
await fs.mkdir(renderDir, { recursive: true });

const sheetNames = ["Read Me", "Panel", "Assay Requests", "Rat PK Profiles", "hERG Kinetics", "Matched Pairs", "Sources", "Checks"];
for (const sheetName of sheetNames) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(renderDir, `${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const keyCheck = await workbook.inspect({
  kind: "table",
  range: "Checks!A1:D10",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 6,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, sheetNames, keyCheck: keyCheck.ndjson, formulaErrors: errors.ndjson }));
