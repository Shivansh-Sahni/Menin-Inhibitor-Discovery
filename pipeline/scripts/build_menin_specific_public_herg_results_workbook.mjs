import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const artifactToolPath = process.env.MENIN_ARTIFACT_TOOL_PATH;
if (!artifactToolPath) {
  throw new Error("Set MENIN_ARTIFACT_TOOL_PATH to the @oai/artifact-tool ESM entry point.");
}
const { SpreadsheetFile, Workbook } = await import(pathToFileURL(path.resolve(artifactToolPath)).href);

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "../..");
const analysisRoot = path.join(root, "research/reports/pk_herg/menin_specific_public_comparison");
const payload = JSON.parse(await fs.readFile(path.join(analysisRoot, "workbook_payload.json"), "utf8"));
const outputDir = path.resolve(process.env.MENIN_WORKBOOK_OUTPUT_DIR ?? path.join(root, "outputs/menin_public_herg"));
const outputPath = path.join(outputDir, "menin_specific_public_herg_integration_results.xlsx");
const previewDir = path.resolve(
  process.env.MENIN_WORKBOOK_PREVIEW_DIR ?? path.join(outputDir, "previews"),
);

const workbook = Workbook.create();
const C = {
  navy: "#17324D", blue: "#2C6E91", paleBlue: "#EAF3F8", green: "#3D6B4F",
  paleGreen: "#EAF4ED", amber: "#A66B00", paleAmber: "#FFF4D6", red: "#9B2226",
  paleRed: "#FBE9E7", gray: "#5D6870", paleGray: "#F5F7F8", white: "#FFFFFF",
  ink: "#1C2833", border: "#D4DCE2",
};

function colLetter(i) {
  let n = i + 1;
  let out = "";
  while (n) {
    const r = (n - 1) % 26;
    out = String.fromCharCode(65 + r) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function clean(v) {
  if (v === undefined || v === null) return null;
  if (typeof v === "number" && !Number.isFinite(v)) return v > 0 ? "+Inf" : "-Inf";
  return v;
}

function titleBand(sheet, columns, title, subtitle) {
  const end = colLetter(columns - 1);
  sheet.getRange(`A1:${end}1`).values = [[title, ...Array(columns - 1).fill(null)]];
  sheet.getRange(`A1:${end}1`).format = {
    fill: C.navy, font: { bold: true, color: C.white, size: 17 }, verticalAlignment: "center",
  };
  sheet.getRange(`A1:${end}1`).format.rowHeight = 34;
  sheet.getRange(`A2:${end}2`).values = [[subtitle, ...Array(columns - 1).fill(null)]];
  sheet.getRange(`A2:${end}2`).format = {
    fill: C.paleBlue, font: { italic: true, color: C.gray, size: 10 }, verticalAlignment: "center",
  };
  sheet.getRange(`A2:${end}2`).format.rowHeight = 28;
  sheet.showGridLines = false;
}

function sectionBand(sheet, row, columns, text, fill = C.paleBlue, color = C.navy) {
  const end = colLetter(columns - 1);
  sheet.getRange(`A${row}:${end}${row}`).values = [[text, ...Array(columns - 1).fill(null)]];
  sheet.getRange(`A${row}:${end}${row}`).format = {
    fill, font: { bold: true, color, size: 11 }, verticalAlignment: "center",
  };
  sheet.getRange(`A${row}:${end}${row}`).format.rowHeight = 24;
}

function widthFor(header) {
  const h = String(header).toLowerCase();
  if (h.includes("interpret") || h.includes("reason") || h.includes("finding")) return 54;
  if (h.includes("regime") || h.includes("baseline") || h.includes("status")) return 35;
  if (h.includes("record") || h.includes("compound") || h.includes("series")) return 25;
  if (h.includes("source") || h.includes("role") || h.includes("model")) return 27;
  if (h.includes("exclusion")) return 55;
  if (h.includes("quantity")) return 48;
  return 17;
}

function writeTable(sheet, startRow, records, headers, opts = {}) {
  const rows = records.map((record) => headers.map((header) => clean(record[header])));
  const end = colLetter(headers.length - 1);
  const lastRow = startRow + Math.max(rows.length, 1);
  sheet.getRange(`A${startRow}:${end}${startRow}`).values = [headers];
  sheet.getRange(`A${startRow}:${end}${startRow}`).format = {
    fill: opts.headerFill ?? C.blue,
    font: { bold: true, color: C.white, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: C.border },
  };
  sheet.getRange(`A${startRow}:${end}${startRow}`).format.rowHeight = 34;
  if (rows.length) {
    sheet.getRange(`A${startRow + 1}:${end}${lastRow}`).values = rows;
    sheet.getRange(`A${startRow + 1}:${end}${lastRow}`).format = {
      font: { color: C.ink, size: 9 }, wrapText: true, verticalAlignment: "top",
      borders: { insideHorizontal: { style: "thin", color: C.border }, outside: { style: "thin", color: C.border } },
    };
    sheet.getRange(`A${startRow + 1}:${end}${lastRow}`).format.rowHeight = opts.rowHeight ?? 30;
    for (let row = startRow + 1; row <= lastRow; row += 2) {
      sheet.getRange(`A${row}:${end}${row}`).format.fill = opts.altFill ?? C.paleGray;
    }
  }
  headers.forEach((header, index) => {
    sheet.getRangeByIndexes(0, index, Math.max(lastRow, 2), 1).format.columnWidth =
      opts.widths?.[header] ?? widthFor(header);
  });
  return { lastRow, end };
}

function findMetric(regime, evaluation, model = null) {
  return payload.locked_new_results.find((row) =>
    row.comparison_regime === regime && row.evaluation === evaluation && (!model || row.model === model));
}

function findPrior(regime, evaluation) {
  return payload.retained_comparators.find((row) =>
    row.comparison_regime === regime && row.evaluation === evaluation);
}

const evaluationNames = {
  internal_scaffold_cv: "Internal scaffold CV",
  angelo_fixed_nonoverlap: "Fixed Angelo non-overlap",
};

const comparisonRows = [];
for (const evaluation of ["internal_scaffold_cv", "angelo_fixed_nonoverlap"]) {
  const base = findPrior("retained_internal_only", evaluation);
  const broad = findPrior("retained_internal_plus_broad_public", evaluation);
  const censoredControl = findMetric("internal_only_censored", evaluation, "censored_ridge");
  const meninExact = findMetric("internal_plus_menin_public_exact", evaluation, "ridge");
  const meninCensored = findMetric("internal_plus_menin_public_censored", evaluation, "censored_ridge");
  const externalOnly = findMetric("menin_public_only_censored", evaluation, "censored_ridge");
  comparisonRows.push(
    { evaluation: evaluationNames[evaluation], regime: "Internal only (retained)", endpoint_handling: "Exact pIC50", feature_model: "Hybrid / ridge", n: base.n, mae: base.mae, rmse: base.rmse, spearman: base.spearman, delta_vs_matched_internal: 0, delta_ci_95: "0 to 0", interpretation: "Current locked reference." },
    { evaluation: evaluationNames[evaluation], regime: "Internal + broad public", endpoint_handling: "Exact pIC50", feature_model: "Hybrid / ridge", n: broad.n, mae: broad.mae, rmse: broad.rmse, spearman: broad.spearman, delta_vs_matched_internal: broad.mae_delta_vs_internal, delta_ci_95: "Not recomputed here", interpretation: "Retained broad-public comparator; not Menin-specific." },
    { evaluation: evaluationNames[evaluation], regime: "Internal only (matched censored estimator)", endpoint_handling: "Exact likelihood", feature_model: "Hybrid / censored ridge", n: censoredControl.n, mae: censoredControl.mae, rmse: censoredControl.rmse, spearman: censoredControl.spearman, delta_vs_matched_internal: 0, delta_ci_95: "0 to 0", interpretation: "Estimator control required to isolate the effect of public data." },
    { evaluation: evaluationNames[evaluation], regime: "Internal + Menin-public exact", endpoint_handling: "9 exact public", feature_model: "Hybrid / ridge", n: meninExact.n, mae: meninExact.mae, rmse: meninExact.rmse, spearman: meninExact.spearman, delta_vs_matched_internal: meninExact.mae_delta_vs_internal, delta_ci_95: `${meninExact.mae_delta_lower_95.toFixed(3)} to ${meninExact.mae_delta_upper_95.toFixed(3)}`, interpretation: "Locked comparison: essentially unchanged; CI includes no effect." },
    { evaluation: evaluationNames[evaluation], regime: "Internal + Menin-public censored", endpoint_handling: "9 exact + 5 limits", feature_model: "Hybrid / censored ridge", n: meninCensored.n, mae: meninCensored.mae, rmse: meninCensored.rmse, spearman: meninCensored.spearman, delta_vs_matched_internal: meninCensored.mae_delta_vs_internal, delta_ci_95: `${meninCensored.mae_delta_lower_95.toFixed(3)} to ${meninCensored.mae_delta_upper_95.toFixed(3)}`, interpretation: "No robust gain versus the matched censored estimator; discovery track only." },
    { evaluation: evaluationNames[evaluation], regime: "Menin-public only", endpoint_handling: "9 exact + 5 limits", feature_model: "Hybrid / censored ridge", n: externalOnly.n, mae: externalOnly.mae, rmse: externalOnly.rmse, spearman: externalOnly.spearman, delta_vs_matched_internal: externalOnly.mae_delta_vs_internal, delta_ci_95: `${externalOnly.mae_delta_lower_95.toFixed(3)} to ${externalOnly.mae_delta_upper_95.toFixed(3)}`, interpretation: "Fails as a transfer model; not deployable." },
  );
}

const summary = workbook.worksheets.add("Summary");
titleBand(summary, 12, "Menin-Specific Public hERG Integration", "Additional comparison only | Existing internal and broad-public work preserved | Generated 2026-08-03");
for (let column = 0; column < 12; column += 1) {
  summary.getRangeByIndexes(0, column, 20, 1).format.columnWidth = 18;
}
const cards = [
  ["A4:C4", "A5:C7", "Primary exact public labels", 9, C.blue, C.paleBlue],
  ["D4:F4", "D5:F7", "Exact + censored public labels", 14, C.green, C.paleGreen],
  ["G4:I4", "G5:I7", "Public training series", 4, C.amber, C.paleAmber],
  ["J4:L4", "J5:L7", "Structure overlaps with tests", 0, C.red, C.paleRed],
];
for (const [labelRange, valueRange, label, value, dark, light] of cards) {
  summary.getRange(labelRange).merge();
  summary.getRange(labelRange).values = [[label]];
  summary.getRange(labelRange).format = { fill: dark, font: { bold: true, color: C.white }, horizontalAlignment: "center", verticalAlignment: "center" };
  summary.getRange(valueRange).merge();
  summary.getRange(valueRange).values = [[value]];
  summary.getRange(valueRange).format = { fill: light, font: { bold: true, color: dark, size: 22 }, horizontalAlignment: "center", verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: C.border } };
}
sectionBand(summary, 9, 12, "Decision summary");
for (const [range, value] of [["A10:C10", "Finding"], ["D10:I10", "Interpretation"], ["J10:L10", "Status"]]) {
  summary.getRange(range).merge();
  summary.getRange(range).values = [[value]];
  summary.getRange(range).format = { fill: C.blue, font: { bold: true, color: C.white, size: 9 }, verticalAlignment: "center", borders: { preset: "all", style: "thin", color: C.border } };
}
payload.interpretation.forEach((record, index) => {
  const row = 11 + index;
  const fill = index % 2 === 0 ? C.paleGray : C.white;
  for (const [range, value] of [[`A${row}:C${row}`, record.finding], [`D${row}:I${row}`, record.interpretation], [`J${row}:L${row}`, record.status]]) {
    summary.getRange(range).merge();
    summary.getRange(range).values = [[value]];
    summary.getRange(range).format = { fill, font: { color: C.ink, size: 9 }, wrapText: true, verticalAlignment: "top", borders: { preset: "all", style: "thin", color: C.border } };
  }
  const statusRange = summary.getRange(`J${row}:L${row}`);
  if (record.status === "Discovery track") statusRange.format = { fill: C.paleAmber, font: { color: C.amber, bold: true, size: 9 }, wrapText: true, verticalAlignment: "top", borders: { preset: "all", style: "thin", color: C.border } };
  if (record.status === "Do not promote" || record.status === "Not deployable") statusRange.format = { fill: C.paleRed, font: { color: C.red, bold: true, size: 9 }, wrapText: true, verticalAlignment: "top", borders: { preset: "all", style: "thin", color: C.border } };
  summary.getRange(`A${row}:L${row}`).format.rowHeight = 48;
});
summary.getRange("A17:L20").merge();
summary.getRange("A17:L20").values = [["Bottom line: adding the Menin-specific public workbook does not yet produce a defensible decision-track improvement. Exact-label hybrid ridge is unchanged; censored integration has no robust advantage over the matched censored estimator; and public-only transfer fails. The data remain valuable as mechanistic/protocol evidence and as a design guide for acquiring independent Menin series."]];
summary.getRange("A17:L20").format = { fill: C.paleAmber, font: { bold: true, color: "#684900", size: 11 }, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: C.amber } };
summary.freezePanes.freezeRows(2);

const locked = workbook.worksheets.add("Locked Comparison");
titleBand(locked, 11, "Locked, Paired Model Comparison", "Same fixed test compounds and folds | Negative MAE delta favors augmentation | No post-hoc model promotion");
writeTable(locked, 4, comparisonRows, ["evaluation", "regime", "endpoint_handling", "feature_model", "n", "mae", "rmse", "spearman", "delta_vs_matched_internal", "delta_ci_95", "interpretation"], {
  rowHeight: 46,
  widths: { evaluation: 28, regime: 43, endpoint_handling: 25, feature_model: 27, delta_ci_95: 25, interpretation: 55 },
});
locked.getRange(`F5:I${4 + comparisonRows.length}`).format.numberFormat = "0.000";
locked.freezePanes.freezeRows(4);

const domain = workbook.worksheets.add("Dataset & Domain");
titleBand(domain, 9, "Dataset Eligibility and Chemical Domain", "Limits remain limits | Duplicate protocols and unresolved structures are not independent training rows");
writeTable(domain, 4, payload.dataset_summary, ["quantity", "value"], { widths: { quantity: 62, value: 20 }, rowHeight: 27 });
sectionBand(domain, 20, 9, "Primary Menin-public IC50 evidence and nearest-domain similarity");
writeTable(domain, 21, payload.public_evidence, ["compound_id", "preferred_name", "series_id", "pIC50_relation", "pic50_lower", "pic50_upper", "max_internal_tanimoto", "max_fixed_angelo_tanimoto"], {
  widths: { preferred_name: 28, series_id: 24 }, rowHeight: 32,
});
domain.getRange(`E22:H${21 + payload.public_evidence.length}`).format.numberFormat = "0.000";
domain.freezePanes.freezeRows(4);

const matrix = workbook.worksheets.add("Full Model Matrix");
titleBand(matrix, 16, "Full Model-Sensitivity Matrix", "All tested feature/model combinations retained | Best row must not be mistaken for prespecified evidence");
const matrixHeaders = ["evaluation", "data_regime", "feature_layer", "model", "n", "mae", "rmse", "r2", "spearman", "mean_signed_error", "fraction_within_0p5_log", "comparison_baseline_regime", "mae_delta_vs_internal", "mae_delta_lower_95", "mae_delta_upper_95", "bootstrap_probability_improved"];
writeTable(matrix, 4, payload.full_model_matrix, matrixHeaders, {
  rowHeight: 27,
  widths: { evaluation: 28, data_regime: 42, feature_layer: 24, model: 22, comparison_baseline_regime: 36 },
});
matrix.getRange(`F5:P${4 + payload.full_model_matrix.length}`).format.numberFormat = "0.000";
matrix.freezePanes.freezeRows(4);

const controls = workbook.worksheets.add("Controls");
titleBand(controls, 10, "Falsification and Reproducibility Controls", "Correct-label permutation, matched-estimator control, fold reconstruction, and zero-overlap checks");
sectionBand(controls, 4, 10, "Exact public-label permutation test");
writeTable(controls, 5, payload.permutation_summary, ["feature_layer", "model", "selection_status", "evaluation", "draws", "observed_mae_delta_vs_matched_internal", "null_mean_delta", "null_lower_95", "null_upper_95", "one_sided_permutation_p_correct_pairing_better"], {
  widths: { selection_status: 34, evaluation: 30, one_sided_permutation_p_correct_pairing_better: 40 }, rowHeight: 40,
});
controls.getRange("F6:J9").format.numberFormat = "0.000";
sectionBand(controls, 12, 10, "Interpretation of the permutation test", C.paleAmber, C.amber);
controls.getRange("A13:J16").merge();
controls.getRange("A13:J16").values = [["The apparent post-hoc hybrid-SVR improvement is reproduced when the nine public hERG labels are shuffled across public structures (permutation p values are not favorable). Therefore the effect is attributable to adding distant structures/altering the learned representation or regularization—not to a validated Menin-specific structure–hERG relationship. The locked hybrid-ridge result also fails to show a significant benefit."]];
controls.getRange("A13:J16").format = { fill: C.paleAmber, font: { color: "#684900", bold: true, size: 10 }, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: C.amber } };
sectionBand(controls, 19, 10, "Exact reproduction check", C.paleGreen, C.green);
writeTable(controls, 20, payload.reproduction, ["n", "maximum_absolute_prediction_difference", "mean_absolute_prediction_difference"], { headerFill: C.green });
controls.getRange("B21:C21").format.numberFormat = "0.000000";
controls.freezePanes.freezeRows(2);

const audit = workbook.worksheets.add("Inclusion Audit");
titleBand(audit, 14, "Source-Row Inclusion Audit", "Every public model-view row remains visible with its primary, sensitivity, or exclusion status");
writeTable(audit, 4, payload.inclusion_audit, ["model_record_id", "compound_id", "preferred_name", "series_id", "endpoint", "relation", "value", "unit", "pIC50", "pIC50_relation", "model_role", "included_primary_interval", "included_conditional_sensitivity", "exclusion_reason"], {
  rowHeight: 44,
  widths: { model_record_id: 34, model_role: 42, exclusion_reason: 62 },
});
audit.getRange(`G5:I${4 + payload.inclusion_audit.length}`).format.numberFormat = "0.000";
audit.freezePanes.freezeRows(4);

const inspectRanges = [
  ["Summary", "A1:L20"],
  ["Locked Comparison", `A1:K${4 + comparisonRows.length}`],
  ["Dataset & Domain", `A1:I${21 + payload.public_evidence.length}`],
  ["Full Model Matrix", `A1:P${4 + payload.full_model_matrix.length}`],
  ["Controls", "A1:J21"],
  ["Inclusion Audit", `A1:N${4 + payload.inclusion_audit.length}`],
];
for (const [sheetName, range] of inspectRanges) {
  const result = await workbook.inspect({ kind: "table", range: `${sheetName}!${range}`, include: "values,formulas", tableMaxRows: 120, tableMaxCols: 18, maxChars: 18000 });
  console.log(`INSPECT ${sheetName}`);
  console.log(result.ndjson);
}
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 200 }, summary: "final formula error scan" });
console.log("FORMULA_ERROR_SCAN");
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
const renderRanges = {
  "Summary": "A1:L20",
  "Locked Comparison": "A1:K16",
  "Dataset & Domain": "A1:I35",
  "Full Model Matrix": "A1:P35",
  "Controls": "A1:J21",
  "Inclusion Audit": "A1:N29",
};
for (const [sheetName, range] of Object.entries(renderRanges)) {
  const rendered = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const safe = sheetName.toLowerCase().replaceAll(/[^a-z0-9]+/g, "_");
  await fs.writeFile(path.join(previewDir, `${safe}.png`), new Uint8Array(await rendered.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(`EXPORTED ${outputPath}`);
console.log(`PREVIEWS ${previewDir}`);
