import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/shivanshsahni/Downloads/Personal/Wang/Menin";
const PACKAGE = path.join(ROOT, "research/reports/platform/herg_meeting_2026_08_18");
const OUT = path.join(ROOT, "research/reports/platform/herg_meeting_2026_08_18_deck");
const FIG = path.join(PACKAGE, "figures");
const PPTX = path.join(OUT, "HERG_MEETING_DECK_2026_08_18_REVISED.pptx");

const C = {
  bg: "#FFFFFF",
  ink: "#0B0F14",
  muted: "#59616D",
  light: "#F2F4F7",
  line: "#D6DAE0",
  blue: "#1E63D5",
  blue2: "#EAF2FF",
  cyan: "#4CB6E8",
  green: "#16805A",
  green2: "#E9F7F1",
  orange: "#C96A18",
  orange2: "#FFF2E8",
  red: "#B84040",
  red2: "#FCECEC",
  purple: "#7057C7",
  purple2: "#F1EDFF",
};

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function addBox(slide, x, y, w, h, fill = C.light, radius = "rounded-xl", line = C.line) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function addLine(slide, x, y, w, h = 0, color = C.line, width = 1) {
  return slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function addText(slide, text, x, y, w, h, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: opts.fill ?? "none",
    line: { style: "solid", fill: opts.line ?? "none", width: opts.lineWidth ?? 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize: opts.fontSize ?? 22,
    typeface: opts.typeface ?? "Helvetica Neue",
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    insets: opts.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return box;
}

function title(slide, headline) {
  addText(slide, headline, 46, 34, 1188, 62, {
    fontSize: 39,
    bold: true,
    autoFit: "shrinkText",
  });
  addLine(slide, 46, 108, 1188, 0, C.line, 1);
}

function note(slide, text, sources = []) {
  const sourceBlock = sources.length
    ? `\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]`
    : "";
  slide.speakerNotes.textFrame.setText(`${text}${sourceBlock}`);
  slide.speakerNotes.setVisible(true);
}

async function imageBytes(file) {
  const b = await fs.readFile(file);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

async function addFigure(slide, filename, x, y, w, h, alt) {
  slide.images.add({
    blob: await imageBytes(path.join(FIG, filename)),
    contentType: "image/png",
    alt,
    fit: "contain",
    position: { left: x, top: y, width: w, height: h },
  });
}

function metric(slide, x, y, w, h, value, label, accent = C.blue, fill = C.light) {
  addBox(slide, x, y, w, h, fill, "rounded-xl", fill);
  addText(slide, value, x + 20, y + 18, w - 40, 60, {
    fontSize: 45,
    bold: true,
    color: accent,
    verticalAlignment: "middle",
  });
  addText(slide, label, x + 20, y + 82, w - 40, h - 96, {
    fontSize: 20,
    color: C.ink,
    bold: true,
  });
}

function timelineCard(slide, x, y, w, h, version, heading, body, color, fill) {
  addBox(slide, x, y, w, h, fill, "rounded-xl", fill);
  addText(slide, version, x + 18, y + 16, 54, 34, { fontSize: 24, bold: true, color });
  addText(slide, heading, x + 76, y + 16, w - 94, 34, { fontSize: 21, bold: true });
  addText(slide, body, x + 18, y + 60, w - 36, h - 74, { fontSize: 17, color: C.muted });
}

function tableCell(slide, text, x, y, w, h, opts = {}) {
  if (opts.fill) {
    slide.shapes.add({
      geometry: "rect",
      position: { left: x, top: y, width: w, height: h },
      fill: opts.fill,
      line: { style: "solid", fill: opts.line ?? C.line, width: 1 },
    });
  }
  return addText(slide, text, x + 10, y + 8, w - 20, h - 16, {
    fontSize: opts.fontSize ?? 17,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    verticalAlignment: opts.verticalAlignment ?? "middle",
    alignment: opts.alignment ?? "left",
  });
}

// Slide 1: Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addText(slide, "hERG Prediction Platform", 48, 84, 540, 132, {
    fontSize: 58,
    bold: true,
    autoFit: "none",
    verticalAlignment: "middle",
  });
  addText(slide, "Nine Iterations of Honest Model Development", 48, 236, 540, 54, {
    fontSize: 29,
    color: C.blue,
    bold: true,
  });
  addText(slide, "Continuous potency prediction, transparent failure domains, and a clear experimental path", 48, 318, 520, 108, {
    fontSize: 25,
    color: C.muted,
  });
  addText(slide, "18,801 Structures", 48, 488, 210, 34, { fontSize: 21, bold: true, color: C.blue });
  addText(slide, "8,455 Scaffolds", 270, 488, 210, 34, { fontSize: 21, bold: true, color: C.green });
  addText(slide, "5 Nested Folds", 48, 536, 210, 34, { fontSize: 21, bold: true, color: C.orange });
  addBox(slide, 620, 42, 614, 612, C.light, "rounded-2xl", C.line);
  await addFigure(slide, "06_graphical_summary.png", 640, 62, 574, 572, "Graphical summary of the hERG modeling platform");
  note(slide, "Open with the development arc: the work moved from data governance and 2D baselines through physics, measurement modeling, exhaustive feature combinations, and domain specialists. The core contribution is an honest broad prediction platform with clearly measured limits.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(FIG, "06_graphical_summary.png")]);
}

// Slide 2: Nine-version timeline
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Nine Iterations of Model Development");
  const cards = [
    ["V1", "Dataset Governance", "Structure identities, provenance, frozen splits, and leakage controls.", C.blue, C.blue2],
    ["V2", "Tuned 2D XGBoost", "First strong scaffold anchor at approximately 0.446 MAE.", C.blue, C.blue2],
    ["V3", "Broader Model Search", "More model families and features did not improve the honest anchor.", C.blue, C.blue2],
    ["V4", "24-Conformer Surface", "Fresh ligand 3D features generated across 24,901 quantitative structures.", C.purple, C.purple2],
    ["V5", "Feature Attribution", "Held-out ablations isolated the unique contribution of each block.", C.purple, C.purple2],
    ["V6", "Fundamental Optimization", "Physics-focused combinations tested redundancy and failure modes.", C.purple, C.purple2],
    ["V7", "Measurement Objectives", "Accuracy and safety-tail objectives separated instead of hidden.", C.green, C.green2],
    ["V8", "Feature Lattice", "2,048 coalitions mapped what works, what fails, and for whom.", C.green, C.green2],
    ["V9", "Domain Mixture", "Specialists, local analogs, and applicability context reached 0.4328 MAE.", C.green, C.green2],
  ];
  cards.forEach((c, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    timelineCard(slide, 46 + col * 400, 132 + row * 184, 372, 156, ...c);
  });
  note(slide, "Use this as the narrative spine for the meeting. V1 to V3 established the governed baseline. V4 to V6 tested ligand-only physics and feature causality. V7 to V9 focused on measurement heterogeneity, exhaustive combinations, domain specialists, and local evidence.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(ROOT, "research/local_runs/herg_feature_lattice_analysis_v81_superseded_cross_fold_mmp/ANALYSIS.md"), path.join(ROOT, "research/local_runs/herg_honest_measurement_campaign_v7_1/analysis.json")]);
}

// Slide 3: Headline result
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Best Honest Internal Performance");
  await addFigure(slide, "01_model_progress.png", 46, 132, 704, 516, "Model progression through the V9 nested scaffold stack");
  metric(slide, 784, 136, 214, 148, "0.4328", "Nested Stack MAE", C.blue, C.blue2);
  metric(slide, 1018, 136, 214, 148, "91.8%", "Within 1.0 Log", C.green, C.green2);
  metric(slide, 784, 306, 214, 148, "69.8%", "Within 0.5 Log", C.orange, C.orange2);
  metric(slide, 1018, 306, 214, 148, "0.4370", "Deployable XGBoost MAE", C.purple, C.purple2);
  addBox(slide, 784, 484, 448, 130, C.light, "rounded-xl", C.line);
  addText(slide, "0.0114 MAE Improvement", 806, 504, 404, 40, { fontSize: 29, bold: true, color: C.blue });
  addText(slide, "V9 versus V8; 95% scaffold CI 0.0086 to 0.0141. Every outer fold improved.", 806, 554, 404, 50, { fontSize: 19, color: C.ink });
  note(slide, "The nested stack is the strongest unbiased internal evidence. The frozen deployable molecular model is a single XGBoost artifact and is slightly weaker. Do not present the stack as an external or prospective result.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "RESULT_TRACEABILITY.csv"), path.join(FIG, "01_model_progress.png")]);
}

// Slide 4: Evaluation design
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Evaluation Designed to Resist Leakage");
  metric(slide, 46, 132, 268, 128, "18,801", "Exact Training Structures", C.blue, C.blue2);
  metric(slide, 332, 132, 268, 128, "8,455", "Scaffold Groups", C.green, C.green2);
  metric(slide, 618, 132, 268, 128, "5", "Nested Outer Folds", C.orange, C.orange2);
  metric(slide, 904, 132, 328, 128, "0", "Validation Or Test Labels Opened", C.purple, C.purple2);
  const steps = [
    ["1", "Structure Collapse", "Replicate observations become one structure-level target."],
    ["2", "Scaffold Separation", "Related chemotypes stay together in a single held-out fold."],
    ["3", "Inner Selection", "Features, parameters, and mixtures are chosen without outer labels."],
    ["4", "Single Outer Prediction", "Every structure is evaluated once in its untouched scaffold context."],
  ];
  addLine(slide, 92, 382, 1088, 0, C.line, 4);
  steps.forEach(([n, h, b], i) => {
    const x = 54 + i * 300;
    slide.shapes.add({ geometry: "ellipse", position: { left: x, top: 348, width: 68, height: 68 }, fill: i === 3 ? C.blue : C.ink, line: { style: "solid", fill: i === 3 ? C.blue : C.ink, width: 1 } });
    addText(slide, n, x, 365, 68, 32, { fontSize: 24, bold: true, color: C.bg, alignment: "center" });
    addText(slide, h, x, 444, 258, 34, { fontSize: 23, bold: true });
    addText(slide, b, x, 490, 258, 110, { fontSize: 18, color: C.muted });
  });
  note(slide, "This design deliberately produces harder and more transferable estimates than random splitting. It prevents replicated measurements and close scaffold relatives from appearing as independent evidence.", [path.join(PACKAGE, "TECHNICAL_QA.md"), path.join(PACKAGE, "MEETING_BRIEF.md")]);
}

// Slide 5: Feature impact table
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Feature Families Ranked by Unique Held-Out Impact");
  const x = [46, 256, 478, 664, 938];
  const w = [210, 222, 186, 274, 296];
  const headers = ["Feature Test", "MAE Change", "95% Scaffold CI", "Evidence", "What We Learn"];
  headers.forEach((h, i) => tableCell(slide, h, x[i], 134, w[i], 52, { fill: C.ink, color: C.bg, bold: true, fontSize: 17 }));
  const rows = [
    ["Remove RDKit2D", "+0.0250", "+0.0204 to +0.0295", "Reliable In 5 Of 5 Folds", "Global physicochemical structure is the strongest unique foundation.", C.blue2, C.blue],
    ["Remove Morgan", "+0.0065", "+0.0031 to +0.0101", "Reliable In 4 Of 5 Folds", "Local substructure context adds independent scaffold-transfer signal.", C.green2, C.green],
    ["Add Old Ligand 3D", "-0.0052", "-0.0078 to -0.0026", "Reliable Deterioration", "Generic or unstable 3D can actively dilute a strong 2D model.", C.red2, C.red],
    ["Add Polarity and Contacts", "-0.0007", "-0.0022 to +0.0008", "No Stable Gain", "These ligand-only physics proxies are mostly redundant after 2D chemistry.", C.light, C.muted],
    ["Add Energy and Flexibility", "-0.0004", "-0.0019 to +0.0012", "No Stable Gain", "Force-field summaries do not improve broad aggregate transfer.", C.light, C.muted],
    ["Add Selected Interactions", "+0.0003", "-0.0013 to +0.0020", "No Stable Gain", "Interaction hypotheses may matter locally, not as a global block.", C.light, C.muted],
  ];
  rows.forEach((r, row) => {
    const y = 186 + row * 76;
    const [a, b, c, d, e, fill, accent] = r;
    [a, b, c, d, e].forEach((v, i) => tableCell(slide, v, x[i], y, w[i], 76, { fill, color: i === 1 ? accent : C.ink, bold: i === 1 || i === 3, fontSize: i === 4 ? 16 : 17 }));
  });
  note(slide, "Positive MAE change means prediction worsened when an included block was removed or improved when an omitted block was added, depending on the named test. The decisive positive evidence is RDKit2D and Morgan. The generic 3D and physics blocks did not show broad independent value.", [path.join(PACKAGE, "tables/feature_family_effects.csv"), path.join(PACKAGE, "MEETING_BRIEF.md")]);
}

// Slide 6: Physics result
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "What Ligand-Only Physics Added");
  await addFigure(slide, "02_feature_family_effects.png", 46, 128, 736, 532, "Held-out feature-family effects including ligand-only 3D and physics blocks");
  addBox(slide, 812, 134, 420, 138, C.purple2, "rounded-xl", C.purple2);
  addText(slide, "24 Conformers", 838, 156, 368, 48, { fontSize: 36, bold: true, color: C.purple });
  addText(slide, "Fresh conformer surface across 24,901 quantitative structures", 838, 212, 368, 44, { fontSize: 19, bold: true });
  addBox(slide, 812, 294, 420, 138, C.red2, "rounded-xl", C.red2);
  addText(slide, "No Aggregate Gain", 838, 316, 368, 48, { fontSize: 34, bold: true, color: C.red });
  addText(slide, "Shape, WHIM, energy, flexibility, and charge-contact blocks were unstable or redundant", 838, 372, 368, 48, { fontSize: 18, bold: true });
  addBox(slide, 812, 454, 420, 170, C.green2, "rounded-xl", C.green2);
  addText(slide, "Scientific Interpretation", 838, 478, 368, 34, { fontSize: 24, bold: true, color: C.green });
  addText(slide, "The negative result narrows the next experiment: improve microstate, receptor-state, membrane, and assay context instead of adding more generic ligand descriptors.", 838, 526, 368, 82, { fontSize: 19, color: C.ink });
  note(slide, "This is not evidence that physics is unimportant. It is evidence that generic ligand-only conformer summaries do not add stable global information after strong 2D representations. Receptor-aware and microstate-aware physics remain untested because preparation and environment contracts were not yet satisfied.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "tables/feature_family_effects.csv"), path.join(ROOT, "research/local_runs/herg_quantitative_24conformer_v4/manifest.json"), path.join(FIG, "02_feature_family_effects.png")]);
}

// Slide 7: Applicability domain
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Applicability Domain Defines Reliability");
  await addFigure(slide, "03_risk_coverage.png", 46, 128, 748, 520, "Applicability-domain error and interval coverage");
  metric(slide, 824, 134, 192, 146, "0.379", "In-Domain MAE", C.green, C.green2);
  metric(slide, 1036, 134, 196, 146, "0.560", "Flagged MAE", C.red, C.red2);
  metric(slide, 824, 304, 408, 148, "91.3%", "Coverage of the Nominal 90% Interval", C.blue, C.blue2);
  addBox(slide, 824, 478, 408, 144, C.light, "rounded-xl", C.line);
  addText(slide, "Operational Rule", 848, 500, 360, 34, { fontSize: 25, bold: true, color: C.blue });
  addText(slide, "Report the prediction together with the label-blind domain flag. Interval width alone was not a useful error ranking.", 848, 548, 360, 62, { fontSize: 19, color: C.ink });
  note(slide, "The domain flag is label blind and operationally meaningful. It identifies a 0.181 MAE gap between supported and extrapolative predictions. Global interval coverage is calibrated, but interval width alone does not rank errors well.", [path.join(PACKAGE, "tables/domain_flag_performance.csv"), path.join(PACKAGE, "tables/risk_coverage.csv"), path.join(FIG, "03_risk_coverage.png")]);
}

// Slide 8: Failure domains and heavy compounds
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Remaining Errors Are Structured");
  await addFigure(slide, "04_failure_modes.png", 46, 128, 794, 530, "Error patterns across disagreement, potency, similarity, flexibility, and molecular mass");
  const items = [
    ["1.046", "MAE With Replicate Spread Above 1 Log", C.red, C.red2],
    ["0.518", "MAE At 700 Da Or Higher", C.orange, C.orange2],
    ["0.560", "MAE In Flagged Extrapolation", C.blue, C.blue2],
  ];
  items.forEach(([v, label, color, fill], i) => {
    const y = 136 + i * 142;
    addBox(slide, 868, y, 364, 120, fill, "rounded-xl", fill);
    addText(slide, v, 890, y + 18, 114, 56, { fontSize: 42, bold: true, color, verticalAlignment: "middle" });
    addText(slide, label, 1014, y + 24, 194, 72, { fontSize: 19, bold: true, verticalAlignment: "middle" });
  });
  addBox(slide, 868, 562, 364, 86, C.light, "rounded-xl", C.line);
  addText(slide, "500 To 700 Da Remains Competitive", 890, 584, 320, 44, { fontSize: 22, bold: true, color: C.green, alignment: "center" });
  note(slide, "The largest errors are not random. Label disagreement, potency extremes, lower similarity, flexibility, and the very heavy tail dominate. Performance through 500 to 700 Da remains credible; the 700 Da and higher subgroup is small and should be reported with uncertainty.", [path.join(PACKAGE, "tables/label_disagreement_summary.csv"), path.join(PACKAGE, "tables/subgroup_sensitivity.csv"), path.join(ROOT, "research/local_runs/herg_heavy_compound_sidecar_v9/HEAVY_COMPOUND_REPORT.md"), path.join(FIG, "04_failure_modes.png")]);
}

// Slide 9: Local analog evidence
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Local Analogs Improve Covered Chemistry");
  await addFigure(slide, "05_mmp_analog_assistance.png", 46, 128, 744, 528, "Performance of broad and analog-assisted predictions on MMP-supported structures");
  metric(slide, 820, 134, 190, 146, "41.7%", "Structures With Analog Support", C.blue, C.blue2);
  metric(slide, 1030, 134, 202, 146, "7,847", "Supported Structures", C.green, C.green2);
  addBox(slide, 820, 304, 412, 146, C.green2, "rounded-xl", C.green2);
  addText(slide, "0.400 To 0.381 MAE", 844, 328, 364, 44, { fontSize: 31, bold: true, color: C.green });
  addText(slide, "Broad anchor to analog-assisted prediction on supported structures", 844, 384, 364, 46, { fontSize: 19, bold: true });
  addBox(slide, 820, 476, 412, 148, C.orange2, "rounded-xl", C.orange2);
  addText(slide, "Activity-Cliff Direction Remains Weak", 844, 500, 364, 34, { fontSize: 23, bold: true, color: C.orange });
  addText(slide, "Use local analogs as a supported specialist, not as causal transformation rules.", 844, 548, 364, 56, { fontSize: 19, color: C.ink });
  note(slide, "Analog support produces a useful local prediction improvement, but only on 41.7 percent of the structures. The remaining transformation-direction results are not strong enough for causal medicinal chemistry rules.", [path.join(PACKAGE, "tables/mmp_covered_structure_performance.csv"), path.join(FIG, "05_mmp_analog_assistance.png")]);
}

// Slide 10: Competitor comparison
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Published Competitors Use Different Evidence Surfaces");
  const x = [46, 188, 346, 556, 778, 990];
  const w = [142, 158, 210, 222, 212, 242];
  const headers = ["Tool and Task", "Labeled Data", "Publicized Result", "Benchmark Tailoring", "Features", "Real-World Limitation"];
  headers.forEach((h, i) => tableCell(slide, h, x[i], 128, w[i], 58, { fill: C.ink, color: C.bg, bold: true, fontSize: 16 }));
  const rows = [
    ["Our Platform\nContinuous pIC50", "18,801\nstructures", "0.4328 MAE\nNested scaffold", "Low\nStructure collapse; sealed outcomes", "RDKit2D, Morgan, specialists, domain flag", "Internal only; WT-or-unspecified; external series still needed", C.blue2],
    ["Pred-hERG 5.0\nRegression", "7,609\nrecords", "0.35 MAE\n0.44 RMSE", "High\nTask curation; random 80/20 split", "ECFP, FCFP, SVM, consensus", "Random split; pooled assay types; current repository lacks full assets", C.light],
    ["hERGBoost\nRegression", "10,798\nmolecules", "0.438 MAE\nExternal n=706", "Moderate\nExternal set; grouping policy unclear", "Descriptors, fingerprints, XGBoost", "External R2 0.394; extreme-potency RMSE 1.154", C.light],
    ["HERGAI\nBinary at 20 µM", "299,927\n0.65% active", "86.4% recall\nAUC 0.863", "High\nExtreme imbalance; pose selection", "Docking PLEC, RF, XGB, DNN stack", "Binary screening task; asymmetric active and negative evidence", C.light],
    ["XGB-ISE\nBinary at 10 µM", "290,731\nmolecules", "MCC 0.72\nSelected strata", "Very High\nBest result covers 64% of external set", "Selected descriptors, balanced XGB ensemble", "Coverage selection and pooled HTS labels limit broad deployment", C.light],
  ];
  rows.forEach((r, row) => {
    const y = 186 + row * 92;
    r.slice(0, 6).forEach((v, i) => tableCell(slide, v, x[i], y, w[i], 92, { fill: r[6], bold: i === 0 || (row === 0 && i === 2), color: row === 0 && i === 2 ? C.blue : C.ink, fontSize: 16 }));
  });
  note(slide, "Benchmark tailoring is a neutral description of how much the reported result depends on curation, split choice, thresholding, selective coverage, or class construction. It is not an accusation of misconduct. Use the exact published task beside every number.", [path.join(PACKAGE, "LITERATURE_COMPARABILITY.md"), path.join(ROOT, "research/reports/platform/herg_paper/model_landscape_v2/model_comparison_matrix.csv"), "https://pmc.ncbi.nlm.nih.gov/articles/PMC11187631/", "https://doi.org/10.1016/j.compbiomed.2024.109416", "https://doi.org/10.1186/s13321-025-01063-8", "https://doi.org/10.1038/s41598-025-99766-3"]);
}

// Slide 11: Accuracy versus safety
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Accuracy and Safety Require Separate Operating Modes");
  addBox(slide, 46, 134, 566, 476, C.blue2, "rounded-2xl", C.blue2);
  addText(slide, "Global Accuracy Model", 78, 168, 502, 42, { fontSize: 30, bold: true, color: C.blue });
  addText(slide, "0.433", 78, 252, 220, 80, { fontSize: 62, bold: true, color: C.blue });
  addText(slide, "Global MAE", 78, 342, 220, 34, { fontSize: 22, bold: true });
  addText(slide, "1.425", 340, 252, 220, 80, { fontSize: 62, bold: true, color: C.red });
  addText(slide, "Tail MAE", 340, 342, 220, 34, { fontSize: 22, bold: true });
  addText(slide, "Best default for broad continuous potency prediction", 78, 454, 482, 66, { fontSize: 23, bold: true, color: C.ink });
  addBox(slide, 638, 134, 594, 476, C.orange2, "rounded-2xl", C.orange2);
  addText(slide, "Safety and Tail Model", 670, 168, 530, 42, { fontSize: 30, bold: true, color: C.orange });
  addText(slide, "+0.011", 670, 252, 230, 80, { fontSize: 62, bold: true, color: C.red });
  addText(slide, "Global MAE Cost", 670, 342, 230, 34, { fontSize: 22, bold: true });
  addText(slide, "1.354", 940, 252, 230, 80, { fontSize: 62, bold: true, color: C.green });
  addText(slide, "Tail MAE", 940, 342, 230, 34, { fontSize: 22, bold: true });
  addText(slide, "Better balance at potency extremes; worse average error", 670, 454, 500, 66, { fontSize: 23, bold: true, color: C.ink });
  note(slide, "The safety-selected objective improves the potency tails but pays a statistically supported global cost. The defensible product design is two transparent modes rather than one hidden objective.", [path.join(PACKAGE, "tables/safety_objective_tradeoff.csv"), path.join(PACKAGE, "MEETING_BRIEF.md")]);
}

// Slide 12: Next experiments
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Recommended Next Experiments");
  const rows = [
    ["1", "External Functional Series", "Freeze an explicit human-WT patch-clamp series and evaluate once.", "Tests real-world transfer and ends the internal-metric debate.", C.blue, C.blue2],
    ["2", "Measurement Model", "Adjudicate construct, modality, source, cell system, and protocol context.", "Separates chemistry error from assay and label error.", C.green, C.green2],
    ["3", "Activity-Cliff Specialist", "Acquire matched pairs in the highest-error and highest-value chemical series.", "Targets the 41.7% analog-supported domain where local models already help.", C.orange, C.orange2],
    ["4", "Receptor and Microstate Physics", "Prepare receptor states, ligand microstates, membrane context, and pinned software.", "Tests physics that the ligand-only conformer campaign could not represent.", C.red, C.red2],
  ];
  rows.forEach(([n, h, action, value, color, fill], i) => {
    const y = 132 + i * 132;
    addBox(slide, 46, y, 1186, 110, fill, "rounded-xl", fill);
    addText(slide, n, 68, y + 24, 56, 58, { fontSize: 40, bold: true, color, alignment: "center" });
    addText(slide, h, 146, y + 18, 310, 34, { fontSize: 24, bold: true, color });
    addText(slide, action, 146, y + 58, 430, 40, { fontSize: 18, color: C.ink });
    addText(slide, value, 620, y + 24, 574, 66, { fontSize: 20, bold: true, color: C.ink, verticalAlignment: "middle" });
  });
  note(slide, "Close with an explicit decision request. The first two experiments address the largest evidence gaps. Activity-cliff modeling builds on the successful local-analog signal. Receptor physics should start only after receptor, microstate, membrane, and environment preparation is complete.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "TECHNICAL_QA.md")]);
}

await fs.mkdir(OUT, { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(PPTX);
console.log(PPTX);
