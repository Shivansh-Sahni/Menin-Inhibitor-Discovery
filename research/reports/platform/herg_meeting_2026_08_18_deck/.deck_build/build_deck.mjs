import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/shivanshsahni/Downloads/Personal/Wang/Menin";
const PACKAGE = path.join(ROOT, "research/reports/platform/herg_meeting_2026_08_18");
const OUT = path.join(ROOT, "research/reports/platform/herg_meeting_2026_08_18_deck");
const FIG = path.join(PACKAGE, "figures");
const PPTX = path.join(OUT, "HERG_MEETING_DECK_2026_08_18.pptx");

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

function title(slide, headline, kicker = null) {
  if (kicker) addText(slide, kicker.toUpperCase(), 42, 28, 400, 24, { fontSize: 13, bold: true, color: C.blue });
  addText(slide, headline, 42, kicker ? 54 : 36, 1168, 72, { fontSize: 39, bold: true, autoFit: "shrinkText" });
  addLine(slide, 42, 128, 1196, 0, C.line, 1);
}

function footer(slide, n, source = "Internal analysis | train-only nested scaffold evaluation") {
  addText(slide, source, 42, 678, 1000, 18, { fontSize: 10.5, color: C.muted });
  addText(slide, String(n).padStart(2, "0"), 1182, 674, 56, 20, { fontSize: 11, color: C.muted, alignment: "right" });
}

function note(slide, text, sources = []) {
  const sourceBlock = sources.length ? `\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]` : "";
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

function metric(slide, x, y, w, h, value, label, accent = C.blue) {
  addBox(slide, x, y, w, h, C.light, "rounded-xl", C.line);
  addText(slide, value, x + 22, y + 24, w - 44, 64, { fontSize: 46, bold: true, color: accent, verticalAlignment: "bottom" });
  addText(slide, label, x + 22, y + 94, w - 44, h - 110, { fontSize: 17, color: C.muted });
}

function pill(slide, text, x, y, w, fill, color) {
  addBox(slide, x, y, w, 32, fill, "rounded-full", fill);
  addText(slide, text, x + 10, y + 7, w - 20, 18, { fontSize: 12, bold: true, color, alignment: "center" });
}

// 1 — Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addText(slide, "PLATFORM PROGRESS REVIEW", 42, 38, 420, 24, { fontSize: 13, bold: true, color: C.blue });
  addText(slide, "hERG Prediction\nPlatform", 42, 124, 520, 170, { fontSize: 58, bold: true, autoFit: "none" });
  addText(slide, "Honest gains, clear boundaries,\nand the next experiments", 42, 322, 520, 82, { fontSize: 27, color: C.muted });
  pill(slide, "18,801 structures", 42, 454, 164, C.blue2, C.blue);
  pill(slide, "8,455 scaffolds", 220, 454, 154, C.green2, C.green);
  pill(slide, "5 nested folds", 388, 454, 142, C.orange2, C.orange);
  addText(slide, "Shivansh Sahni  •  August 18, 2026", 42, 626, 520, 28, { fontSize: 17, color: C.muted });
  addBox(slide, 618, 42, 620, 610, C.light, "rounded-2xl", C.line);
  await addFigure(slide, "06_graphical_summary.png", 638, 60, 580, 574, "Graphical summary of the hERG modeling platform and major results");
  footer(slide, 1, "hERG platform meeting | internal results");
  note(slide, "Open with the contribution, not just the metric: we now have a broad, traceable hERG potency platform evaluated under a deliberately difficult scaffold-held-out design. The meeting goal is to agree on the next evidence-generating experiments.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(FIG, "06_graphical_summary.png")]);
}

// 2 — Headline result
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "The strongest honest internal result so far", "Headline result");
  await addFigure(slide, "01_model_progress.png", 42, 146, 708, 492, "Model progress from earlier baselines through the V9 nested scaffold stack");
  metric(slide, 782, 158, 214, 148, "0.4328", "MAE in pIC50\nV9 nested stack", C.blue);
  metric(slide, 1016, 158, 214, 148, "91.8%", "predictions within\n1.0 log unit", C.green);
  addBox(slide, 782, 332, 448, 138, C.blue2, "rounded-xl", C.blue2);
  addText(slide, "−0.0114 MAE vs V8", 806, 354, 400, 46, { fontSize: 31, bold: true, color: C.blue });
  addText(slide, "95% scaffold-bootstrap CI: −0.0141 to −0.0086\nAll five outer folds improved.", 806, 406, 400, 50, { fontSize: 16.5, color: C.muted });
  addText(slide, "Important: the stack is the best cross-fitted internal evidence. The frozen deployable XGBoost artifact is slightly weaker (MAE 0.4370).", 782, 500, 448, 92, { fontSize: 18, color: C.ink });
  footer(slide, 2);
  note(slide, "The improvement is small in absolute pIC50 units, but it is repeatable across every outer fold and the scaffold-cluster confidence interval excludes zero. Do not call this external superiority. Also distinguish the cross-fitted stack from the single deployable XGBoost model.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "RESULT_TRACEABILITY.csv"), path.join(FIG, "01_model_progress.png")]);
}

// 3 — Evaluation design
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "The evaluation is designed to resist inflated metrics", "Why the result is credible");
  const steps = [
    ["1", "Collapse", "Structure-level targets prevent repeated measurements from masquerading as independent compounds."],
    ["2", "Separate", "8,455 scaffold groups keep related chemotypes together during splitting."],
    ["3", "Select", "Hyperparameters and mixtures are chosen only inside each outer-training context."],
    ["4", "Evaluate", "Each compound is predicted once in a held-out scaffold fold; repository validation/test outcomes stay sealed."],
  ];
  addLine(slide, 94, 340, 1060, 0, C.line, 3);
  steps.forEach(([n, h, b], i) => {
    const x = 62 + i * 297;
    const circle = slide.shapes.add({ geometry: "ellipse", position: { left: x + 12, top: 306, width: 66, height: 66 }, fill: i === 3 ? C.blue : C.ink, line: { style: "solid", fill: i === 3 ? C.blue : C.ink, width: 1 } });
    addText(slide, n, x + 12, 321, 66, 34, { fontSize: 25, bold: true, color: C.bg, alignment: "center" });
    addText(slide, h, x, 406, 240, 34, { fontSize: 24, bold: true });
    addText(slide, b, x, 452, 244, 128, { fontSize: 16.5, color: C.muted });
  });
  metric(slide, 42, 154, 264, 118, "18,801", "exact training structures", C.blue);
  metric(slide, 324, 154, 264, 118, "8,455", "scaffold groups", C.green);
  metric(slide, 606, 154, 264, 118, "5", "nested outer folds", C.orange);
  metric(slide, 888, 154, 342, 118, "0", "repository validation/test\nlabels opened", C.purple);
  footer(slide, 3);
  note(slide, "This slide explains why the MAE is not directly comparable to random-split or duplicate-rich benchmarks. The key governance statement is that all adaptive choices were confined to training folds and the repository validation/test outcomes remained sealed.", [path.join(PACKAGE, "TECHNICAL_QA.md"), path.join(PACKAGE, "MEETING_BRIEF.md")]);
}

// 4 — Feature evidence
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "2D chemistry carries the stable unique signal", "Feature-family evidence");
  await addFigure(slide, "02_feature_family_effects.png", 42, 146, 780, 504, "Held-out feature-family ablation effects on hERG potency prediction");
  addBox(slide, 850, 150, 380, 138, C.blue2, "rounded-xl", C.blue2);
  addText(slide, "+0.0250 MAE", 874, 174, 330, 48, { fontSize: 33, bold: true, color: C.blue });
  addText(slide, "when RDKit2D is removed", 874, 230, 330, 30, { fontSize: 18, color: C.muted });
  addBox(slide, 850, 308, 380, 138, C.green2, "rounded-xl", C.green2);
  addText(slide, "+0.0065 MAE", 874, 332, 330, 48, { fontSize: 33, bold: true, color: C.green });
  addText(slide, "when Morgan fingerprints are removed", 874, 388, 330, 34, { fontSize: 18, color: C.muted });
  addText(slide, "Generic ligand-only 3D, shape, WHIM, energy/flexibility, and polarity/charge blocks did not show a stable independent aggregate gain.", 850, 480, 380, 116, { fontSize: 19, color: C.ink });
  footer(slide, 4);
  note(slide, "The scientifically useful result is both positive and negative. RDKit2D and Morgan carry reproducible nonredundant signal. The expensive generic conformer features are mostly redundant or noisy after those representations are present. This does not rule out receptor-aware, membrane-aware, or higher-quality microstate physics; those experiments were not performed.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(FIG, "02_feature_family_effects.png")]);
}

// 5 — Confidence and domain
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Confidence is useful when it is tied to domain", "Operational reliability");
  await addFigure(slide, "03_risk_coverage.png", 42, 146, 746, 500, "Performance in the label-blind applicability domain and cross-fitted uncertainty coverage");
  metric(slide, 820, 154, 194, 134, "0.379", "MAE\nin-domain", C.green);
  metric(slide, 1030, 154, 200, 134, "0.560", "MAE\nflagged", C.red);
  addBox(slide, 820, 314, 410, 138, C.blue2, "rounded-xl", C.blue2);
  addText(slide, "91.3% coverage", 844, 338, 362, 44, { fontSize: 32, bold: true, color: C.blue });
  addText(slide, "for the cross-fitted nominal 90% interval", 844, 392, 362, 36, { fontSize: 17, color: C.muted });
  addText(slide, "Negative control: ranking by interval width alone was nearly flat. The label-blind domain flag is the defensible abstention signal.", 820, 486, 410, 102, { fontSize: 18, color: C.ink });
  footer(slide, 5);
  note(slide, "Emphasize the distinction between calibrated interval coverage and useful ranking. The interval is calibrated globally, but width alone did not sharply rank errors. The label-blind domain flag—based on novelty and support—is what separates a much harder prediction population.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(FIG, "03_risk_coverage.png")]);
}

// 6 — Failure modes
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "The remaining error ceiling is structured", "Where the model still fails");
  await addFigure(slide, "04_failure_modes.png", 42, 146, 802, 510, "Error patterns by label disagreement, similarity, potency, and molecular mass");
  const items = [
    ["Label conflict", "Replicate spread >1 log: MAE 1.046", C.red2, C.red],
    ["Potency tails", "Strong regression toward the mean", C.orange2, C.orange],
    ["Low similarity", "Error rises outside supported chemistry", C.blue2, C.blue],
    ["Very heavy", ">=700 Da: MAE 0.518 (n=218)", C.light, C.ink],
  ];
  items.forEach(([h, b, fill, color], i) => {
    const y = 154 + i * 118;
    addBox(slide, 872, y, 358, 96, fill, "rounded-xl", fill);
    addText(slide, h, 894, y + 16, 310, 28, { fontSize: 20, bold: true, color });
    addText(slide, b, 894, y + 52, 310, 26, { fontSize: 15.5, color: C.muted });
  });
  footer(slide, 6);
  note(slide, "The failure analysis changes the recommended research plan. More generic feature breadth is unlikely to fix measurement disagreement, protocol heterogeneity, or activity cliffs. The >=700 Da result is a warning, not a universal failure claim, because the subgroup is small. The 500–700 Da region remains competitive.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "TECHNICAL_QA.md"), path.join(FIG, "04_failure_modes.png")]);
}

// 7 — MMP local analog
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Local analog evidence helps—but only where it exists", "Matched-pair result");
  await addFigure(slide, "05_mmp_analog_assistance.png", 42, 146, 742, 500, "Performance of broad, local analog-assisted, and stack models on MMP-supported structures");
  metric(slide, 820, 154, 190, 130, "41.7%", "of structures have\nMMP analog support", C.blue);
  metric(slide, 1030, 154, 200, 130, "7,847", "supported\nstructures", C.green);
  addBox(slide, 820, 310, 410, 124, C.green2, "rounded-xl", C.green2);
  addText(slide, "0.400 → 0.381 MAE", 844, 332, 362, 42, { fontSize: 30, bold: true, color: C.green });
  addText(slide, "broad anchor to analog-assisted prediction", 844, 382, 362, 30, { fontSize: 16, color: C.muted });
  addText(slide, "But cliff-delta direction remains weak. This supports local-context modeling and targeted matched pairs—not causal transformation rules.", 820, 470, 410, 112, { fontSize: 18.5, color: C.ink });
  footer(slide, 7);
  note(slide, "This is a useful practical insight: local analog context produces meaningful level-prediction improvement on its supported subset. It does not generalize to unsupported structures and it does not yet provide reliable mechanistic direction across activity cliffs.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(FIG, "05_mmp_analog_assistance.png")]);
}

// 8 — Safety tradeoff
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "There is no single metric-free definition of “best”", "Accuracy versus safety objective");
  addBox(slide, 42, 154, 556, 432, C.light, "rounded-2xl", C.line);
  addText(slide, "Global accuracy model", 72, 184, 496, 40, { fontSize: 27, bold: true });
  addText(slide, "Best for broad continuous potency prediction", 72, 230, 496, 28, { fontSize: 17, color: C.muted });
  addText(slide, "GLOBAL MAE", 72, 314, 200, 22, { fontSize: 12, bold: true, color: C.muted });
  addText(slide, "0.433", 72, 344, 220, 70, { fontSize: 53, bold: true, color: C.blue });
  addText(slide, "TAIL MAE", 320, 314, 200, 22, { fontSize: 12, bold: true, color: C.muted });
  addText(slide, "1.425", 320, 344, 220, 70, { fontSize: 53, bold: true, color: C.red });
  addText(slide, "Use as the default reported potency model.", 72, 482, 468, 48, { fontSize: 18 });
  addBox(slide, 626, 154, 604, 432, C.orange2, "rounded-2xl", C.orange2);
  addText(slide, "Safety / tail-selected model", 656, 184, 544, 40, { fontSize: 27, bold: true, color: C.orange });
  addText(slide, "Better balance at potency extremes, worse globally", 656, 230, 544, 28, { fontSize: 17, color: C.muted });
  addText(slide, "GLOBAL CHANGE", 656, 314, 230, 22, { fontSize: 12, bold: true, color: C.muted });
  addText(slide, "+0.011", 656, 344, 230, 70, { fontSize: 53, bold: true, color: C.red });
  addText(slide, "TAIL MAE", 926, 314, 230, 22, { fontSize: 12, bold: true, color: C.muted });
  addText(slide, "1.354", 926, 344, 230, 70, { fontSize: 53, bold: true, color: C.green });
  addText(slide, "Report as a separate operating mode, not a replacement.", 656, 482, 520, 48, { fontSize: 18 });
  addBox(slide, 258, 610, 764, 42, C.blue2, "rounded-full", C.blue2);
  addText(slide, "Recommendation: report both objectives transparently and choose by use case.", 278, 621, 724, 20, { fontSize: 16, bold: true, color: C.blue, alignment: "center" });
  footer(slide, 8);
  note(slide, "The safety-oriented objective is a successful sensitivity experiment, not the universally superior model. It materially reduces tail error and potency-bin imbalance, but the cost to average performance is statistically supported. Presenting both modes is more honest than hiding the tradeoff.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "TECHNICAL_QA.md")]);
}

// 9 — Literature positioning
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Why published headline metrics can look much better", "Fair comparison rule");
  addBox(slide, 42, 160, 548, 426, C.light, "rounded-2xl", C.line);
  addText(slide, "Common published headline", 72, 190, 488, 38, { fontSize: 27, bold: true });
  addText(slide, "• Binary blocker classification\n• Selected potency threshold\n• Random or internal split\n• Dataset-specific deduplication\n• Recall, AUROC, accuracy, or MCC", 72, 260, 470, 222, { fontSize: 22, color: C.muted });
  addText(slide, "These may be valid results—but they answer a different question.", 72, 510, 470, 54, { fontSize: 17.5, bold: true });
  addBox(slide, 622, 160, 608, 426, C.blue2, "rounded-2xl", C.blue2);
  addText(slide, "Our primary evidence", 652, 190, 548, 38, { fontSize: 27, bold: true, color: C.blue });
  addText(slide, "• Continuous pIC50 regression\n• Structure-collapsed targets\n• Scaffold-held-out nested selection\n• Assay, source, mass, cliff, and disagreement boundaries\n• Calibration and applicability domain", 652, 260, 540, 230, { fontSize: 22, color: C.ink });
  addText(slide, "Contribution: broader and more honest measurement—not a copied superiority claim.", 652, 510, 540, 54, { fontSize: 17.5, bold: true, color: C.blue });
  footer(slide, 9, "Published metrics are not head-to-head without a common frozen benchmark");
  note(slide, "If asked why another paper reports a much larger percentage, first identify its endpoint, threshold, split, and dataset. Pred-hERG 5.0 is the closest public regression comparator, but its curation and split differ. HERGAI and HergSPred are primarily classification tools. A fair superiority claim requires replaying all methods on the same frozen structures and endpoints.", [path.join(PACKAGE, "LITERATURE_COMPARABILITY.md"), "https://pmc.ncbi.nlm.nih.gov/articles/PMC11187631/", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12291323/", "https://pubs.acs.org/doi/10.1021/acs.jcim.2c00256", "https://doi.org/10.1016/j.compbiomed.2024.109416", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12756696/", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11245006/"]);
}

// 10 — Decisions and next steps
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "The next breakthrough requires better evidence, not just more descriptors", "Decision slide");
  const cards = [
    ["1", "Freeze an external series", "Explicit human-WT functional patch clamp, one-time evaluation, protocol-resolved.", C.blue2, C.blue],
    ["2", "Model the measurement", "Assay-conditioned or hierarchical model with construct, modality, source, and protocol context.", C.green2, C.green],
    ["3", "Target activity cliffs", "Acquire and model matched pairs where the broad model fails; train local-delta specialists.", C.orange2, C.orange],
    ["4", "Gate receptor physics", "Prepare receptor states, ligand microstates, membrane environment, and software pins before docking/MD claims.", C.red2, C.red],
  ];
  cards.forEach(([n, h, b, fill, color], i) => {
    const x = i % 2 === 0 ? 42 : 646;
    const y = i < 2 ? 158 : 398;
    addBox(slide, x, y, 584, 202, fill, "rounded-2xl", fill);
    addText(slide, n, x + 24, y + 22, 56, 56, { fontSize: 35, bold: true, color });
    addText(slide, h, x + 94, y + 24, 454, 40, { fontSize: 25, bold: true, color });
    addText(slide, b, x + 94, y + 82, 454, 88, { fontSize: 18, color: C.ink });
  });
  addText(slide, "Meeting ask: prioritize the external series and assay-context adjudication; use those results to decide whether receptor physics is justified.", 110, 626, 1060, 34, { fontSize: 18, bold: true, color: C.blue, alignment: "center" });
  footer(slide, 10);
  note(slide, "This is the decision point. My recommended priority is the explicit human-WT functional external series plus assay-context adjudication. Those measurements address the strongest current uncertainty. Receptor-aware physics becomes valuable only after the receptor, ligand-state, membrane, and environment contracts are actually prepared.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "TECHNICAL_QA.md")]);
}

// 11 — Appendix thresholds
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "Threshold views support tool comparisons without changing the primary task", "Appendix A");
  addText(slide, "Threshold", 66, 186, 190, 34, { fontSize: 16, bold: true, color: C.muted });
  addText(slide, "Sensitivity", 312, 186, 190, 34, { fontSize: 16, bold: true, color: C.muted });
  addText(slide, "Specificity", 554, 186, 190, 34, { fontSize: 16, bold: true, color: C.muted });
  addText(slide, "Balanced accuracy", 796, 186, 260, 34, { fontSize: 16, bold: true, color: C.muted });
  const rows = [
    ["20 μM", "94.8%", "43.0%", "68.9%", C.blue2, C.blue],
    ["10 μM", "80.2%", "68.6%", "74.4%", C.green2, C.green],
    ["1 μM", "43.1%", "96.8%", "70.0%", C.orange2, C.orange],
  ];
  rows.forEach(([t, se, sp, ba, fill, color], i) => {
    const y = 236 + i * 118;
    addBox(slide, 42, y, 1188, 92, fill, "rounded-xl", fill);
    addText(slide, t, 66, y + 25, 190, 42, { fontSize: 26, bold: true, color });
    addText(slide, se, 312, y + 25, 190, 42, { fontSize: 26, bold: true });
    addText(slide, sp, 554, y + 25, 190, 42, { fontSize: 26, bold: true });
    addText(slide, ba, 796, y + 25, 260, 42, { fontSize: 26, bold: true });
  });
  addText(slide, "These operating points are derived from the continuous model. They are secondary views—not a redefinition of the primary continuous-potency task.", 112, 618, 1056, 42, { fontSize: 17.5, color: C.muted, alignment: "center" });
  footer(slide, 11);
  note(slide, "Use this appendix only when asked for classifier-style numbers. Sensitivity and specificity change substantially with the threshold. Never quote one percentage without naming the threshold and task.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "RESULT_TRACEABILITY.csv")]);
}

// 12 — Appendix claims and close
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  title(slide, "What the evidence supports—and what it does not", "Appendix B");
  addBox(slide, 42, 158, 572, 390, C.green2, "rounded-2xl", C.green2);
  addText(slide, "SUPPORTED", 72, 184, 500, 30, { fontSize: 14, bold: true, color: C.green });
  addText(slide, "• Robust internal nested-scaffold improvement\n• Stable unique value from RDKit2D and Morgan\n• Calibrated cross-fitted intervals\n• Label-blind applicability-domain stratification\n• Local analog assistance on supported chemistry\n• Explicit accuracy–safety tradeoff", 72, 236, 500, 264, { fontSize: 20, color: C.ink });
  addBox(slide, 646, 158, 584, 390, C.red2, "rounded-2xl", C.red2);
  addText(slide, "NOT YET SUPPORTED", 676, 184, 520, 30, { fontSize: 14, bold: true, color: C.red });
  addText(slide, "• External or prospective superiority\n• Fully adjudicated explicit human-WT scope\n• Clinical QT-risk prediction\n• Causal feature or MMP transformation rules\n• Receptor-binding mechanism from ligand-only 3D\n• Universal performance above 700 Da", 676, 236, 520, 264, { fontSize: 20, color: C.ink });
  addText(slide, "The platform is ready for a decisive external measurement campaign.", 112, 590, 1056, 44, { fontSize: 28, bold: true, color: C.blue, alignment: "center" });
  footer(slide, 12, "Use RESULT_TRACEABILITY.csv and TECHNICAL_QA.md for exact questions");
  note(slide, "Close by being precise. The internal result, feature conclusions, uncertainty calibration, and domain boundaries are defensible. External superiority, causal biology, explicit-WT scope, clinical QT risk, and universal heavy-molecule coverage are not yet established. The next meeting outcome should be an agreed external evidence plan.", [path.join(PACKAGE, "MEETING_BRIEF.md"), path.join(PACKAGE, "TECHNICAL_QA.md"), path.join(PACKAGE, "RESULT_TRACEABILITY.csv")]);
}

await fs.mkdir(OUT, { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(PPTX);
console.log(PPTX);
