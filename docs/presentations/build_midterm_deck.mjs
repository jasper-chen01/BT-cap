import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ARTIFACT_TOOL = "file:///C:/Users/gojas/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const SKILL_DIR = "C:/Users/gojas/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations";
const RUNTIME_PYTHON = "C:/Users/gojas/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
process.env.RUNTIME_NODE_MODULES = "C:/Users/gojas/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const { Presentation, PresentationFile } = await import(ARTIFACT_TOOL);
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

const DIR = path.dirname(fileURLToPath(import.meta.url));
const WORKSPACE = path.resolve(DIR, "../..");
const OUTPUT = path.join(DIR, "BT_CAP_Midterm_2026-10-19.pptx");
const RENDER_DIR = path.join(DIR, "rendered");
const STAGING = path.join(WORKSPACE, ".codex", "midterm-finalizer");
const IMAGES = {
  overview: new Uint8Array(await fs.readFile(path.join(DIR, "BTCAP_Bioreasoning_Overview.png"))),
  q39: new Uint8Array(await fs.readFile(path.join(DIR, "screenshots", "chat-q39-trace.png"))),
  nrxn1: new Uint8Array(await fs.readFile(path.join(DIR, "screenshots", "genes-nrxn1.png"))),
  novel: new Uint8Array(await fs.readFile(path.join(DIR, "screenshots", "ranked-novel.png"))),
  negative: new Uint8Array(await fs.readFile(path.join(DIR, "screenshots", "not-in-tables.png"))),
};

const C = {
  bg: "#0D1316",
  surface: "#141C20",
  surface2: "#182329",
  text: "#E1E9EC",
  muted: "#92A3A9",
  line: "#2A383D",
  teal: "#46BFC9",
  tealDark: "#17343A",
  green: "#66BF8B",
  greenDark: "#15281D",
  amber: "#E2A84E",
  amberDark: "#2A2013",
  purple: "#A99CF5",
};

function addShape(slide, geometry, left, top, width, height, fill, line = "none", radius = undefined) {
  return slide.shapes.add({
    geometry,
    position: { left, top, width, height },
    fill,
    line: line === "none" ? { style: "solid", fill: "none", width: 0 } : { style: "solid", fill: line, width: 1 },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function addText(slide, text, left, top, width, height, fontSize = 22, color = C.text, bold = false, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontFamily: "Arial",
    fontSize,
    color,
    bold,
    ...(options.italic ? { italic: true } : {}),
    ...(options.align ? { alignment: options.align } : {}),
  };
  return box;
}

function addTitle(slide, title, eyebrow, page = "") {
  addText(slide, eyebrow.toUpperCase(), 64, 34, 700, 24, 13, C.teal, true);
  addText(slide, title, 64, 65, 1110, 58, 34, C.text, true);
  addShape(slide, "rect", 64, 130, 1152, 2, C.line);
  if (page) addText(slide, page, 1170, 37, 46, 22, 12, C.muted, true, { align: "right" });
}

function addFooter(slide, source) {
  addText(slide, source, 64, 684, 1060, 18, 10, C.muted, false);
  addText(slide, "BT-CAP · 18 OCT 2026", 1040, 684, 176, 18, 10, C.muted, true, { align: "right" });
}

function addPill(slide, text, left, top, width, fill = C.greenDark, color = C.green) {
  addShape(slide, "roundRect", left, top, width, 30, fill, color, "rounded-xl");
  addText(slide, text, left + 10, top + 5, width - 20, 20, 13, color, true, { align: "center" });
}

function addMetric(slide, value, label, left, top, width, color = C.teal) {
  addShape(slide, "roundRect", left, top, width, 112, C.surface, C.line, "rounded-2xl");
  addText(slide, value, left + 18, top + 16, width - 36, 42, 34, color, true);
  addText(slide, label, left + 18, top + 63, width - 36, 34, 15, C.muted, false);
}

function addBullet(slide, text, left, top, width, height = 42, accent = C.teal) {
  addShape(slide, "ellipse", left, top + 8, 9, 9, accent);
  addText(slide, text, left + 22, top, width - 22, height, 19, C.text, false);
}

function addImageFrame(slide, imageBytes, alt, left, top, width, height, fit = "cover", crop = undefined) {
  addShape(slide, "roundRect", left - 2, top - 2, width + 4, height + 4, C.surface2, C.line, "rounded-2xl");
  return slide.images.add({
    blob: imageBytes,
    contentType: "image/png",
    alt,
    fit,
    position: { left, top, width, height },
    geometry: "roundRect",
    borderRadius: "rounded-xl",
    ...(crop ? { crop } : {}),
  });
}

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

// 1. Cover
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addShape(slide, "rect", 0, 0, 18, 720, C.teal);
  addText(slide, "BIOREASONING CAPSTONE", 74, 80, 500, 26, 14, C.teal, true);
  addText(slide, "From single-cell evidence\nto auditable gene nominees", 74, 142, 750, 142, 48, C.text, true);
  addText(slide, "Midterm progress · Weeks 1–6", 76, 309, 460, 34, 23, C.muted, false);
  addShape(slide, "roundRect", 820, 100, 372, 444, C.surface, C.line, "rounded-2xl");
  const stages = [
    ["1", "Derived tables", "CellChat · DEG · cell counts"],
    ["2", "Grounded reasoning", "Tool traces before any model answer"],
    ["3", "Gene ranking", "LightGBM + scETM features"],
    ["4", "Portal", "One interface, same evidence"],
  ];
  stages.forEach(([n, title, sub], i) => {
    const y = 134 + i * 96;
    addShape(slide, "ellipse", 852, y, 42, 42, i === 3 ? C.greenDark : C.tealDark, i === 3 ? C.green : C.teal);
    addText(slide, n, 864, y + 7, 18, 24, 17, i === 3 ? C.green : C.teal, true, { align: "center" });
    addText(slide, title, 916, y - 2, 238, 27, 19, C.text, true);
    addText(slide, sub, 916, y + 28, 238, 34, 14, C.muted, false);
  });
  addPill(slide, "WEEK 6 VERIFIED", 76, 566, 172);
  addText(slide, "Jasper Chen · BT-CAP", 76, 622, 400, 30, 17, C.text, true);
  addText(slide, "19 October 2026", 76, 653, 300, 24, 14, C.muted, false);
  slide.speakerNotes.text = [
    "Open with the project goal: convert large single-cell tables into traceable evidence and ranked validation candidates.",
    "The new contribution this week is the portal integration. It does not retrain the models; it exposes the already verified agent and ranker through one user interface.",
  ];
}

// 2. System overview
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(slide, "One evidence base now powers two analysis paths", "System overview", "02");
  addImageFrame(
    slide,
    IMAGES.overview,
    "BioReasoning Capstone architecture diagram showing data, reasoning agent, gene ranker, and Week 6 portal",
    64, 154, 1152, 488, "cover",
    { left: 0, top: 0.01, right: 0, bottom: 0.39 },
  );
  addPill(slide, "DONE", 1064, 610, 108);
  addFooter(slide, "Source: BioReasoning Capstone.html · verified 18 Oct 2026");
  slide.speakerNotes.text = [
    "Walk left to right: single-cell data and public gene lists feed derived tables and scETM features.",
    "The upper lane answers questions with explicit table tools. The lower lane ranks genes with held-out LightGBM scores.",
    "Week 5 linked the ranker back into the agent through nominate_lookup. Week 6 exposes both lanes through the portal.",
  ];
}

// 3. Grounded reasoning
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(slide, "The model writes from retrieved evidence, not memory", "Weeks 1–3 · grounded reasoning", "03");
  const labels = [
    ["Question", "Plain-English biological question"],
    ["Router", "Selects CellChat, DEG, count, annotation, or rank tools"],
    ["Trace", "Rows, filters, and source file are preserved"],
    ["Answer", "Extractive or MedGemma response stays tied to the trace"],
  ];
  labels.forEach(([title, sub], i) => {
    const x = 66 + i * 286;
    addShape(slide, "roundRect", x, 165, 250, 112, C.surface, i === 2 ? C.teal : C.line, "rounded-2xl");
    addText(slide, `${i + 1}`, x + 18, 181, 26, 28, 20, C.teal, true);
    addText(slide, title, x + 52, 180, 176, 28, 20, C.text, true);
    addText(slide, sub, x + 18, 216, 214, 48, 14, C.muted, false);
    if (i < 3) {
      addShape(slide, "rect", x + 250, 217, 36, 2, C.teal);
    }
  });
  addImageFrame(slide, IMAGES.q39, "Portal chat with nominate_lookup trace expanded", 66, 316, 645, 326, "cover", { left: 0.42, top: 0.03, right: 0, bottom: 0.06 });
  addShape(slide, "roundRect", 748, 316, 468, 326, C.surface, C.line, "rounded-2xl");
  addText(slide, "Why this matters", 778, 346, 300, 32, 24, C.text, true);
  addBullet(slide, "The trace shows which lookup fired and what filters were used.", 778, 400, 390, 58);
  addBullet(slide, "Cached MedGemma answers are replayed without a new provider call.", 778, 475, 390, 58, C.green);
  addBullet(slide, "Unsupported questions return “not in the tables.”", 778, 550, 390, 48, C.amber);
  addFooter(slide, "Sources: fall_semester/questions.md · WEEK6_PORTAL_REPORT.md");
  slide.speakerNotes.text = [
    "The agent separates retrieval from generation. A model only receives the evidence returned by the tools.",
    "This screenshot shows the Week 5 nominate_lookup tool firing inside the Week 6 chat.",
    "Q9 through Q32 reused 24 cached MedGemma responses. Verification blocked provider entry points and observed zero calls.",
  ];
}

// 4. Ranker
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(slide, "scETM context improves both held-out gene rankers", "Weeks 4–5 · nomination model", "04");
  addMetric(slide, "0.742", "tumor-synaptic AUROC\ntables + scETM", 64, 160, 250, C.teal);
  addMetric(slide, "0.725", "immune-synapse AUROC\ntables + scETM", 334, 160, 250, C.purple);
  addMetric(slide, "5 × 5", "seeds × grouped folds\nout-of-fold scoring", 604, 160, 250, C.green);
  addMetric(slide, "32-d", "gene embedding\n20,381 cells", 874, 160, 250, C.amber);
  addShape(slide, "roundRect", 64, 305, 538, 322, C.surface, C.line, "rounded-2xl");
  addText(slide, "Feature stack", 92, 334, 300, 32, 24, C.text, true);
  addBullet(slide, "CellChat interaction probability and Ephys lean", 92, 385, 450, 42);
  addBullet(slide, "DEG effect size, adjusted p-value, and IDH agreement", 92, 442, 450, 48);
  addBullet(slide, "Group size safeguards from cell counts", 92, 505, 450, 42);
  addBullet(slide, "scETM embedding and neighbor-label fractions", 92, 562, 450, 42, C.purple);
  addShape(slide, "roundRect", 630, 305, 586, 322, C.surface, C.line, "rounded-2xl");
  addText(slide, "Nomination output", 658, 334, 340, 32, 24, C.text, true);
  addText(slide, "NRXN1 · cycling_tumor · IDH_Mutant", 658, 389, 500, 28, 20, C.teal, true);
  addText(slide, "p_tumor_synaptic", 658, 436, 245, 24, 16, C.muted, false);
  addText(slide, "0.900521", 965, 425, 190, 38, 28, C.green, true, { align: "right" });
  addText(slide, "Tumor rank in context", 658, 482, 245, 24, 16, C.muted, false);
  addText(slide, "#9", 965, 473, 190, 34, 25, C.text, true, { align: "right" });
  addText(slide, "Top drivers", 658, 528, 245, 24, 16, C.muted, false);
  addText(slide, "kNN synaptic · emb_07 · emb_15", 658, 562, 500, 32, 17, C.text, true);
  addFooter(slide, "Source: fall_semester/turnin/lgbm_scetm/WEEK5_RESULTS.md");
  slide.speakerNotes.text = [
    "The ranker uses gene-by-context rows. Cross-validation is grouped so a gene is never scored by a model trained on that gene.",
    "Adding scETM features increased tumor AUROC from 0.675 to 0.742 and immune AUROC from 0.666 to 0.725.",
    "The nominee tool returns probabilities, driving features, evidence flags, minimum cell count, and fold provenance.",
  ];
}

// 5. Portal integration
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(slide, "Week 6 turns the pipeline into an explorable portal", "Product integration", "05");
  addImageFrame(slide, IMAGES.nrxn1, "NRXN1 gene evidence card in the portal", 64, 158, 557, 440, "cover", { left: 0, top: 0.05, right: 0, bottom: 0.17 });
  addImageFrame(slide, IMAGES.novel, "Novel-only ranked gene list in the portal", 659, 158, 557, 440, "cover", { left: 0, top: 0.28, right: 0, bottom: 0.02 });
  addPill(slide, "GENE CARD", 86, 574, 132, C.tealDark, C.teal);
  addPill(slide, "NOVEL TOP 10", 681, 574, 150, C.tealDark, C.teal);
  addText(slide, "Scores · evidence · interactions", 232, 578, 340, 24, 15, C.text, true);
  addText(slide, "Filters · CSV export · neighbors", 849, 578, 320, 24, 15, C.text, true);
  addShape(slide, "roundRect", 64, 620, 1152, 42, C.greenDark, C.green, "rounded-xl");
  addText(slide, "Both views load the same saved score files used by nominate_lookup, so chat and Genes cannot disagree.", 86, 630, 1110, 24, 16, C.green, true, { align: "center" });
  addFooter(slide, "Sources: frontend-react/src/components/GenesTab.jsx · backend/api/reasoning.py");
  slide.speakerNotes.text = [
    "The Genes tab adds four panels: a gene card, ranked list, scETM neighbors, and evidence tables.",
    "The backend loads the reasoning engine, rank scores, and embeddings once and serves them through the new reasoning and gene endpoints.",
    "The central consistency decision is that the portal and nominate_lookup share one loader and the same score files.",
  ];
}

// 6. Verification table
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(slide, "The integrated portal passes every required check", "Verification results", "06");
  addMetric(slide, "44 / 44", "questions matched\nAPI ↔ CLI", 64, 158, 250, C.green);
  addMetric(slide, "24 / 24", "MedGemma cache hits\nzero provider calls", 334, 158, 250, C.green);
  addMetric(slide, "310", "scored outputs checked\npublished-gene equality", 604, 158, 250, C.teal);
  addMetric(slide, "294 + 8", "Python + frontend\ntests passed", 874, 158, 250, C.teal);

  const values = [
    ["Verification", "What was compared", "Result"],
    ["Q1–Q44 regression", "tools + required evidence terms + answer text", "PASS"],
    ["Published 33", "portal scores/ranks vs nominate_lookup", "PASS"],
    ["MedGemma replay", "cache files unchanged; provider entry blocked", "PASS"],
    ["Portal build", "Python tests, React tests, production build", "PASS"],
  ];
  const table = slide.tables.add({ rows: values.length, columns: 3, left: 64, top: 310, width: 1152, height: 286, columnWidths: [260, 676, 216], values });
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({ fill: C.tealDark, textStyle: { fontFamily: "Arial", fontSize: 16, bold: true, color: C.teal }, margins: { left: 14, right: 14, top: 10, bottom: 10 } });
  table.cells.block({ row: 1, column: 0, rowCount: 4, columnCount: 3 }).assign({ fill: C.surface, textStyle: { fontFamily: "Arial", fontSize: 15, color: C.text }, margins: { left: 14, right: 14, top: 9, bottom: 9 } });
  for (let row = 1; row < 5; row += 1) {
    table.getCell(row, 2).fill = C.greenDark;
    table.getCell(row, 2).text.style = { fontFamily: "Arial", fontSize: 15, bold: true, color: C.green };
  }
  addText(slide, "Verified behavior, not biological validation", 64, 618, 600, 24, 16, C.amber, true);
  addText(slide, "These checks establish software consistency and evidence provenance.", 64, 646, 900, 24, 15, C.muted, false);
  addFooter(slide, "Source: fall_semester/turnin/week6_portal/WEEK6_PORTAL_REPORT.md");
  slide.speakerNotes.text = [
    "The 44-of-44 result requires equal tool sets, allowed tool combinations, and every required evidence group on both API and CLI sides.",
    "The published-gene check covered 310 scored outputs. AXL was absent in both places and correctly returned no score.",
    "These are integration and reproducibility checks. They do not replace experimental validation of the biological nominees.",
  ];
}

// 7. Conclusion
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(slide, "We can now move from evidence to reviewable nominees", "Conclusions and next steps", "07");
  addShape(slide, "roundRect", 64, 157, 536, 448, C.surface, C.line, "rounded-2xl");
  addText(slide, "What is complete", 92, 188, 320, 32, 24, C.text, true);
  addBullet(slide, "Grounded question answering across CellChat, DEG, annotation, and counts", 92, 246, 452, 62, C.green);
  addBullet(slide, "Held-out tumor-synaptic and immune-synapse gene ranking", 92, 326, 452, 52, C.green);
  addBullet(slide, "nominate_lookup connects ranker scores back to tool traces", 92, 398, 452, 52, C.green);
  addBullet(slide, "Portal integration, caching, regression tests, and exportable tables", 92, 470, 452, 62, C.green);
  addText(slide, "Next", 92, 553, 80, 24, 16, C.teal, true);
  addText(slide, "Review top novel nominees with the team, then prioritize experimental validation.", 160, 547, 390, 46, 17, C.text, true);
  addImageFrame(slide, IMAGES.negative, "Portal response to an unsupported question showing not in the tables", 638, 157, 578, 330, "cover", { left: 0.36, top: 0.06, right: 0, bottom: 0.03 });
  addShape(slide, "roundRect", 638, 509, 578, 96, C.amberDark, C.amber, "rounded-2xl");
  addText(slide, "Guardrail", 664, 528, 140, 24, 18, C.amber, true);
  addText(slide, "If the evidence is absent, the portal says “not in the tables” instead of inventing a result.", 664, 557, 520, 38, 16, C.text, false);
  addFooter(slide, "Measured results through 18 Oct 2026 · experimental validation remains future work");
  slide.speakerNotes.text = [
    "The portal is now a review surface for the same evidence and rankings produced earlier in the project.",
    "The screenshot demonstrates the grounding boundary: an invented pathway is rejected rather than answered from model memory.",
    "The next scientific step is team review of high-scoring novel nominees and selection of candidates for experimental validation.",
  ];
}

await fs.mkdir(RENDER_DIR, { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(RENDER_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
}
const montage = await presentation.export({ format: "webp", montage: true });
await fs.writeFile(path.join(RENDER_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

await fs.mkdir(STAGING, { recursive: true });
const candidatePath = path.join(STAGING, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

await finalizePresentation({
  explicitTotalSlideCount: 7,
  requiredNativeTableOwnerSlides: [6],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: WORKSPACE,
  candidatePath,
  finalPath: OUTPUT,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "6",
  ],
  fontPolicy: { basis: "design", families: ["Aptos"], scriptFonts: {} },
  verifyArtifactToolImport: true,
  receiptPath: path.join(STAGING, "BT_CAP_Midterm_2026-10-19.pptx.validation.json"),
});

console.log(`Created ${OUTPUT}`);
