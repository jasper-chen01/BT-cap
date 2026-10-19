import fs from "node:fs/promises";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const ARTIFACT_TOOL = "file:///C:/Users/gojas/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const SKILL_DIR = "C:/Users/gojas/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations";
const RUNTIME_PYTHON = "C:/Users/gojas/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
process.env.RUNTIME_NODE_MODULES = "C:/Users/gojas/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";

const { Presentation, PresentationFile } = await import(ARTIFACT_TOOL);
const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);

const DIR = path.dirname(fileURLToPath(import.meta.url));
const WORKSPACE = path.resolve(DIR, "../..");
const SOURCE_PDF = path.join(DIR, "source", "Midterm Presentation Fall 2026.pdf");
const OUTPUT = path.join(DIR, "Midterm_Presentation_Fall_2026_with_Portal.pptx");
const STAGING = path.join(WORKSPACE, ".codex", "combined-midterm-finalizer");
const PAGE_DIR = path.join(STAGING, "source-pages");
const RENDER_DIR = path.join(DIR, "rendered-combined");

await fs.rm(PAGE_DIR, { recursive: true, force: true });
await fs.mkdir(PAGE_DIR, { recursive: true });
execFileSync(
  RUNTIME_PYTHON,
  [
    "-c",
    [
      "import pathlib,sys,pypdfium2 as pdfium",
      "source=pdfium.PdfDocument(sys.argv[1])",
      "output=pathlib.Path(sys.argv[2])",
      "output.mkdir(parents=True, exist_ok=True)",
      "[(page.render(scale=2).to_pil().save(output / f'page-{index+1:02d}.png')) for index,page in enumerate(source)]",
    ].join(";"),
    SOURCE_PDF,
    PAGE_DIR,
  ],
  { stdio: "inherit" },
);

const sourcePages = await Promise.all(
  Array.from({ length: 19 }, async (_, index) =>
    new Uint8Array(await fs.readFile(path.join(PAGE_DIR, `page-${String(index + 1).padStart(2, "0")}.png`))),
  ),
);

const images = {
  overview: new Uint8Array(await fs.readFile(path.join(DIR, "BTCAP_Bioreasoning_Overview.png"))),
  gene: new Uint8Array(await fs.readFile(path.join(DIR, "screenshots", "genes-nrxn1.png"))),
};

const C = {
  bg: "#11002B",
  surface: "#21113F",
  surface2: "#180A34",
  text: "#FFFFFF",
  muted: "#C8C0D8",
  line: "#43265F",
  cyan: "#13D8E5",
  pink: "#FF1F69",
  green: "#59E0B6",
};

function addShape(slide, geometry, left, top, width, height, fill, line = "none", radius = undefined) {
  return slide.shapes.add({
    geometry,
    position: { left, top, width, height },
    fill,
    line: line === "none"
      ? { style: "solid", fill: "none", width: 0 }
      : { style: "solid", fill: line, width: 1 },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function addText(slide, text, left, top, width, height, fontSize = 20, color = C.text, bold = false, options = {}) {
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
    ...(options.align ? { alignment: options.align } : {}),
  };
  return box;
}

function addTitle(slide, title, subtitle) {
  addText(slide, title, 46, 38, 1135, 44, 31, C.text, true);
  addText(slide, subtitle, 48, 88, 1120, 34, 17, C.muted, false);
  addShape(slide, "rect", 48, 130, 1184, 2, C.line);
  addShape(slide, "rect", 48, 130, 172, 3, C.pink);
}

function addBullet(slide, text, left, top, width, height = 54, color = C.text) {
  addShape(slide, "ellipse", left, top + 9, 9, 9, C.cyan);
  addText(slide, text, left + 22, top, width - 22, height, 18, color, false);
}

function addImage(slide, bytes, alt, left, top, width, height, crop = undefined) {
  addShape(slide, "roundRect", left - 2, top - 2, width + 4, height + 4, C.surface2, C.line, "rounded-xl");
  return slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: "cover",
    position: { left, top, width, height },
    geometry: "roundRect",
    borderRadius: "rounded-lg",
    ...(crop ? { crop } : {}),
  });
}

function addSourceSlide(presentation, bytes, pageNumber) {
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt: `Original midterm presentation page ${pageNumber}`,
    fit: "cover",
    position: { left: 0, top: 0, width: 1280, height: 720 },
  });
  return slide;
}

function addMetric(slide, value, label, left, color = C.cyan) {
  addShape(slide, "rect", left, 166, 260, 3, color);
  addText(slide, value, left, 184, 260, 56, 38, color, true);
  addText(slide, label, left, 244, 260, 50, 16, C.muted, false);
}

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

for (let index = 0; index < 17; index += 1) {
  addSourceSlide(presentation, sourcePages[index], index + 1);
}

// 18. Portal architecture
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(
    slide,
    "Portal Integration",
    "Week 6 connects the grounded reasoning agent and gene ranker to the BT-CAP website.",
  );
  addBullet(slide, "Chat routes questions to the same CellChat, DEG, count, annotation, and nomination tools used by the CLI.", 54, 158, 564, 64);
  addBullet(slide, "The Genes tab reads held-out LightGBM scores, evidence flags, and scETM neighbors.", 54, 226, 564, 56);
  addBullet(slide, "The portal reads saved outputs from Weeks 1–5. It does not retrain either model.", 54, 288, 564, 56, C.green);
  addShape(slide, "roundRect", 662, 160, 566, 184, C.surface, C.line, "rounded-xl");
  addText(slide, "One shared evidence layer", 690, 184, 510, 30, 23, C.text, true);
  addText(slide, "The chat and Genes views load the same rank files and reasoning tables, preventing conflicting scores or provenance.", 690, 228, 500, 88, 18, C.muted, false);
  addImage(
    slide,
    images.overview,
    "Week 6 portal lane in the BioReasoning Capstone architecture",
    48,
    382,
    1184,
    280,
    { left: 0, top: 0.327, right: 0, bottom: 0.447 },
  );
  slide.speakerNotes.text = [
    "Week 6 wraps the reasoning agent and ranker in the existing BT-CAP portal.",
    "The portal performs no training. It reads the derived tables, score files, and embeddings produced earlier in the project.",
    "The same loader powers nominate_lookup and the Genes tab, so both views share the same held-out scores and evidence.",
  ];
}

// 19. Gene evidence view
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(
    slide,
    "Auditable Gene Nomination",
    "A gene score now links directly to its context, evidence, features, and cross-validation fold.",
  );
  addImage(
    slide,
    images.gene,
    "NRXN1 evidence and CellChat results in the portal Genes tab",
    48,
    160,
    770,
    478,
    { left: 0, top: 0.03, right: 0, bottom: 0.17 },
  );
  addText(slide, "Each gene card reports", 854, 166, 330, 32, 23, C.text, true);
  addBullet(slide, "Tumor-synaptic and immune-synapse probabilities", 854, 218, 334, 58);
  addBullet(slide, "The three features that drove the score", 854, 286, 334, 54);
  addBullet(slide, "CellChat and DEG evidence with source rows", 854, 350, 334, 54);
  addBullet(slide, "Minimum group size and the fold that scored the gene", 854, 414, 334, 64);
  addShape(slide, "roundRect", 846, 504, 358, 134, C.surface, C.cyan, "rounded-xl");
  addText(slide, "Consistency rule", 870, 524, 300, 24, 17, C.cyan, true);
  addText(slide, "The gene card and chat use the same score loader, so NRXN1 receives the same result in both views.", 870, 556, 302, 70, 16, C.text, false);
  slide.speakerNotes.text = [
    "The Genes tab exposes the Week 5 nominate_lookup output as a visual gene card.",
    "Users can filter ranked candidates, inspect CellChat and DEG support, review scETM neighbors, and export CSV tables.",
    "The default ranked view enforces a minimum cell-group size of 50.",
  ];
}

// 20. Portal verification
{
  const slide = presentation.slides.add({ background: { fill: C.bg } });
  addTitle(
    slide,
    "Portal Regression Results",
    "The web interface now reproduces the verified command-line behavior across the full question set.",
  );
  addMetric(slide, "44 / 44", "questions matched across API and CLI", 48, C.green);
  addMetric(slide, "24 / 24", "MedGemma answers replayed from cache", 346, C.green);
  addMetric(slide, "294", "Python and RAG tests passed", 644, C.cyan);
  addMetric(slide, "8", "frontend tests plus production build", 942, C.cyan);

  const values = [
    ["Check", "Requirement", "Result"],
    ["Grounded Q1–Q44", "Same tools and required evidence terms", "PASS"],
    ["Published genes", "Portal scores and ranks match nominate_lookup", "PASS"],
    ["MedGemma replay", "Cached answers return without a provider call", "PASS"],
    ["Production portal", "Backend, frontend, and build checks complete", "PASS"],
  ];
  const table = slide.tables.add({
    rows: values.length,
    columns: 3,
    left: 48,
    top: 326,
    width: 1184,
    height: 274,
    columnWidths: [278, 676, 230],
    values,
  });
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({
    fill: C.surface2,
    textStyle: { fontFamily: "Arial", fontSize: 16, bold: true, color: C.cyan },
    margins: { left: 14, right: 14, top: 10, bottom: 10 },
  });
  table.cells.block({ row: 1, column: 0, rowCount: 4, columnCount: 3 }).assign({
    fill: C.surface,
    textStyle: { fontFamily: "Arial", fontSize: 15, color: C.text },
    margins: { left: 14, right: 14, top: 9, bottom: 9 },
  });
  for (let row = 1; row < 5; row += 1) {
    table.getCell(row, 2).fill = "#15352F";
    table.getCell(row, 2).text.style = { fontFamily: "Arial", fontSize: 15, bold: true, color: C.green };
  }
  addShape(slide, "roundRect", 48, 624, 1184, 48, C.surface2, C.line, "rounded-lg");
  addText(slide, "These checks establish software consistency and evidence provenance. Experimental validation remains the next scientific step.", 70, 636, 1140, 24, 15, C.muted, false, { align: "center" });
  slide.speakerNotes.text = [
    "All 44 required questions produced matching tool calls and required terms through the portal API and the command line.",
    "All 24 saved MedGemma answers replayed from cache while provider entry points were blocked.",
    "The test suite contains 294 Python and RAG tests plus eight frontend tests. The production frontend build also passed.",
    "These results verify integration and provenance. They do not establish that a nominee will succeed experimentally.",
  ];
}

for (let index = 17; index < 19; index += 1) {
  addSourceSlide(presentation, sourcePages[index], index + 1);
}

await fs.rm(RENDER_DIR, { recursive: true, force: true });
await fs.mkdir(RENDER_DIR, { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(
    path.join(RENDER_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`),
    new Uint8Array(await png.arrayBuffer()),
  );
}
const montage = await presentation.export({ format: "webp", montage: true });
await fs.writeFile(path.join(RENDER_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

await fs.mkdir(STAGING, { recursive: true });
const candidatePath = path.join(STAGING, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

await finalizePresentation({
  explicitTotalSlideCount: 22,
  requiredNativeTableOwnerSlides: [20],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: WORKSPACE,
  candidatePath,
  finalPath: OUTPUT,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
    "--require-native-table-slide", "20",
  ],
  fontPolicy: { basis: "design", families: ["Aptos"], scriptFonts: {} },
  verifyArtifactToolImport: true,
  receiptPath: path.join(STAGING, "Midterm_Presentation_Fall_2026_with_Portal.pptx.validation.json"),
});

console.log(`Created ${OUTPUT}`);
