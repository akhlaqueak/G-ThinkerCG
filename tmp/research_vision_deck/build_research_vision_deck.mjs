import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "/Users/akahmad/Documents/G-ThinkerCG/Research_Vision_ARIA_Memory_System.pptx";
const TMP = "/Users/akahmad/Documents/G-ThinkerCG/tmp/research_vision_deck";
const SOURCE_DOC = "/Users/akahmad/Downloads/Research Vison Exercise.docx";

const W = 1280;
const H = 720;
const C = {
  ink: "#0B0F19",
  muted: "#536071",
  light: "#F2F4F7",
  panel: "#EDEDED",
  rule: "#B8BCC4",
  accent: "#3D8DFF",
  accentLight: "#D0EDFA",
  bluePale: "#EAF6FE",
  white: "#FFFFFF",
};

function addText(slide, text, x, y, w, h, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontFace: "Helvetica Neue",
    fontSize: opts.size ?? 22,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "left",
  };
  return box;
}

function addPanel(slide, x, y, w, h, opts = {}) {
  return slide.shapes.add({
    geometry: opts.geometry ?? "rect",
    position: { left: x, top: y, width: w, height: h },
    fill: opts.fill ?? C.light,
    line: { style: "solid", fill: opts.line ?? C.rule, width: opts.width ?? 1 },
  });
}

function addRule(slide, x, y, w, color = C.ink, weight = 2) {
  addPanel(slide, x, y, w, weight, { fill: color, line: color, width: 0 });
}

function addHeader(slide, title, section = "ARIA research vision") {
  addText(slide, section, 56, 36, 420, 26, { size: 15, bold: true, color: C.muted });
  addText(slide, title, 56, 78, 1080, 84, { size: 36, bold: true });
  addRule(slide, 56, 170, 1168, C.rule, 1);
}

function addFooter(slide, n) {
  addText(slide, String(n).padStart(2, "0"), 1186, 668, 40, 24, {
    size: 14,
    color: C.muted,
    align: "right",
  });
}

function notes(slide, lines) {
  slide.speakerNotes.textFrame.setText([
    ...lines,
    "",
    `[Sources] Research vision exercise prompt: ${SOURCE_DOC}`,
  ]);
  slide.speakerNotes.setVisible(true);
}

function bulletList(slide, items, x, y, w, size = 22, gap = 44) {
  items.forEach((item, i) => {
    const top = y + i * gap;
    addPanel(slide, x, top + 9, 8, 8, { fill: C.accent, line: C.accent, width: 0 });
    addText(slide, item, x + 26, top, w - 26, gap - 4, { size, color: C.ink });
  });
}

function slideTitle(p, title, subtitle) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addText(slide, "Research Scientist Interview | AI(X) Hub", 56, 46, 600, 28, {
    size: 16,
    bold: true,
    color: C.muted,
  });
  addText(slide, title, 56, 190, 860, 150, { size: 56, bold: true });
  addRule(slide, 56, 363, 240, C.accent, 5);
  addText(slide, subtitle, 56, 398, 780, 80, { size: 26, color: C.muted });
  addPanel(slide, 932, 122, 250, 420, { fill: C.bluePale, line: C.accentLight });
  addText(slide, "Retrieval\nMemory\nContinual\nEvaluation", 962, 178, 190, 260, {
    size: 34,
    bold: true,
    color: C.ink,
  });
  addText(slide, "Topic 2", 960, 492, 180, 34, { size: 21, bold: true, color: C.accent });
  notes(slide, [
    "Open by saying that ARIA should not be treated as a single predictive model.",
    "The core thesis: ARIA is an evidence-grounded memory system that retrieves, updates, and explains prior experience as the archive evolves.",
  ]);
  return slide;
}

function slideFrameProblem(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "The hard problem is not prediction alone");
  addText(slide, "ARIA must reason over a living clinical archive where evidence is heterogeneous, incomplete, temporal, and continuously changing.", 72, 188, 640, 116, {
    size: 30,
    bold: true,
  });
  addPanel(slide, 760, 196, 370, 86, { fill: C.light });
  addText(slide, "Images", 790, 224, 120, 34, { size: 26, bold: true });
  addPanel(slide, 760, 305, 370, 86, { fill: C.light });
  addText(slide, "Clinical notes", 790, 333, 210, 34, { size: 26, bold: true });
  addPanel(slide, 760, 414, 370, 86, { fill: C.light });
  addText(slide, "Structured events", 790, 442, 250, 34, { size: 26, bold: true });
  addPanel(slide, 760, 523, 370, 86, { fill: C.light });
  addText(slide, "Outcomes + provenance", 790, 551, 300, 34, { size: 26, bold: true });
  addRule(slide, 72, 348, 500, C.rule, 1);
  bulletList(slide, [
    "Retrieve relevant evidence, not just similar records",
    "Update memory without forgetting or drifting silently",
    "Explain answers through traceable archive evidence",
  ], 72, 388, 600, 22, 50);
  addFooter(slide, 2);
  notes(slide, [
    "Use this slide to define the tension. If ARIA only optimizes predictive accuracy, it misses the archive use case.",
    "The system needs to support multiple biomedical tasks by making evidence retrievable and auditable.",
  ]);
}

function slideThesis(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "My proposal: build ARIA as an evidence-grounded memory system");
  const xs = [74, 350, 626, 902];
  const labels = [
    ["Ingest", "Normalize multimodal cases into stable entities and events."],
    ["Represent", "Maintain linked text, image, structured, and temporal views."],
    ["Retrieve", "Combine semantic search, filters, graph expansion, and provenance."],
    ["Update", "Refresh indexes often; update models cautiously and measurably."],
  ];
  xs.forEach((x, i) => {
    addPanel(slide, x, 224, 220, 198, { fill: i === 2 ? C.bluePale : C.light, line: i === 2 ? C.accent : C.rule });
    addText(slide, labels[i][0], x + 24, 252, 170, 38, { size: 27, bold: true });
    addText(slide, labels[i][1], x + 24, 314, 166, 82, { size: 17, color: C.muted });
    if (i < xs.length - 1) {
      addText(slide, "→", x + 232, 292, 38, 42, { size: 34, bold: true, color: C.accent });
    }
  });
  addText(slide, "The system should answer with evidence trails: what was retrieved, why it was relevant, and whether current behavior is improving over time.", 126, 500, 1000, 86, {
    size: 28,
    bold: true,
  });
  addFooter(slide, 3);
  notes(slide, [
    "This is the central architecture claim. ARIA needs an explicit memory layer, not just an embedding model.",
    "Emphasize separation of concerns: data ingestion, representation, retrieval, and updates can evolve independently.",
  ]);
}

function slideMemoryLayers(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "Memory should be layered, not monolithic");
  const layers = [
    ["Raw archive", "Original images, notes, records, outcomes"],
    ["Normalized entities", "Patient, encounter, finding, treatment, time"],
    ["Learned views", "Image/text embeddings and task-specific representations"],
    ["Evidence graph", "Links across entities, modalities, time, and provenance"],
    ["Task views", "Retrieval and reasoning views for specific applications"],
  ];
  layers.forEach((l, i) => {
    const y = 202 + i * 76;
    addPanel(slide, 110 + i * 34, y, 870 - i * 68, 54, {
      fill: i === 3 ? C.bluePale : C.light,
      line: i === 3 ? C.accent : C.rule,
    });
    addText(slide, l[0], 136 + i * 34, y + 13, 220, 26, { size: 22, bold: true });
    addText(slide, l[1], 365 + i * 34, y + 14, 580 - i * 58, 26, { size: 18, color: C.muted });
  });
  addText(slide, "Design implication", 890, 218, 250, 30, { size: 24, bold: true, color: C.accent });
  addText(slide, "A failed retrieval should be debuggable: was the raw data missing, the entity mapping wrong, the embedding stale, or the evidence graph incomplete?", 890, 264, 270, 170, {
    size: 22,
    color: C.ink,
  });
  addFooter(slide, 4);
  notes(slide, [
    "Explain why layers matter: they make failure modes diagnosable.",
    "Tie this to your systems background: scalable systems become research platforms when intermediate representations are explicit and reproducible.",
  ]);
}

function slideRetrieval(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "Retrieval should combine similarity, structure, time, and provenance");
  const cols = [
    ["Semantic retrieval", "Find cases, images, or notes with similar learned representations."],
    ["Structured constraints", "Respect cohort definitions, time windows, labels, and outcomes."],
    ["Graph expansion", "Move from a retrieved item to connected evidence and clinical context."],
  ];
  cols.forEach((c, i) => {
    const x = 80 + i * 380;
    addText(slide, `0${i + 1}`, x, 198, 54, 42, { size: 30, bold: true, color: C.accent });
    addText(slide, c[0], x, 252, 300, 36, { size: 28, bold: true });
    addRule(slide, x, 302, 280, C.rule, 1);
    addText(slide, c[1], x, 328, 310, 130, { size: 22, color: C.muted });
  });
  addPanel(slide, 150, 524, 980, 64, { fill: C.bluePale, line: C.accentLight });
  addText(slide, "Retrieval output should be an evidence set, not a single nearest neighbor.", 180, 542, 920, 30, {
    size: 25,
    bold: true,
    color: C.ink,
  });
  addFooter(slide, 5);
  notes(slide, [
    "This is where you can briefly mention your graph/search work: the archive is naturally a heterogeneous graph plus learned indexes.",
    "Make the key point that retrieval quality must be measured at the evidence-set level.",
  ]);
}

function slideContinualLearning(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "Continual learning needs different update cadences");
  const rows = [
    ["Fast", "Index refresh", "New cases become retrievable quickly."],
    ["Medium", "Representation calibration", "Embeddings and uncertainty are checked against recent data."],
    ["Slow", "Model retraining", "Major model changes require gated evaluation and rollback."],
  ];
  rows.forEach((r, i) => {
    const y = 210 + i * 112;
    addText(slide, r[0], 92, y + 10, 120, 34, { size: 28, bold: true, color: C.accent });
    addPanel(slide, 232, y, 850, 70, { fill: i === 0 ? C.bluePale : C.light, line: i === 0 ? C.accent : C.rule });
    addText(slide, r[1], 262, y + 13, 300, 34, { size: 26, bold: true });
    addText(slide, r[2], 590, y + 18, 440, 28, { size: 20, color: C.muted });
  });
  addText(slide, "Do not let every new case rewrite the model. Let it enter memory first, then require evidence before changing learned behavior.", 126, 570, 1000, 58, {
    size: 27,
    bold: true,
  });
  addFooter(slide, 6);
  notes(slide, [
    "Use this slide to answer the balance between stored knowledge and new information.",
    "The phrase to remember: new evidence enters retrieval quickly; model behavior changes slowly.",
  ]);
}

function slideEvaluation(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "ARIA should be evaluated temporally, continuously, and by task");
  const leftItems = [
    "Temporal splits: archive state at time t, test on future cases",
    "Retrieval relevance: recall@k, precision@k, expert judgment",
    "Clinical task utility: task metrics plus decision-support usefulness",
    "Reliability: calibration, uncertainty, subgroup performance",
    "Drift: embedding shift, label shift, degradation alarms",
  ];
  bulletList(slide, leftItems, 82, 198, 650, 20, 54);
  addPanel(slide, 780, 205, 360, 350, { fill: C.bluePale, line: C.accentLight });
  addText(slide, "Core evaluation loop", 812, 232, 285, 30, { size: 25, bold: true });
  addText(slide, "Snapshot → benchmark → deploy index → monitor → compare", 812, 292, 285, 124, {
    size: 22,
    bold: true,
  });
  addRule(slide, 812, 426, 260, C.accent, 3);
  addText(slide, "Improvement must beat historical performance and stability checks.", 812, 456, 270, 58, { size: 20, color: C.muted });
  addFooter(slide, 7);
  notes(slide, [
    "This is a required topic in the prompt. Be explicit that random train/test splits are not enough for a living archive.",
    "Evaluation cadence: retrieval indexes can be evaluated weekly or monthly; representation/model updates should be gated by stronger retrospective tests.",
  ]);
}

function slideFirst18Months(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "First 18 months: build the research instrument before the ambitious system");
  const milestones = [
    ["0–6 months", "Archive + schema", "Ingestion, entity model, provenance, snapshotting"],
    ["6–12 months", "Hybrid retrieval", "Vector + structured + temporal retrieval baseline"],
    ["12–18 months", "Evaluation harness", "Temporal benchmarks, drift checks, expert review workflow"],
  ];
  milestones.forEach((m, i) => {
    const x = 92 + i * 370;
    addPanel(slide, x, 218, 300, 210, { fill: i === 1 ? C.bluePale : C.light, line: i === 1 ? C.accent : C.rule });
    addText(slide, m[0], x + 24, 242, 240, 28, { size: 22, bold: true, color: C.accent });
    addText(slide, m[1], x + 24, 292, 240, 38, { size: 30, bold: true });
    addText(slide, m[2], x + 24, 350, 235, 58, { size: 19, color: C.muted });
  });
  addText(slide, "The first deliverable is not a perfect reasoner; it is a reproducible platform that makes ARIA measurable and extensible.", 112, 510, 1000, 70, {
    size: 29,
    bold: true,
  });
  addFooter(slide, 8);
  notes(slide, [
    "This is the sequencing answer. Make clear what comes first and why.",
    "Small team logic: build infrastructure that makes later model work cheap, measurable, and trustworthy.",
  ]);
}

function slideDefer(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "What I would deliberately defer");
  const items = [
    ["Autonomous model updates", "Too risky before the evaluation harness is trusted."],
    ["Broad clinical deployment", "Start with retrospective and expert-in-the-loop studies."],
    ["Open-ended reasoning agents", "Useful later, but only after retrieval and provenance are reliable."],
    ["Training foundation models from scratch", "High cost; unclear benefit before platform baselines exist."],
  ];
  items.forEach((it, i) => {
    const y = 198 + i * 92;
    addText(slide, it[0], 92, y, 370, 30, { size: 26, bold: true });
    addText(slide, it[1], 500, y + 3, 560, 30, { size: 22, color: C.muted });
    addRule(slide, 92, y + 56, 960, C.rule, 1);
  });
  addPanel(slide, 920, 515, 210, 68, { fill: C.bluePale, line: C.accentLight });
  addText(slide, "Depth over coverage", 944, 536, 160, 24, { size: 20, bold: true, color: C.accent });
  addFooter(slide, 9);
  notes(slide, [
    "This slide directly responds to the prompt's instruction to state what you are setting aside.",
    "The message should sound disciplined, not conservative: defer what cannot yet be measured well.",
  ]);
}

function slideClose(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addText(slide, "The long-term vision", 56, 48, 430, 34, { size: 20, bold: true, color: C.muted });
  addText(slide, "ARIA becomes a reusable platform for evidence-grounded biomedical AI.", 56, 162, 900, 146, {
    size: 50,
    bold: true,
  });
  addRule(slide, 56, 335, 240, C.accent, 5);
  addText(slide, "Every answer should carry an evidence trail: what was retrieved, why it mattered, how confident the system is, and whether behavior is improving over time.", 56, 378, 920, 112, {
    size: 30,
    color: C.ink,
  });
  addPanel(slide, 1040, 126, 96, 430, { fill: C.bluePale, line: C.accentLight });
  addText(slide, "Q&A", 1055, 315, 72, 38, { size: 30, bold: true, color: C.accent, align: "center" });
  addFooter(slide, 10);
  notes(slide, [
    "Close by returning to the thesis. The platform is useful because it combines retrieval, memory, continual learning, and evaluation.",
    "Invite discussion around tradeoffs: update cadence, expert-in-the-loop review, and what first biomedical use case should anchor the platform.",
  ]);
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(`${TMP}/render`, { recursive: true });
  await fs.writeFile(`${TMP}/source-notes.txt`, `Research vision exercise prompt: ${SOURCE_DOC}\n`);

  const p = Presentation.create({ slideSize: { width: W, height: H } });
  slideTitle(
    p,
    "ARIA as an evidence-grounded memory system",
    "A 20-minute research vision for retrieval, memory, continual learning, evaluation, and sequencing"
  );
  slideFrameProblem(p);
  slideThesis(p);
  slideMemoryLayers(p);
  slideRetrieval(p);
  slideContinualLearning(p);
  slideEvaluation(p);
  slideFirst18Months(p);
  slideDefer(p);
  slideClose(p);

  for (const [i, slide] of p.slides.items.entries()) {
    const png = await p.export({ slide, format: "png", scale: 1 });
    await writeBlob(`${TMP}/render/slide-${String(i + 1).padStart(2, "0")}.png`, png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${TMP}/render/slide-${String(i + 1).padStart(2, "0")}.layout.json`, await layout.text());
  }
  const montage = await p.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(`${TMP}/render/montage.webp`, montage);
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
