import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const TMP_DIR = "/Users/akahmad/Documents/G-ThinkerCG/tmp/rabitq_gpu_deck";
const ASSET_DIR = `${TMP_DIR}/assets`;
const FINAL_PPTX = "/Users/akahmad/Documents/G-ThinkerCG/output/IVF-RaBitQ_GPU_Paper_Presentation.pptx";
const SOURCE =
  "/Users/akahmad/Library/CloudStorage/OneDrive-IndianaUniversity(2)/Vector DB/papers/RabitQ-GPU.pdf";

const W = 1280;
const H = 720;
const C = {
  ink: "#000000",
  white: "#FFFFFF",
  panel: "#EDEDED",
  panel2: "#F7F7F7",
  rule: "#B8BCC4",
  accent: "#6DCBF4",
  accentStrong: "#3D8DFF",
  accentPale: "#D0EDFA",
  red: "#D34A4A",
  green: "#5C9A43",
  grayText: "#4B5563",
};

const assets = {};
for (const [key, file] of Object.entries({
  fig1: "figure1_bitwise.png",
  fig2: "figure2_layout.png",
  table1: "table1_datasets.png",
  fig3: "figure3_qps_recall.png",
  fig4: "figure4_build.png",
  fig5: "figure5_storage.png",
  fig6: "figure6_bits.png",
  fig7: "figure7_gpu_cpu.png",
})) {
  assets[key] = await fs.readFile(`${ASSET_DIR}/${file}`);
}

function addText(slide, text, left, top, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position: { left, top, width, height },
    fill: options.fill ?? "none",
    line: options.line ?? { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: options.fontSize ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    typeface: options.typeface ?? "Helvetica Neue",
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "shrinkText",
  };
  return shape;
}

function addPanel(slide, left, top, width, height, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry ?? "rect",
    name: options.name,
    position: { left, top, width, height },
    fill: options.fill ?? C.panel2,
    line: options.line ?? { style: "solid", fill: C.rule, width: 1 },
    borderRadius: options.borderRadius ?? 0,
  });
}

function addImage(slide, blob, left, top, width, height, alt, fit = "contain") {
  return slide.images.add({
    blob,
    contentType: "image/png",
    alt,
    fit,
    position: { left, top, width, height },
  });
}

function addRule(slide, left, top, width, color = C.rule, weight = 1) {
  return slide.shapes.add({
    geometry: "line",
    position: { left, top, width, height: 0 },
    fill: "none",
    line: { style: "solid", fill: color, width: weight },
  });
}

function baseSlide(presentation, title, number, kicker = "IVF-RaBitQ · GPU-native ANNS") {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addText(slide, kicker.toUpperCase(), 42, 24, 460, 24, {
    fontSize: 16,
    bold: true,
    color: C.accentStrong,
    autoFit: "none",
  });
  addText(slide, title, 42, 54, 1125, 68, {
    fontSize: 48,
    bold: false,
    autoFit: "shrinkText",
    name: `slide-${number}-title`,
  });
  addRule(slide, 42, 126, 1196, C.ink, 1);
  addText(slide, String(number).padStart(2, "0"), 1180, 674, 58, 20, {
    fontSize: 14,
    alignment: "right",
    color: C.grayText,
    autoFit: "none",
  });
  return slide;
}

function addNotes(slide, pages, talkingPoints = "") {
  const pageLabel = Array.isArray(pages) ? pages.join(", ") : pages;
  slide.speakerNotes.textFrame.setText(
    `${talkingPoints}\n\n[Sources]\n- ${SOURCE}, paper page(s) ${pageLabel}.`,
  );
  slide.speakerNotes.setVisible(true);
}

function addMetric(slide, value, label, left, top, width, options = {}) {
  addText(slide, value, left, top, width, 72, {
    fontSize: options.fontSize ?? 58,
    bold: true,
    color: options.color ?? C.accentStrong,
    verticalAlignment: "bottom",
  });
  addText(slide, label, left, top + 76, width, 64, {
    fontSize: 22,
    color: C.grayText,
  });
}

function addNode(slide, title, body, left, top, width, height, options = {}) {
  const node = addPanel(slide, left, top, width, height, {
    fill: options.fill ?? C.panel2,
    line: { style: "solid", fill: options.line ?? C.rule, width: options.lineWidth ?? 1 },
  });
  addText(slide, title, left + 18, top + 16, width - 36, 34, {
    fontSize: 25,
    bold: true,
    color: options.titleColor ?? C.ink,
  });
  addText(slide, body, left + 18, top + 58, width - 36, height - 74, {
    fontSize: options.bodySize ?? 21.5,
    color: C.grayText,
  });
  return node;
}

function addArrow(slide, left, top, width, height = 20, fill = C.accentStrong) {
  return slide.shapes.add({
    geometry: "rightArrow",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

const presentation = Presentation.create({
  slideSize: { width: W, height: H },
});

// 1 — Cover, adapted from Codex Grid slide-02.
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addText(slide, "AKHLAQUE AHMAD", 42, 36, 350, 35, {
    fontSize: 24,
    bold: true,
    color: C.accentStrong,
  });
  addText(slide, "PAPER PRESENTATION · 2026", 830, 36, 408, 35, {
    fontSize: 22,
    alignment: "right",
    color: C.grayText,
  });
  addText(
    slide,
    "GPU-Native Approximate\nNearest Neighbor Search\nwith IVF-RaBitQ",
    42,
    205,
    1040,
    330,
    { fontSize: 74, bold: false, verticalAlignment: "bottom" },
  );
  addRule(slide, 42, 570, 1196, C.ink, 2);
  addText(
    slide,
    "Jifan Shi · Jianyang Gao · James Xia · Tamás Béla Fehér · Cheng Long",
    42,
    594,
    1040,
    40,
    { fontSize: 22, color: C.grayText },
  );
  addText(slide, "arXiv:2602.23999", 1035, 650, 203, 24, {
    fontSize: 16,
    alignment: "right",
    color: C.grayText,
  });
  addNotes(slide, 1, "Introduce the paper and its objective: an end-to-end GPU-native IVF-RaBitQ system.");
}

// 2 — Motivation.
{
  const slide = baseSlide(
    presentation,
    "GPU ANNS must optimize four objectives at once",
    2,
    "Motivation",
  );
  addText(
    slide,
    "Modern semantic search, RAG, and recommendation operate over massive, high-dimensional embeddings. A useful GPU index must do more than answer queries quickly.",
    42,
    154,
    760,
    94,
    { fontSize: 27, color: C.grayText },
  );
  const labels = [
    ["High recall", "Preserve true neighbors at demanding accuracy targets."],
    ["High QPS", "Exploit batching, memory bandwidth, and massive parallelism."],
    ["Fast build", "Avoid expensive graph construction or trained codebooks."],
    ["Compact index", "Keep vectors GPU-resident without raw-vector reranking."],
  ];
  labels.forEach((item, i) => {
    const x = 42 + (i % 2) * 600;
    const y = 286 + Math.floor(i / 2) * 174;
    addPanel(slide, x, y, 560, 140, {
      fill: i === 0 ? C.accentPale : C.panel2,
      line: { style: "solid", fill: i === 0 ? C.accentStrong : C.rule, width: 1 },
    });
    addText(slide, item[0], x + 22, y + 18, 250, 34, {
      fontSize: 28,
      bold: true,
    });
    addText(slide, item[1], x + 22, y + 62, 500, 60, {
      fontSize: 22,
      color: C.grayText,
    });
  });
  addNotes(slide, 1, "Frame the design problem as a four-way systems trade-off.");
}

// 3 — Baseline tension.
{
  const slide = baseSlide(
    presentation,
    "Existing GPU indexes occupy different trade-off points",
    3,
    "Problem",
  );
  addPanel(slide, 42, 158, 570, 435, { fill: C.panel2 });
  addPanel(slide, 628, 158, 610, 435, { fill: C.white, line: { style: "solid", fill: C.ink, width: 1 } });
  addText(slide, "Graph-based indexes", 70, 190, 500, 42, {
    fontSize: 34,
    bold: true,
  });
  addText(slide, "CAGRA and related methods", 70, 244, 500, 28, {
    fontSize: 20,
    color: C.grayText,
  });
  addText(
    slide,
    "• Strong recall-throughput frontier\n• Irregular traversal and memory access\n• Heavy graph construction\n• Raw or lightly compressed storage",
    70,
    304,
    490,
    230,
    { fontSize: 25, color: C.grayText },
  );
  addText(slide, "Cluster-based indexes", 658, 190, 540, 42, {
    fontSize: 34,
    bold: true,
  });
  addText(slide, "IVF-Flat and IVF-PQ", 658, 244, 500, 28, {
    fontSize: 20,
    color: C.grayText,
  });
  addText(
    slide,
    "• Regular, GPU-friendly probing\n• Fast and simple construction\n• IVF-Flat is bandwidth-heavy\n• IVF-PQ often reranks raw vectors for high recall",
    658,
    304,
    520,
    230,
    { fontSize: 25, color: C.grayText },
  );
  addText(
    slide,
    "The paper asks whether a cluster-based index can reach graph-like recall and throughput without graph-like build and storage costs.",
    86,
    618,
    1108,
    48,
    { fontSize: 25, bold: true, alignment: "center" },
  );
  addNotes(slide, [1, 2], "Contrast graph-based and IVF-based baselines, then state the paper's central question.");
}

// 4 — Contributions, based on Codex Grid slide-13.
{
  const slide = baseSlide(
    presentation,
    "The contribution is an end-to-end GPU co-design",
    4,
    "Contributions",
  );
  const items = [
    ["01 · Quantization", "Cluster-at-a-time, block-per-vector encoding plus a two-phase parallel grid search."],
    ["02 · Distance kernels", "LUT and bitwise inner-product schemes for the dominant 1-bit filtering stage."],
    ["03 · Fused search", "Filtering, refinement, and top-K selection execute inside one cluster-local kernel."],
    ["04 · Data layout", "CSR-like inverted lists and interleaved short codes align storage with warp access."],
  ];
  items.forEach((item, i) => {
    const x = 42 + (i % 2) * 604;
    const y = 166 + Math.floor(i / 2) * 230;
    addText(slide, item[0], x, y, 560, 38, {
      fontSize: 28,
      bold: true,
      color: i === 0 ? C.accentStrong : C.ink,
    });
    addRule(slide, x, y + 48, 560, i === 0 ? C.accentStrong : C.rule, 2);
    addText(slide, item[1], x, y + 66, 540, 112, {
      fontSize: 24,
      color: C.grayText,
    });
  });
  addNotes(slide, 2, "Preview the four contributions that structure the remainder of the presentation.");
}

// 5 — IVF + RaBitQ primer.
{
  const slide = baseSlide(
    presentation,
    "IVF narrows the search; RaBitQ compresses each candidate",
    5,
    "Method primer",
  );
  // Arrows first, behind the nodes.
  addArrow(slide, 270, 315, 70);
  addArrow(slide, 580, 315, 70);
  addArrow(slide, 890, 315, 70);
  addNode(slide, "Raw vectors", "N vectors in D dimensions", 42, 252, 230, 150, {
    fill: C.white,
    line: C.ink,
  });
  addNode(slide, "IVF clustering", "Assign each vector to its nearest centroid", 338, 252, 244, 150, {
    fill: C.panel2,
  });
  addNode(slide, "RaBitQ codes", "Normalize residuals, rotate, and encode B bits per dimension", 648, 252, 244, 150, {
    fill: C.accentPale,
    line: C.accentStrong,
  });
  addNode(slide, "Probe and rank", "Visit nearby lists and estimate distances from codes", 958, 252, 280, 150, {
    fill: C.panel2,
  });
  addText(slide, "Index build", 42, 184, 540, 34, {
    fontSize: 26,
    bold: true,
    color: C.grayText,
  });
  addText(slide, "Query time", 958, 184, 280, 34, {
    fontSize: 26,
    bold: true,
    alignment: "right",
    color: C.grayText,
  });
  addText(
    slide,
    "IVF controls how many vectors are examined. RaBitQ controls the bytes moved and the cost of scoring each examined vector.",
    150,
    482,
    980,
    86,
    { fontSize: 28, bold: true, alignment: "center" },
  );
  addNotes(slide, [2, 3], "Explain the division of labor between IVF partitioning and RaBitQ quantization.");
}

// 6 — RaBitQ quantization.
{
  const slide = baseSlide(
    presentation,
    "RaBitQ encodes direction on a rotated integer grid",
    6,
    "RaBitQ background",
  );
  addPanel(slide, 42, 160, 545, 450, { fill: C.panel2 });
  addText(slide, "1 · Center and normalize", 70, 190, 470, 34, {
    fontSize: 28,
    bold: true,
  });
  addText(slide, "o = (oᵣ − c) / ‖oᵣ − c‖", 70, 244, 470, 56, {
    fontSize: 32,
    typeface: "Cambria Math",
    color: C.accentStrong,
  });
  addText(slide, "2 · Rotate into a shared orthogonal basis", 70, 326, 470, 34, {
    fontSize: 28,
    bold: true,
  });
  addText(slide, "o′ = P⁻¹o     and     PᵀP = I", 70, 380, 470, 56, {
    fontSize: 32,
    typeface: "Cambria Math",
    color: C.accentStrong,
  });
  addText(slide, "3 · Store the closest integer direction", 70, 462, 470, 34, {
    fontSize: 28,
    bold: true,
  });
  addText(slide, "x̂ = arg max ⟨x / ‖x‖, o′⟩", 70, 516, 470, 56, {
    fontSize: 32,
    typeface: "Cambria Math",
    color: C.accentStrong,
  });
  addText(slide, "Why this helps", 644, 180, 540, 42, {
    fontSize: 34,
    bold: true,
  });
  addText(
    slide,
    "• No learned quantization codebook\n\n• Distances and dot products are estimated from compact integer codes plus small factors\n\n• Orthogonal rotation preserves geometry while spreading coordinates across the grid\n\n• Bit width B directly controls the storage-accuracy trade-off",
    644,
    246,
    545,
    330,
    { fontSize: 25, color: C.grayText },
  );
  addNotes(slide, 3, "Walk through centering, normalization, orthogonal rotation, and nearest-direction coding.");
}

// 7 — GPU challenges.
{
  const slide = baseSlide(
    presentation,
    "A direct RaBitQ port fights the GPU execution model",
    7,
    "Design challenge",
  );
  const items = [
    ["Sequential factor search", "The CPU enumerates D·2ᴮ⁻¹ critical scales with per-vector state and a priority queue."],
    ["Irregular control flow", "Fine-grained state updates and branches create warp divergence."],
    ["No direct SIMD equivalent", "CPU packed/shuffle and bit-scan routines do not map directly to SIMT execution."],
    ["Pipeline overhead", "Separate filtering, refinement, and top-K kernels repeatedly touch global memory."],
  ];
  items.forEach((item, i) => {
    const y = 164 + i * 116;
    addText(slide, String(i + 1).padStart(2, "0"), 42, y, 70, 42, {
      fontSize: 30,
      bold: true,
      color: C.accentStrong,
    });
    addText(slide, item[0], 130, y, 380, 38, {
      fontSize: 28,
      bold: true,
    });
    addText(slide, item[1], 540, y, 660, 68, {
      fontSize: 23,
      color: C.grayText,
    });
    addRule(slide, 130, y + 84, 1070);
  });
  addText(
    slide,
    "The solution changes both the algorithm and the storage layout—not just the programming language.",
    130,
    630,
    980,
    40,
    { fontSize: 26, bold: true },
  );
  addNotes(slide, 2, "Make clear why GPU-native design is necessary rather than a straightforward CUDA port.");
}

// 8 — Index build pipeline.
{
  const slide = baseSlide(
    presentation,
    "Index construction exposes intra-vector parallelism",
    8,
    "GPU index build",
  );
  addArrow(slide, 284, 292, 64);
  addArrow(slide, 600, 292, 64);
  addArrow(slide, 916, 292, 64);
  addNode(slide, "Balanced K-means", "Learn centroids and assign vectors to IVF lists", 42, 224, 244, 180);
  addNode(slide, "Normalize + rotate", "Process residuals by cluster; use GEMM for the shared orthogonal transform", 348, 224, 254, 180, {
    fill: C.panel2,
  });
  addNode(slide, "Block per vector", "Threads cooperate across dimensions and candidate scaling factors", 664, 224, 254, 180, {
    fill: C.accentPale,
    line: C.accentStrong,
  });
  addNode(slide, "Pack the index", "Separate 1-bit codes, ex-codes, factors, and vector IDs", 980, 224, 258, 180);
  addMetric(slide, ">1M", "960-D vectors quantized per second to 8-bit codes", 76, 478, 320);
  addText(
    slide,
    "Cluster-at-a-time launches bound temporary storage. Block-per-vector mapping enables intra-vector parallelism and coalesced reads.",
    440,
    490,
    750,
    110,
    { fontSize: 27, bold: true },
  );
  addNotes(slide, [2, 4, 5], "Explain the cluster-at-a-time launch granularity and block-per-vector thread mapping.");
}

// 9 — Algorithm 1.
{
  const slide = baseSlide(
    presentation,
    "Algorithm 1 uses two GPU-parallel grids",
    9,
    "Two-phase rescaling search",
  );
  addText(slide, "Objective", 42, 158, 160, 30, {
    fontSize: 24,
    bold: true,
    color: C.grayText,
  });
  addText(slide, "maximize  f(t) = ⟨x(t), o′⟩ / ‖x(t)‖", 42, 194, 540, 50, {
    fontSize: 30,
    typeface: "Cambria Math",
    color: C.accentStrong,
  });
  // Timeline/arrow foundations first.
  addArrow(slide, 328, 342, 72, 18);
  addArrow(slide, 718, 342, 72, 18);
  addNode(slide, "1 · Search range", "Use maxᵢ|o′ᵢ| and B to initialize [tstart, tend].", 42, 288, 288, 150);
  addNode(slide, "2 · Coarse grid", "Evaluate Ncoarse uniformly spaced t values in parallel; reduce to tcenter.", 400, 288, 320, 150, {
    fill: C.accentPale,
    line: C.accentStrong,
  });
  addNode(slide, "3 · Fine grid", "Search [tcenter−δ, tcenter+δ] with Nfine parallel samples.", 790, 288, 448, 150);
  addText(
    slide,
    "Classical ternary search repeatedly shrinks a unimodal interval. Algorithm 1 keeps that coarse-to-fine idea but uses two fixed GPU-parallel rounds—64 coarse and 32 fine samples in the experiments.",
    82,
    492,
    1116,
    102,
    { fontSize: 26, bold: true, alignment: "center" },
  );
  addText(
    slide,
    "O(1) rounds means fixed synchronization depth; each sample still rounds and scores D coordinates.",
    160,
    620,
    960,
    36,
    { fontSize: 20, color: C.grayText, alignment: "center" },
  );
  addNotes(slide, [5, 9], "Describe Algorithm 1 line-by-line and clarify that it is inspired by, but does not literally execute, ternary search.");
}

// 10 — Query pipeline.
{
  const slide = baseSlide(
    presentation,
    "Queries become independent work units",
    10,
    "Search pipeline",
  );
  addArrow(slide, 246, 308, 56);
  addArrow(slide, 496, 308, 56);
  addArrow(slide, 746, 308, 56);
  addArrow(slide, 996, 308, 56);
  const nodes = [
    ["Batch rotation", "One GEMM produces Q′"],
    ["Cluster selection", "Top-nprobe centroids"],
    ["Pair scheduling", "Sort (query, cluster) pairs"],
    ["Cluster-local search", "Filter, refine, local top-K"],
    ["Batch merge", "Global top-K per query"],
  ];
  nodes.forEach((n, i) => {
    addNode(
      slide,
      n[0],
      n[1],
      42 + i * 250,
      246,
      i === 4 ? 196 : 206,
      150,
      i === 3 ? { fill: C.accentPale, line: C.accentStrong } : {},
    );
  });
  addText(
    slide,
    "Sorting pairs by cluster improves code reuse and lets blocks process probed lists independently.",
    148,
    470,
    984,
    68,
    { fontSize: 29, bold: true, alignment: "center" },
  );
  addText(
    slide,
    "The dominant work happens inside cluster-local search, so the rest of the design is organized around that kernel.",
    200,
    566,
    880,
    60,
    { fontSize: 23, color: C.grayText, alignment: "center" },
  );
  addNotes(slide, [4, 5, 6], "Follow a batch from rotation through cluster selection, pair scheduling, local search, and final merge.");
}

// 11 — Two-stage distance estimation.
{
  const slide = baseSlide(
    presentation,
    "Most vectors pay only for a 1-bit distance estimate",
    11,
    "Two-stage distance estimation",
  );
  addPanel(slide, 42, 166, 740, 430, { fill: C.panel2 });
  addText(slide, "All candidates in a probed list", 74, 198, 650, 40, {
    fontSize: 30,
    bold: true,
  });
  addArrow(slide, 120, 286, 460, 34, C.accentStrong);
  addText(slide, "1-bit code filter", 188, 248, 260, 34, {
    fontSize: 26,
    bold: true,
    alignment: "center",
  });
  addArrow(slide, 490, 408, 210, 34, C.green);
  addText(slide, "Only survivors", 480, 366, 220, 32, {
    fontSize: 24,
    bold: true,
    alignment: "center",
  });
  addText(slide, "Refine with ex-code", 468, 464, 250, 34, {
    fontSize: 26,
    bold: true,
    alignment: "center",
  });
  addText(slide, "discard", 82, 378, 120, 30, {
    fontSize: 22,
    color: C.red,
  });
  addText(slide, "short code", 84, 514, 160, 30, {
    fontSize: 22,
    color: C.grayText,
  });
  addText(slide, "long code", 594, 514, 160, 30, {
    fontSize: 22,
    color: C.grayText,
  });
  addText(slide, "Why it matters", 842, 190, 350, 42, {
    fontSize: 34,
    bold: true,
  });
  addText(
    slide,
    "• Filtering touches many vectors and dominates runtime.\n\n• The ex-code is fetched only for promising candidates.\n\n• High recall does not require loading raw vectors for reranking.",
    842,
    266,
    350,
    250,
    { fontSize: 25, color: C.grayText },
  );
  addNotes(slide, [4, 6], "Explain short-code filtering and ex-code refinement, emphasizing avoided memory traffic.");
}

// 12 — LUT vs bitwise.
{
  const slide = baseSlide(
    presentation,
    "Two inner-product kernels target different GPU bottlenecks",
    12,
    "Kernel co-design",
  );
  addPanel(slide, 42, 164, 570, 438, { fill: C.panel2 });
  addPanel(slide, 628, 164, 610, 438, { fill: C.white, line: { style: "solid", fill: C.accentStrong, width: 2 } });
  addText(slide, "Lookup-table kernel", 72, 194, 500, 42, {
    fontSize: 34,
    bold: true,
  });
  addText(
    slide,
    "• Precompute query-dependent partial inner products\n\n• Keep LUTs in shared memory\n\n• Reuse tables across hundreds of threads\n\n• Fast when shared-memory residency is sufficient",
    72,
    266,
    490,
    270,
    { fontSize: 25, color: C.grayText },
  );
  addText(slide, "Bitwise decomposition", 658, 194, 530, 42, {
    fontSize: 34,
    bold: true,
    color: C.accentStrong,
  });
  addText(
    slide,
    "• Quantize each query into signed integers\n\n• Decompose the inner product by bit plane\n\n• Use AND + POPCNT over 32 dimensions per operation\n\n• Much smaller shared-memory footprint",
    658,
    266,
    530,
    270,
    { fontSize: 25, color: C.grayText },
  );
  addText(
    slide,
    "At high dimensionality, LUT capacity can limit occupancy; the bitwise kernel remains lightweight.",
    160,
    628,
    960,
    38,
    { fontSize: 24, bold: true, alignment: "center" },
  );
  addNotes(slide, [6, 7], "Compare the memory-centric LUT method with the compute-centric bitwise method.");
}

// 13 — Figure 1.
{
  const slide = baseSlide(
    presentation,
    "Bitwise decomposition turns dot products into popcounts",
    13,
    "Bitwise kernel",
  );
  addPanel(slide, 42, 158, 1196, 330, { fill: C.white });
  addImage(
    slide,
    assets.fig1,
    58,
    172,
    1164,
    292,
    "Figure 1 from the paper: bitwise inner-product computation",
  );
  addText(slide, "Pack", 64, 524, 120, 32, {
    fontSize: 26,
    bold: true,
    color: C.accentStrong,
  });
  addText(slide, "32 code dimensions per machine word", 64, 562, 300, 52, {
    fontSize: 22,
    color: C.grayText,
  });
  addText(slide, "Decompose", 446, 524, 160, 32, {
    fontSize: 26,
    bold: true,
    color: C.accentStrong,
  });
  addText(slide, "One binary inner product per query bit plane", 446, 562, 330, 52, {
    fontSize: 22,
    color: C.grayText,
  });
  addText(slide, "Accumulate", 858, 524, 160, 32, {
    fontSize: 26,
    bold: true,
    color: C.accentStrong,
  });
  addText(slide, "Shift and add the weighted popcount results", 858, 562, 330, 52, {
    fontSize: 22,
    color: C.grayText,
  });
  addNotes(slide, 7, "Use Figure 1 to show bit-plane decomposition, AND operations, popcount, and weighted accumulation.");
}

// 14 — Fused kernel.
{
  const slide = baseSlide(
    presentation,
    "One fused kernel keeps intermediate state on chip",
    14,
    "Fused cluster-local search",
  );
  addPanel(slide, 42, 164, 1196, 356, {
    fill: C.panel2,
    line: { style: "solid", fill: C.ink, width: 2 },
  });
  addArrow(slide, 296, 314, 54);
  addArrow(slide, 588, 314, 54);
  addArrow(slide, 880, 314, 54);
  addNode(slide, "1-bit estimate", "Score every code", 72, 248, 226, 146, {
    fill: C.white,
  });
  addNode(slide, "Candidate filter", "Reject by lower bound", 350, 248, 240, 146, {
    fill: C.white,
  });
  addNode(slide, "Ex-code refine", "Improve only survivors", 642, 248, 240, 146, {
    fill: C.accentPale,
    line: C.accentStrong,
  });
  addNode(slide, "In-block top-K", "Emit local candidates", 934, 248, 274, 146, {
    fill: C.white,
  });
  addText(slide, "Shared memory + registers", 414, 458, 450, 32, {
    fontSize: 25,
    bold: true,
    alignment: "center",
    color: C.grayText,
  });
  addText(
    slide,
    "Fusion removes sequential kernel launches, intermediate global-memory writes, and synchronization between stages.",
    146,
    566,
    988,
    70,
    { fontSize: 28, bold: true, alignment: "center" },
  );
  addNotes(slide, [7, 8], "Contrast four separate kernels with the paper's fused cluster-local implementation.");
}

// 15 — Figure 2.
{
  const slide = baseSlide(
    presentation,
    "The memory layout is designed around the dominant access pattern",
    15,
    "GPU index layout",
  );
  addImage(
    slide,
    assets.fig2,
    42,
    148,
    1196,
    440,
    "Figure 2 from the paper: IVF-RaBitQ inverted-list data layout",
  );
  addText(
    slide,
    "CSR-like offsets make each cluster contiguous; short codes are interleaved so a warp reads adjacent dimensions from adjacent vectors.",
    96,
    604,
    1088,
    54,
    { fontSize: 24, bold: true, alignment: "center" },
  );
  addNotes(slide, 9, "Explain the separate arrays for short codes, factors, long codes, and IDs, then the interleaved short-code layout.");
}

// 16 — Evaluation.
{
  const slide = baseSlide(
    presentation,
    "Evaluation spans six diverse vector datasets",
    16,
    "Experimental setup",
  );
  addPanel(slide, 42, 158, 600, 448, { fill: C.white });
  addImage(
    slide,
    assets.table1,
    56,
    176,
    572,
    405,
    "Table 1 from the paper: datasets used in evaluation",
  );
  addText(slide, "Platform", 690, 174, 220, 34, {
    fontSize: 29,
    bold: true,
  });
  addText(
    slide,
    "NVIDIA L40S, 48 GB\nCUDA 13.1\nUbuntu 22.04\nBatch size = 10⁴",
    690,
    224,
    430,
    128,
    { fontSize: 24, color: C.grayText },
  );
  addText(slide, "Baselines", 690, 382, 220, 34, {
    fontSize: 29,
    bold: true,
  });
  addText(
    slide,
    "CAGRA\nIVF-Flat\nIVF-PQ with / without refinement\nRaBitQLib CPU counterpart",
    690,
    432,
    500,
    146,
    { fontSize: 24, color: C.grayText },
  );
  addText(
    slide,
    "Default RaBitQ: 8 bits/dimension · Ncoarse = 64 · Nfine = 32",
    90,
    628,
    1100,
    32,
    { fontSize: 22, bold: true, alignment: "center" },
  );
  addNotes(slide, 9, "Summarize platform, datasets, baselines, batch size, and default RaBitQ settings.");
}

// 17 — Recall-QPS.
{
  const slide = baseSlide(
    presentation,
    "IVF-RaBitQ leads the medium-to-high recall region",
    17,
    "Search performance",
  );
  addImage(
    slide,
    assets.fig3,
    42,
    144,
    910,
    500,
    "Figure 3 from the paper: recall-QPS trade-offs across six datasets",
  );
  addPanel(slide, 978, 160, 260, 438, {
    fill: C.panel2,
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  addText(slide, "At Recall = 0.95", 1000, 188, 216, 30, {
    fontSize: 24,
    bold: true,
  });
  addMetric(slide, "2.2×", "average QPS vs. CAGRA", 1000, 234, 210, {
    fontSize: 52,
  });
  addMetric(slide, "2.0–31.4×", "QPS vs. IVF-PQ without refinement", 1000, 370, 220, {
    fontSize: 39,
    color: C.ink,
  });
  addText(
    slide,
    "Bitwise scales better than LUT on very high-dimensional vectors.",
    1000,
    520,
    214,
    58,
    { fontSize: 20, color: C.grayText },
  );
  addNotes(slide, 10, "Read Figure 3 from the high-recall region and compare against CAGRA and both IVF-PQ modes.");
}

// 18 — Build and storage.
{
  const slide = baseSlide(
    presentation,
    "The same index is faster to build and substantially smaller",
    18,
    "System efficiency",
  );
  addImage(
    slide,
    assets.fig4,
    42,
    142,
    850,
    240,
    "Figure 4 from the paper: index build time",
  );
  addImage(
    slide,
    assets.fig5,
    42,
    400,
    850,
    230,
    "Figure 5 from the paper: storage requirements",
  );
  addPanel(slide, 930, 160, 308, 438, { fill: C.panel2 });
  addMetric(slide, "7.7×", "faster index build than CAGRA on average", 956, 188, 252, {
    fontSize: 56,
  });
  addMetric(slide, "4.4×", "faster build than IVF-PQ on average", 956, 336, 252, {
    fontSize: 56,
    color: C.ink,
  });
  addText(
    slide,
    "Wiki-all example:\n7.49 GB for IVF-RaBitQ\n>30 GB for CAGRA and IVF-PQ with refinement",
    956,
    494,
    250,
    92,
    { fontSize: 21, color: C.grayText },
  );
  addNotes(slide, 11, "Use Figures 4 and 5 to connect algorithmic simplicity to build time and raw-vector avoidance to storage.");
}

// 19 — Bit budget and GPU/CPU.
{
  const slide = baseSlide(
    presentation,
    "Bit width controls recall; GPU execution preserves accuracy",
    19,
    "Sensitivity and portability",
  );
  addText(slide, "Quantization-bit trade-off", 42, 150, 520, 34, {
    fontSize: 28,
    bold: true,
  });
  addImage(
    slide,
    assets.fig6,
    42,
    194,
    560,
    320,
    "Figure 6 from the paper: bit-width recall-QPS trade-off",
  );
  addText(slide, "GPU versus CPU", 656, 150, 520, 34, {
    fontSize: 28,
    bold: true,
  });
  addImage(
    slide,
    assets.fig7,
    656,
    194,
    582,
    320,
    "Figure 7 from the paper: GPU versus CPU recall-QPS trade-off",
  );
  addText(slide, "B = 5", 82, 548, 120, 42, {
    fontSize: 35,
    bold: true,
    color: C.accentStrong,
  });
  addText(slide, ">0.95 recall", 188, 554, 240, 34, {
    fontSize: 24,
    color: C.grayText,
  });
  addText(slide, "B = 7", 82, 606, 120, 42, {
    fontSize: 35,
    bold: true,
    color: C.ink,
  });
  addText(slide, ">0.99 recall", 188, 612, 240, 34, {
    fontSize: 24,
    color: C.grayText,
  });
  addMetric(slide, "12.9×", "average GPU QPS speedup at Recall@10 = 0.95", 706, 526, 430, {
    fontSize: 48,
  });
  addNotes(slide, [11, 12], "Discuss the bit-width control knob, then show that the GPU version matches CPU recall while greatly increasing QPS.");
}

// 20 — Close, adapted from Codex Grid slide-26.
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addText(slide, "TAKEAWAYS", 42, 36, 260, 34, {
    fontSize: 24,
    bold: true,
    color: C.accentStrong,
  });
  addText(
    slide,
    "IVF-RaBitQ shifts the\nGPU ANNS frontier",
    42,
    132,
    900,
    180,
    { fontSize: 72, bold: false, verticalAlignment: "bottom" },
  );
  addRule(slide, 42, 346, 1196, C.ink, 2);
  addText(
    slide,
    "GPU-native quantization",
    42,
    382,
    360,
    34,
    { fontSize: 27, bold: true },
  );
  addText(
    slide,
    "Parallel factor exploration makes RaBitQ practical at scale.",
    42,
    426,
    350,
    86,
    { fontSize: 22, color: C.grayText },
  );
  addText(
    slide,
    "Kernel + layout co-design",
    450,
    382,
    360,
    34,
    { fontSize: 27, bold: true },
  );
  addText(
    slide,
    "Two-stage scoring, bitwise/LUT kernels, and fusion reduce data movement.",
    450,
    426,
    350,
    86,
    { fontSize: 22, color: C.grayText },
  );
  addText(
    slide,
    "Strong measured trade-off",
    858,
    382,
    360,
    34,
    { fontSize: 27, bold: true },
  );
  addText(
    slide,
    "High recall and QPS with fast builds and compact storage.",
    858,
    426,
    350,
    86,
    { fontSize: 22, color: C.grayText },
  );
  addPanel(slide, 42, 558, 1196, 92, {
    fill: C.panel2,
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  addText(
    slide,
    "Scope to remember: one L40S, GPU-resident large batches, and an empirically tuned two-phase grid search. Small-batch latency, out-of-core behavior, and broader GPU portability remain useful next questions.",
    68,
    578,
    1144,
    54,
    { fontSize: 20.5, color: C.grayText, alignment: "center" },
  );
  addNotes(slide, [2, 12], "Close with the three design lessons, then state the evaluation scope and open questions.");
}

await fs.mkdir(`${TMP_DIR}/rendered`, { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(
    `${TMP_DIR}/rendered/${stem}.png`,
    new Uint8Array(await png.arrayBuffer()),
  );
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${TMP_DIR}/rendered/${stem}.layout.json`, await layout.text());
}

const montage = await presentation.export({
  format: "webp",
  montage: true,
  scale: 1,
});
await fs.writeFile(
  `${TMP_DIR}/rendered/deck-montage.webp`,
  new Uint8Array(await montage.arrayBuffer()),
);

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(FINAL_PPTX);
console.log(FINAL_PPTX);
