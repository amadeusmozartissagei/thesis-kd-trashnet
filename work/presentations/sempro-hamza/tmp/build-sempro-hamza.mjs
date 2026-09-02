import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = String.raw`C:\Users\Pratama\Documents\hamza\COLLEGE STUFF\UNNES\Thesis\thesis-kd-trashnet\outputs\sempro-hamza-knowledge-distillation-wastenet.pptx`;
const PREVIEW_DIR = String.raw`C:\Users\Pratama\Documents\hamza\COLLEGE STUFF\UNNES\Thesis\thesis-kd-trashnet\work\presentations\sempro-hamza\tmp\preview`;
const QA_DIR = String.raw`C:\Users\Pratama\Documents\hamza\COLLEGE STUFF\UNNES\Thesis\thesis-kd-trashnet\work\presentations\sempro-hamza\tmp\qa`;

const W = 1280;
const H = 720;
const C = {
  ink: "#000000",
  muted: "#555555",
  panel: "#EDEDED",
  panel2: "#F5F5F5",
  rule: "#B8BCC4",
  white: "#FFFFFF",
  highlight: "#FF6B35",
};

const titleStyle = { fontSize: 39, typeface: "Helvetica Neue", color: C.ink, verticalAlignment: "top" };
const bodyStyle = { fontSize: 22, typeface: "Helvetica Neue", color: C.ink, verticalAlignment: "top" };
const smallStyle = { fontSize: 16, typeface: "Helvetica Neue", color: C.muted, verticalAlignment: "top" };
const labelStyle = { fontSize: 15, typeface: "Helvetica Neue", color: C.muted, bold: true, verticalAlignment: "top" };

async function writeBlob(path, blob) {
  await fs.writeFile(path, Buffer.from(await blob.arrayBuffer()));
}

function textbox(slide, name, text, position, style = bodyStyle) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { ...style, autoFit: "shrinkText", wrap: "square" };
  return shape;
}

function title(slide, text, footerNo) {
  textbox(slide, "slide-title", text, { left: 42, top: 36, width: 980, height: 70 }, titleStyle);
  textbox(slide, "slide-number", String(footerNo).padStart(2, "0"), { left: 1184, top: 659, width: 54, height: 25 }, {
    fontSize: 15,
    typeface: "Helvetica Neue",
    color: C.ink,
    alignment: "right",
    verticalAlignment: "middle",
  });
}

function kicker(slide, text) {
  textbox(slide, "kicker", text.toUpperCase(), { left: 42, top: 38, width: 300, height: 22 }, labelStyle);
}

function panel(slide, name, position, fill = C.panel) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position,
    fill,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function rule(slide, left, top, width) {
  slide.shapes.add({
    geometry: "rect",
    name: "rule",
    position: { left, top, width, height: 1 },
    fill: C.rule,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function bulletText(items) {
  return items.map((item) => `- ${item}`).join("\n");
}

function addBlackSquare(slide, left, top, size = 24) {
  slide.shapes.add({
    geometry: "rect",
    name: "marker",
    position: { left, top, width: size, height: size },
    fill: C.ink,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function cover(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  kicker(slide, "Seminar Proposal Skripsi");
  textbox(slide, "cover-title",
    "Knowledge Distillation Berbasis Teacher-Assistant untuk Model Ringan WasteNet pada Klasifikasi Citra Sampah TrashNet",
    { left: 42, top: 118, width: 960, height: 245 },
    { fontSize: 50, typeface: "Helvetica Neue", color: C.ink, bold: true, verticalAlignment: "top" },
  );
  panel(slide, "cover-band", { left: 0, top: 438, width: 1280, height: 282 }, C.panel);
  addBlackSquare(slide, 454, 489, 27);
  addBlackSquare(slide, 865, 489, 27);
  textbox(slide, "author-label", "Mahasiswa", { left: 454, top: 535, width: 340, height: 28 }, labelStyle);
  textbox(slide, "author", "Hamza Pratama\n4611422068", { left: 454, top: 566, width: 340, height: 70 }, {
    fontSize: 26,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
  });
  textbox(slide, "program-label", "Program", { left: 865, top: 535, width: 340, height: 28 }, labelStyle);
  textbox(slide, "program", "S1 Teknik Informatika\nFMIPA Universitas Negeri Semarang - 2026", { left: 865, top: 566, width: 340, height: 70 }, {
    fontSize: 24,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
  });
  textbox(slide, "slide-number", "01", { left: 1184, top: 659, width: 54, height: 25 }, {
    fontSize: 15,
    typeface: "Helvetica Neue",
    color: C.ink,
    alignment: "right",
    verticalAlignment: "middle",
  });
}

function agenda(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  kicker(slide, "Alur Presentasi");
  const items = [
    "Konteks dan masalah penelitian",
    "Celah riset dan kebaruan",
    "Rumusan masalah dan tujuan",
    "Dataset TrashNet dan skema label",
    "Model: EfficientNet-B4, Focus-RCNet, WasteNet",
    "Strategi distilasi dan protokol eksperimen",
    "Evaluasi, uji statistik, dan kontribusi",
  ];
  textbox(slide, "agenda-items", items.join("\n"), { left: 42, top: 96, width: 790, height: 540 }, {
    fontSize: 37,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
    lineSpacing: 1.08,
  });
  textbox(slide, "agenda-no", items.map((_, i) => String(i + 1).padStart(2, "0")).join("\n"), { left: 1040, top: 96, width: 180, height: 540 }, {
    fontSize: 37,
    typeface: "Helvetica Neue",
    color: C.ink,
    alignment: "right",
    verticalAlignment: "top",
    lineSpacing: 1.08,
  });
  textbox(slide, "slide-number", "02", { left: 1184, top: 659, width: 54, height: 25 }, {
    fontSize: 15,
    typeface: "Helvetica Neue",
    color: C.ink,
    alignment: "right",
    verticalAlignment: "middle",
  });
}

function contextSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Masalah utama: akurasi perlu bertemu efisiensi", 3);
  textbox(slide, "lead",
    "Pemilahan sampah manual masih rentan tidak konsisten, sementara CNN akurat sering terlalu berat untuk smart bin atau edge device.",
    { left: 42, top: 132, width: 1120, height: 92 },
    { fontSize: 28, typeface: "Helvetica Neue", color: C.ink, verticalAlignment: "top" },
  );
  const cards = [
    ["Kebutuhan aplikasi", "Klasifikasi cepat untuk cardboard, glass, metal, paper, plastic, dan trash."],
    ["Kendala deployment", "Model besar membutuhkan komputasi tinggi, tidak ideal untuk perangkat terbatas."],
    ["Solusi riset", "Knowledge distillation mentransfer dark knowledge dari teacher ke student ringan."],
  ];
  cards.forEach(([h, b], i) => {
    const left = 42 + i * 411;
    panel(slide, `context-card-${i}`, { left, top: 282, width: 375, height: 296 }, C.panel);
    textbox(slide, `context-head-${i}`, h, { left: left + 28, top: 314, width: 315, height: 48 }, {
      fontSize: 27,
      typeface: "Helvetica Neue",
      color: C.ink,
      bold: true,
      verticalAlignment: "top",
    });
    textbox(slide, `context-body-${i}`, b, { left: left + 28, top: 388, width: 315, height: 125 }, bodyStyle);
  });
}

function gapSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Celah riset: capacity gap pada student sangat kecil", 4);
  textbox(slide, "left",
    "Penelitian terdahulu seperti Focus-RCNet melaporkan akurasi sekitar 92% pada TrashNet dengan KD dari EfficientNet-B4. Namun, fokusnya belum pada perbandingan sistematis kapasitas teacher untuk student ultra-ringan.",
    { left: 42, top: 152, width: 520, height: 235 },
    { fontSize: 26, typeface: "Helvetica Neue", color: C.ink, verticalAlignment: "top" },
  );
  panel(slide, "right-panel", { left: 648, top: 126, width: 590, height: 425 }, C.panel);
  textbox(slide, "gap-label", "Kebaruan yang diuji", { left: 690, top: 160, width: 360, height: 28 }, labelStyle);
  textbox(slide, "gap-points", bulletText([
    "Menggunakan WasteNet-128K dan WasteNet-256K sebagai target student ringan.",
    "Membandingkan large-teacher, teacher-assistant, dan self-distillation secara terkontrol.",
    "Menggunakan dua ruang label: K6 dan K5 tanpa kelas trash.",
    "Memvalidasi selisih performa dengan uji statistik."
  ]), { left: 690, top: 210, width: 500, height: 280 }, {
    fontSize: 24,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
  });
}

function questionsSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Rumusan masalah dan tujuan penelitian", 5);
  const qs = [
    ["01", "Bagaimana membangun WasteNet ringan untuk klasifikasi citra sampah TrashNet dengan KD?"],
    ["02", "Bagaimana pengaruh kapasitas teacher terhadap performa student WasteNet?"],
    ["03", "Apakah performa model berbeda signifikan dibanding baseline tanpa KD?"],
  ];
  qs.forEach(([n, q], i) => {
    const top = 150 + i * 136;
    addBlackSquare(slide, 42, top + 8, 28);
    textbox(slide, `q-no-${i}`, n, { left: 92, top, width: 62, height: 42 }, {
      fontSize: 30,
      typeface: "Helvetica Neue",
      color: C.ink,
      bold: true,
      verticalAlignment: "top",
    });
    textbox(slide, `q-text-${i}`, q, { left: 178, top, width: 960, height: 70 }, {
      fontSize: 30,
      typeface: "Helvetica Neue",
      color: C.ink,
      verticalAlignment: "top",
    });
    rule(slide, 178, top + 94, 780);
  });
  textbox(slide, "objective", "Tujuan akhirnya adalah menentukan strategi distilasi paling efektif untuk student berkapasitas kecil pada klasifikasi sampah.", { left: 178, top: 585, width: 840, height: 58 }, smallStyle);
}

function datasetSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Dataset: TrashNet sebagai benchmark utama", 6);
  panel(slide, "metric-1", { left: 42, top: 170, width: 270, height: 230 }, C.panel);
  panel(slide, "metric-2", { left: 354, top: 170, width: 270, height: 230 }, C.panel);
  panel(slide, "metric-3", { left: 666, top: 170, width: 270, height: 230 }, C.panel);
  panel(slide, "metric-4", { left: 978, top: 170, width: 260, height: 230 }, C.panel);
  const metrics = [
    ["2.527", "citra sampah berlabel"],
    ["6", "kelas: cardboard, glass, metal, paper, plastic, trash"],
    ["K6 / K5", "dua ruang label untuk menguji pengaruh kelas trash"],
    ["70/15/15", "split train, validation, test secara stratified"],
  ];
  metrics.forEach(([m, d], i) => {
    const left = [42, 354, 666, 978][i];
    textbox(slide, `m-${i}`, m, { left: left + 28, top: 215, width: 205, height: 58 }, {
      fontSize: 42,
      typeface: "Helvetica Neue",
      color: i === 2 ? C.highlight : C.ink,
      bold: true,
      verticalAlignment: "top",
    });
    textbox(slide, `md-${i}`, d, { left: left + 28, top: 302, width: 205, height: 78 }, smallStyle);
  });
  textbox(slide, "note", "Kelas trash bersifat heterogen dan memiliki jumlah citra paling sedikit, sehingga proposal menguji skema K5 untuk melihat konsistensi efektivitas distilasi.", { left: 42, top: 486, width: 1080, height: 80 }, {
    fontSize: 25,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
  });
}

function modelSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Tiga peran model di sepanjang kurva kapasitas", 7);
  const models = [
    ["EfficientNet-B4", "17,6 juta parameter", "Guru besar dan batas atas performa."],
    ["Focus-RCNet", "520 ribu parameter", "Teacher assistant untuk menjembatani capacity gap."],
    ["WasteNet", "128K / 256K parameter", "Student ringan sebagai target deployment."],
  ];
  models.forEach(([h, stat, b], i) => {
    const left = 42 + i * 411;
    panel(slide, `model-card-${i}`, { left, top: 190, width: 375, height: 370 }, C.panel);
    textbox(slide, `model-h-${i}`, h, { left: left + 28, top: 230, width: 315, height: 70 }, {
      fontSize: 30,
      typeface: "Helvetica Neue",
      color: C.ink,
      bold: true,
      verticalAlignment: "top",
    });
    textbox(slide, `model-stat-${i}`, stat, { left: left + 28, top: 366, width: 315, height: 52 }, {
      fontSize: 27,
      typeface: "Helvetica Neue",
      color: i === 2 ? C.highlight : C.ink,
      verticalAlignment: "top",
    });
    textbox(slide, `model-b-${i}`, b, { left: left + 28, top: 458, width: 315, height: 78 }, smallStyle);
  });
}

function distillationSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Strategi distilasi yang dibandingkan", 8);
  const strategies = [
    ["Direct KD", "EfficientNet-B4 -> WasteNet", "Transfer langsung dari guru besar; menguji risiko capacity gap."],
    ["TA-KD", "Focus-RCNet -> WasteNet", "Teacher assistant menengah digunakan sebagai jembatan kapasitas."],
    ["Self-distillation", "WasteNet gen. sebelumnya -> WasteNet", "Born-Again Networks dengan teacher berkapasitas sama."],
  ];
  strategies.forEach(([h, path, b], i) => {
    const left = 42 + i * 411;
    addBlackSquare(slide, left, 232, 32);
    textbox(slide, `s-h-${i}`, h, { left, top: 294, width: 340, height: 45 }, {
      fontSize: 30,
      typeface: "Helvetica Neue",
      color: C.ink,
      bold: true,
      verticalAlignment: "top",
    });
    textbox(slide, `s-path-${i}`, path, { left, top: 362, width: 350, height: 45 }, {
      fontSize: 24,
      typeface: "Helvetica Neue",
      color: C.highlight,
      verticalAlignment: "top",
    });
    textbox(slide, `s-b-${i}`, b, { left, top: 432, width: 350, height: 105 }, bodyStyle);
  });
  textbox(slide, "formula", "L_KD = (1 - alpha) L_CE + alpha T^2 D_KL(p_teacher || p_student)", { left: 42, top: 595, width: 960, height: 38 }, smallStyle);
}

function procedureSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Prosedur penelitian", 9);
  const steps = [
    ["01", "Studi literatur", "KD, capacity gap, TA-KD, BAN, klasifikasi sampah."],
    ["02", "Data dan manifest", "TrashNet, deduplikasi, split deterministik per seed."],
    ["03", "Perancangan model", "Guru besar, asisten, dan dua varian WasteNet."],
    ["04", "Pelatihan", "Semua skenario memakai konfigurasi identik."],
    ["05", "Evaluasi", "Accuracy, macro F1-score, confusion matrix."],
    ["06", "Uji statistik", "Paired t-test, Wilcoxon, dan McNemar."],
  ];
  steps.forEach(([n, h, b], i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const left = 42 + col * 411;
    const top = 168 + row * 220;
    textbox(slide, `step-no-${i}`, n, { left, top, width: 58, height: 38 }, {
      fontSize: 24,
      typeface: "Helvetica Neue",
      color: C.ink,
      bold: true,
      verticalAlignment: "top",
    });
    textbox(slide, `step-h-${i}`, h, { left: left + 70, top, width: 260, height: 35 }, {
      fontSize: 24,
      typeface: "Helvetica Neue",
      color: C.ink,
      bold: true,
      verticalAlignment: "top",
    });
    textbox(slide, `step-b-${i}`, b, { left: left + 70, top: top + 50, width: 285, height: 72 }, {
      fontSize: 18,
      typeface: "Helvetica Neue",
      color: C.muted,
      verticalAlignment: "top",
    });
    rule(slide, left, top + 132, 335);
  });
}

function pipelineSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Teknik analisis data dan konfigurasi pelatihan", 10);
  textbox(slide, "pipeline-left", bulletText([
    "Preprocessing: RGB, resize 380x380 untuk guru/asisten dan 160x160 untuk WasteNet.",
    "Augmentasi hanya pada data latih: flip, brightness/contrast, dan coarse dropout.",
    "Optimizer SGD momentum 0,9, weight decay 1e-4, cosine annealing.",
    "Pelatihan 100 epoch, mixed precision, checkpoint terbaik dari validation accuracy."
  ]), { left: 42, top: 150, width: 560, height: 390 }, {
    fontSize: 23,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
  });
  panel(slide, "seed-panel", { left: 690, top: 150, width: 450, height: 310 }, C.panel);
  textbox(slide, "seed-title", "Replikasi eksperimen", { left: 728, top: 194, width: 360, height: 42 }, {
    fontSize: 29,
    typeface: "Helvetica Neue",
    color: C.ink,
    bold: true,
    verticalAlignment: "top",
  });
  textbox(slide, "seed-stat", "5 seed", { left: 728, top: 272, width: 220, height: 64 }, {
    fontSize: 50,
    typeface: "Helvetica Neue",
    color: C.highlight,
    bold: true,
    verticalAlignment: "top",
  });
  textbox(slide, "seed-body", "42, 123, 777, 2026, 3407\n\nSemua skenario memakai split dan konfigurasi yang sama agar perbandingan adil.", { left: 728, top: 358, width: 350, height: 88 }, smallStyle);
}

function scenariosSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Skenario pengujian model", 11);
  textbox(slide, "intro", "R1-R9 membentuk benchmark dari baseline, direct KD, teacher-assistant, hingga self-distillation.", { left: 42, top: 112, width: 900, height: 42 }, smallStyle);
  const rows = [
    ["Kode", "Student", "Guru soft-label", "Peran"],
    ["R1", "EfficientNet-B4", "-", "Guru besar / batas atas"],
    ["R2", "Focus-RCNet", "-", "Asisten menengah"],
    ["R3", "WasteNet-128K", "-", "Baseline 128K"],
    ["R4", "WasteNet-128K", "EfficientNet-B4", "Direct KD"],
    ["R5", "WasteNet-128K", "Focus-RCNet", "TA-KD 128K"],
    ["R6", "WasteNet-256K", "-", "Baseline 256K"],
    ["R7", "WasteNet-256K", "EfficientNet-B4", "Direct KD"],
    ["R8", "WasteNet-256K", "Focus-RCNet", "TA-KD 256K"],
    ["R9", "WasteNet-256K", "WasteNet-256K gen. sebelumnya", "Self-distillation"],
  ];
  const table = slide.tables.add({
    rows: rows.length,
    columns: 4,
    left: 42,
    top: 178,
    width: 1196,
    height: 440,
    columnWidths: [80, 300, 365, 451],
    values: rows,
  });
  table.borders.assign({ style: "solid", fill: C.rule, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({
    fill: C.ink,
    textStyle: { fontSize: 15, typeface: "Helvetica Neue", color: C.white, bold: true },
  });
  table.cells.block({ row: 1, column: 0, rowCount: rows.length - 1, columnCount: 4 }).assign({
    textStyle: { fontSize: 14, typeface: "Helvetica Neue", color: C.ink },
    margins: { left: 6, right: 6, top: 3, bottom: 3 },
  });
  table.cells.block({ row: 4, column: 0, rowCount: 2, columnCount: 4 }).assign({ fill: "#FFF3EE" });
  table.cells.block({ row: 8, column: 0, rowCount: 1, columnCount: 4 }).assign({ fill: "#FFF3EE" });
}

function evaluationSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Evaluasi dan validasi statistik", 12);
  const cols = [
    ["Metrik utama", ["Accuracy", "Macro F1-score", "Confusion matrix"]],
    ["Uji antar-seed", ["Paired t-test", "Wilcoxon signed-rank", "alpha = 0,05"]],
    ["Uji sampel", ["McNemar test", "Koreksi kontinuitas", "Pola benar-salah per sampel"]],
  ];
  cols.forEach(([h, items], i) => {
    const left = 42 + i * 411;
    panel(slide, `eval-card-${i}`, { left, top: 210, width: 375, height: 330 }, C.panel);
    textbox(slide, `eval-h-${i}`, h, { left: left + 28, top: 250, width: 315, height: 42 }, {
      fontSize: 29,
      typeface: "Helvetica Neue",
      color: C.ink,
      bold: true,
      verticalAlignment: "top",
    });
    textbox(slide, `eval-b-${i}`, bulletText(items), { left: left + 28, top: 330, width: 315, height: 135 }, {
      fontSize: 24,
      typeface: "Helvetica Neue",
      color: C.ink,
      verticalAlignment: "top",
    });
  });
  textbox(slide, "criteria", "Strategi dinyatakan unggul bila peningkatan performa konsisten terhadap baseline dan didukung hasil uji statistik.", { left: 42, top: 594, width: 1030, height: 42 }, smallStyle);
}

function contributionSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  title(slide, "Kontribusi yang diharapkan", 13);
  addBlackSquare(slide, 42, 360, 34);
  addBlackSquare(slide, 454, 360, 34);
  addBlackSquare(slide, 866, 360, 34);
  textbox(slide, "c1", "Model ringan\nuntuk smart bin", { left: 42, top: 430, width: 330, height: 95 }, {
    fontSize: 30,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
  });
  textbox(slide, "c2", "Rekomendasi\nstrategi KD", { left: 454, top: 430, width: 330, height: 95 }, {
    fontSize: 30,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
  });
  textbox(slide, "c3", "Bukti empiris\ncapacity gap", { left: 866, top: 430, width: 330, height: 95 }, {
    fontSize: 30,
    typeface: "Helvetica Neue",
    color: C.ink,
    verticalAlignment: "top",
  });
  textbox(slide, "closing", "Pertanyaan inti sempro: apakah teacher menengah lebih efektif daripada teacher besar ketika student sangat ringan?", { left: 42, top: 118, width: 1010, height: 105 }, {
    fontSize: 34,
    typeface: "Helvetica Neue",
    color: C.ink,
    bold: true,
    verticalAlignment: "top",
  });
}

async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  cover(presentation);
  agenda(presentation);
  contextSlide(presentation);
  gapSlide(presentation);
  questionsSlide(presentation);
  datasetSlide(presentation);
  modelSlide(presentation);
  distillationSlide(presentation);
  procedureSlide(presentation);
  pipelineSlide(presentation);
  scenariosSlide(presentation);
  evaluationSlide(presentation);
  contributionSlide(presentation);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(`${PREVIEW_DIR}\\${stem}.png`, await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${QA_DIR}\\${stem}.layout.json`, await layout.text(), "utf8");
  }

  await writeBlob(`${QA_DIR}\\deck-montage.webp`, await presentation.export({ format: "webp", montage: true, scale: 1 }));
  await fs.writeFile(`${QA_DIR}\\inspect.ndjson`, (await presentation.inspect({ kind: "slide,textbox,shape,table", maxChars: 20000 })).ndjson, "utf8");
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);
  console.log(OUT);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
