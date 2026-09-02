const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, BorderStyle, ShadingType, PageBreak
} = require("docx");
const fs = require("fs");

const F = "Times New Roman";
const TW = 9020;

const th = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const noB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function t(x, o = {}) {
  return new TextRun({ text: x, font: F, size: o.size || 21, bold: !!o.b, italics: !!o.i, color: o.c });
}
function p(runs, o = {}) {
  return new Paragraph({
    alignment: o.align, spacing: { before: o.before || 0, after: o.after == null ? 90 : o.after, line: o.line || 250 },
    indent: o.ind ? { left: o.ind } : undefined,
    children: Array.isArray(runs) ? runs : [runs],
  });
}
function h1(x) {
  return new Paragraph({
    spacing: { before: 200, after: 120, line: 250 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3864" } },
    children: [t(x, { b: true, size: 26, c: "1F3864" })],
  });
}
function h2(x) {
  return new Paragraph({ spacing: { before: 170, after: 70, line: 250 }, children: [t(x, { b: true, size: 23 })] });
}
function bul(x, o = {}) {
  return new Paragraph({
    bullet: { level: o.level || 0 }, spacing: { after: 55, line: 250 },
    children: Array.isArray(x) ? x : [t(x)],
  });
}
function cell(children, w, o = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined,
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    children: children.map((c) => new Paragraph({
      alignment: o.align, spacing: { after: 0, line: 240 },
      children: Array.isArray(c) ? c : [c],
    })),
  });
}
function tbl(widths, rows, o = {}) {
  return new Table({
    width: { size: TW, type: WidthType.DXA }, columnWidths: widths,
    borders: o.plain
      ? { top: noB, bottom: noB, left: noB, right: noB, insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC" }, insideVertical: noB }
      : { top: th, bottom: th, left: th, right: th, insideHorizontal: th, insideVertical: th },
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0 && !o.plain,
      children: r.map((c, j) => cell(
        Array.isArray(c) ? c : [t(String(c), { b: i === 0 && !o.plain })],
        widths[j],
        { fill: i === 0 && !o.plain ? "DEE6F2" : undefined, align: j === 0 ? AlignmentType.LEFT : (o.centre ? AlignmentType.CENTER : AlignmentType.LEFT) }
      )),
    })),
  });
}
function glos(items) {
  return tbl([2400, 6620], items.map(([a, b]) => [[t(a, { b: true })], [t(b)]]), { plain: true });
}

const kids = [];
const A = (...x) => kids.push(...x);

A(p([t("Cheat Sheet Ablation Study", { b: true, size: 34 })], { align: AlignmentType.CENTER, after: 40 }));
A(p([t("Ablasi Komponen Arsitektur WasteNet-128K dan WasteNet-256K", { b: true, size: 20 })], { align: AlignmentType.CENTER, after: 40 }));
A(p([t("Hamza Pratama · 4611422068 · pendamping Laporan_Ablation_WasteNet.docx dan subbab 4.8 skripsi", { i: true, size: 18, c: "666666" })], { align: AlignmentType.CENTER, after: 200 }));

/* ------------------------------------------------ 1. INTI */
A(h1("1. Inti Ablasi dalam 30 Detik"));
A(p(t("Ablasi adalah cara menjawab pertanyaan penguji: “bagian mana dari model yang paling berpengaruh?” Caranya satu komponen diganti atau dilepas, lalu dilihat akurasinya turun berapa. Empat komponen WasteNet yang diuji: fungsi aktivasi, expand ratio, koneksi residual, dan lapisan tersembunyi pengklasifikasi.")));
A(p([t("Tiga kalimat jawaban: ", { b: true }), t("(1) Komponen paling menentukan adalah "), t("fungsi aktivasi", { b: true }), t(", dan hanya pada model kecil 128K — mengganti SiLU dengan ReLU menurunkan akurasi "), t("5,33 poin persentase (p = 0,004)", { b: true }), t(", kalah di kelima seed. (2) Pada 256K "), t("tidak ada satu pun komponen yang signifikan", { b: true }), t(" — arsitekturnya tahan banting. (3) "), t("Expand ratio berlebih", { b: true }), t(": parameternya dipangkas 64–75 persen tetapi akurasinya praktis tidak berubah.")]));
A(p([t("Kalimat pamungkas: ", { b: true }), t("pentingnya sebuah komponen bergantung pada kapasitas model — saat kapasitas sempit setiap komponen berharga, saat kapasitas longgar model punya cadangan untuk menutupi komponen yang hilang.", { b: true })]));

/* ------------------------------------------------ 2. ANGKA */
A(h1("2. Tabel Angka (hafalkan pola, bukan digitnya)"));

A(h2("WasteNet-128K (R10) — baseline SiLU, expand 2,0, hidden 32"));
A(tbl([2900, 1900, 1200, 1000, 2020], [
  ["Konfigurasi", "Akurasi", "Δ", "p", "Parameter"],
  ["Baseline", "0,7689 ± 0,0172", "—", "—", "128.086"],
  [[t("SiLU → ReLU", { b: true })], [t("0,7156 ± 0,0193", { b: true })], [t("−5,33 pp", { b: true })], [t("0,004 *", { b: true })], [t("128.086 (sama)")]],
  ["Expand 2,0 → 1,0", "0,7689 ± 0,0198", "+0,00 pp", "1,000", "45.638"],
  ["Tanpa hidden 32 → 0", "0,7646 ± 0,0192", "−0,42 pp", "0,634", "122.870"],
]));
A(p([t("ReLU kalah di ", { size: 19 }), t("5 dari 5 seed", { b: true, size: 19 }), t(" (−2,4 · −4,7 · −5,3 · −7,9 · −6,3 pp); macro F1 turun 7,76 pp. Wilcoxon = 0,062, yaitu nilai terkecil yang mungkin untuk n = 5.", { size: 19 })], { after: 40 }));

A(h2("WasteNet-256K (R11) — baseline SiLU, expand 2,5, residual aktif"));
A(tbl([2900, 1900, 1200, 1000, 2020], [
  ["Konfigurasi", "Akurasi", "Δ", "p", "Parameter"],
  ["Baseline", "0,7694 ± 0,0164", "—", "—", "256.002"],
  ["SiLU → ReLU", "0,7873 ± 0,0223", "+1,79 pp", "0,268", "256.002 (sama)"],
  ["Tanpa residual", "0,7662 ± 0,0130", "−0,32 pp", "0,702", "256.002 (sama)"],
  [[t("Expand 2,5 → 1,0", { b: true })], [t("0,7689 ± 0,0091", { b: true })], [t("−0,05 pp", { b: true })], [t("0,918", { b: true })], [t("64.206 (−75%)", { b: true })]],
]));
A(p([t("Seluruh p di atas 0,26 → tidak ada yang signifikan. Baris expand adalah temuan efisiensi: seperempat parameter, akurasi sama.", { size: 19 })], { after: 40 }));

A(h2("Angka metode"));
A(tbl([3000, 6020], [
  ["Hal", "Angka"],
  ["Jumlah pelatihan", "40 run = (4 konfigurasi × 5 seed) × 2 varian model"],
  ["Seed", "42, 123, 777, 2026, 3407 — pembagian data 70:15:15 berbeda tiap seed"],
  ["Skema label", "K6 (enam kelas) saja"],
  ["Uji statistik", "Paired t-test per seed terhadap baseline internal, taraf 0,05"],
  ["Lingkungan", "Kaggle Notebook, 1× NVIDIA Tesla T4, 100 epoch, SGD + cosine annealing"],
  ["Letak di skripsi", "Metode: BAB 3 Skenario Pengujian (baris R10 & R11) · Hasil: subbab 4.8, Tabel 4.8–4.9, Gambar 4.6 · Simpulan: butir ke-4 BAB 5"],
], { plain: false }));

/* ------------------------------------------------ 3. DESAIN */
A(new Paragraph({ children: [new PageBreak()] }));
A(h1("3. Lima Keputusan Desain dan Alasannya"));
A(p([t("Ini bagian yang paling sering dikorek penguji: bukan hasilnya, tapi kenapa dirancang begitu.", { i: true, size: 19, c: "666666" })]));
A(bul([t("Hanya model rancangan sendiri yang diablasi. ", { b: true }), t("Ablasi arsitektur masuk akal untuk WasteNet-128K dan 256K karena sayalah yang merancangnya. EfficientNet-B4 arsitektur siap pakai, dan komponen Focus-RCNet sudah diablasi sendiri oleh Zheng dkk. (2023) sehingga cukup dikutip.")]));
A(bul([t("Baseline dilatih ulang di dalam notebook ablasi. ", { b: true }), t("Tidak memakai angka R3/R6 supaya pipeline, versi kode, dan pengaturan pelatihan terbukti identik antara baseline dan varian. Perbandingannya jadi bersih.")]));
A(bul([t("Residual tidak diablasi pada 128K. ", { b: true }), t("Pada konfigurasi 128K tidak ada blok dengan dimensi masuk dan keluar yang sama, sehingga koneksi residual memang tidak pernah aktif. Mengablasinya nol efek. Pada 256K ada tepat satu koneksi aktif di stage repeat-2, dan itulah yang diablasi.")]));
A(bul([t("Komponen dipisah: bebas parameter vs mengubah parameter. ", { b: true }), t("Aktivasi dan residual tidak mengubah jumlah parameter → efeknya murni komponen. Expand ratio dan hidden layer mengubah jumlah parameter → efeknya bercampur dengan berkurangnya kapasitas, karena itu jumlah parameter dicantumkan di setiap baris.")]));
A(bul([t("K6 saja, bukan K5 dan K6. ", { b: true }), t("Arsitektur WasteNet identik pada kedua skema label, hanya berbeda jumlah neuron keluaran, sehingga kepentingan relatif antarkomponen tidak berubah. Menjalankan 40 run tambahan tidak menambah informasi.")]));

/* ------------------------------------------------ 4. NYAMBUNG */
A(h1("4. Kenapa Ablasi Ini Nyambung dengan Temuan Utama"));
A(p(t("Temuan utama skripsi: TA-KD menaikkan akurasi WasteNet-256K sebesar 1,00 poin persentase (p = 0,024, menang 5 dari 5 seed). Pertanyaan wajarnya: dari mana kenaikan itu?")));
A(p([t("Ablasi menjawabnya secara tidak langsung. Karena arsitektur 256K terbukti "), t("tahan", { b: true }), t(" terhadap perubahan komponen — aktivasi, residual, maupun expand semuanya tidak signifikan — kenaikan 1,00 pp itu "), t("tidak mungkin berasal dari kekhususan rancangan arsitektur", { b: true }), t(". Sisanya tinggal satu penjelasan: cara model dilatih, yaitu distilasi dari asisten.")]));
A(p([t("Ablasi juga memperkuat pola kapasitas yang jadi benang merah skripsi: pada 128K komponen tunggal berpengaruh besar dan distilasi gagal; pada 256K komponen tunggal tidak berpengaruh dan distilasi berhasil. Dua-duanya menunjuk hal yang sama — 128K kapasitasnya sudah habis untuk tugas dasarnya.")]));

/* ------------------------------------------------ 5. ISTILAH */
A(h1("5. Istilah yang Mungkin Ditanya Artinya"));
A(glos([
  ["Ablation study", "Melepas atau mengganti satu komponen model lalu melihat akurasinya berubah berapa. Seperti mencabut satu kabel untuk tahu kabel itu penting atau tidak."],
  ["Fungsi aktivasi", "Saklar di dalam jaringan yang menentukan sinyal mana diteruskan dan seberapa kuat."],
  ["SiLU vs ReLU", "ReLU memotong semua nilai negatif menjadi nol secara kaku. SiLU memotongnya melengkung dan halus, sehingga sinyal negatif kecil masih bisa lewat sedikit."],
  ["Dead neuron", "Neuron yang keluarannya selalu nol sehingga berhenti belajar. Risikonya lebih besar pada ReLU, dan mahal kalau neuronnya memang sedikit."],
  ["Expand ratio", "Seberapa lebar kanal dilebarkan sementara di dalam blok sebelum diciutkan lagi. Nilai 2,5 berarti dilebarkan 2,5 kali."],
  ["Koneksi residual", "Jalan pintas yang menambahkan masukan blok langsung ke keluarannya, supaya informasi tidak hilang dan pelatihan lebih stabil."],
  ["Classifier hidden", "Satu lapisan tambahan di bagian penentu kelas, sebelum lapisan keluaran akhir."],
  ["Bebas parameter", "Perubahan yang tidak mengubah jumlah bobot sama sekali, sehingga selisih akurasinya murni akibat komponen tersebut."],
  ["Confound", "Dua sebab tercampur dalam satu perubahan, sehingga tidak bisa dipastikan yang mana penyebabnya. Di sini: hilangnya komponen bercampur dengan berkurangnya kapasitas."],
  ["Poin persentase (pp)", "Selisih langsung antar-persentase. Dari 76,9% ke 71,6% berarti turun 5,33 poin persentase."],
  ["Over-parameterized", "Punya lebih banyak parameter daripada yang sebenarnya dipakai untuk mencapai akurasi itu."],
  ["Daya uji (power)", "Kemampuan uji statistik menangkap efek yang benar-benar ada. Dengan lima seed, daya ujinya terbatas sehingga efek kecil bisa lolos tidak terdeteksi."],
]));

/* ------------------------------------------------ 6. TANYA JAWAB */
A(h1("6. Pertanyaan yang Mungkin Ditanya"));
const qa = [
  ["Jadi komponen mana yang paling berpengaruh?",
   "Fungsi aktivasi. Tetapi jawaban lengkapnya ada syaratnya: pengaruhnya besar dan signifikan hanya pada varian 128K, yaitu turun 5,33 poin persentase dengan p sebesar 0,004 dan kalah di kelima seed. Pada varian 256K komponen yang sama tidak berpengaruh signifikan. Jadi temuannya bukan sekadar “aktivasi paling penting”, melainkan “pentingnya komponen bergantung pada kapasitas model”."],
  ["Kenapa ReLU di 256K malah naik 1,79 pp? Berarti lebih bagus dong?",
   "Belum bisa dikatakan begitu. Nilai p-nya 0,268 dan hanya unggul di 3 dari 5 seed, dengan simpangan baku yang justru lebih besar dari baseline. Pola seperti itu ciri khas fluktuasi acak, bukan perbaikan nyata. Kalau saya klaim ReLU lebih baik, saya melanggar taraf signifikansi yang saya pakai sendiri di seluruh skripsi. Yang benar dikatakan: pada 256K perbedaan antar-aktivasi tidak nyata."],
  ["Akurasi ReLU 256K itu 0,7873, sama persis dengan hasil TA-KD Anda. Berarti cukup ganti aktivasi, tidak perlu distilasi?",
   "Angkanya kebetulan sama, tetapi statusnya jauh berbeda. Varian ReLU dibandingkan dengan baseline-nya hanya unggul 3 dari 5 seed dengan p sebesar 0,268 dan simpangan baku 0,0223 — tidak signifikan dan tidak stabil. Sedangkan TA-KD unggul 5 dari 5 seed terhadap kontrol berpasangannya dengan p sebesar 0,024 dan simpangan baku 0,0089 — signifikan dan paling stabil di antara seluruh skenario. Ditambah lagi keduanya tidak saling meniadakan: TA-KD mengubah cara melatih, bukan arsitekturnya, sehingga secara prinsip bisa digabung dan itu justru saran penelitian lanjutan."],
  ["Expand ratio dipangkas sampai 75 persen parameter hilang tapi akurasi tetap. Berarti model Anda kebanyakan parameter?",
   "Pada dimensi expand memang berlebih, dan saya tulis terbuka sebagai temuan, bukan saya sembunyikan. Ini justru sinyal efisiensi yang berguna: masih ada ruang memangkas model tanpa kehilangan akurasi. Saya tidak langsung mengganti model utama menjadi versi kecil itu karena seluruh rangkaian percobaan distilasi R1 sampai R9 sudah dijalankan pada dua kelas kapasitas yang ditetapkan di awal, yaitu 128K dan 256K. Mengganti arsitektur di tengah jalan akan membatalkan perbandingan yang sudah berpasangan. Pemanfaatan temuan ini saya tulis sebagai saran di BAB 5."],
  ["Kalau expand 1,0 pada 256K hanya butuh 64 ribu parameter dengan akurasi sama, kenapa 128K tidak ikut membaik?",
   "Karena dua hal itu berbeda. Menurunkan expand memangkas parameter di dimensi yang memang berlebih, tetapi tidak menambah kemampuan model. Yang membatasi 128K bukan dimensi expand, melainkan kapasitas keseluruhannya yang sudah terpakai habis untuk mempelajari tugas dasarnya — terbukti dari fakta bahwa di 128K justru komponen bebas parameter seperti aktivasi yang berdampak besar."],
  ["Kenapa hanya paired t-test, tidak McNemar seperti di bagian lain?",
   "Karena pertanyaannya beda tingkat. McNemar membandingkan dua model pada sampel uji yang sama untuk melihat gambar mana berpindah benar-salah, dan itu tepat untuk membandingkan dua cara melatih model yang arsitekturnya sama. Ablasi membandingkan arsitektur yang berbeda, sehingga yang relevan adalah selisih akurasi berpasangan antar-seed. Wilcoxon juga saya hitung dan untuk varian ReLU 128K nilainya 0,062, yaitu nilai terkecil yang mungkin dihasilkan uji itu pada lima seed, sehingga arahnya sejalan."],
  ["Kenapa hanya lima seed?",
   "Konsisten dengan seluruh skripsi, yang memakai lima seed dengan pembagian data berbeda tiap seed sebagai validasi silang berulang. Konsekuensinya daya uji terbatas, dan itu saya tulis terbuka: efek kecil seperti selisih 1,79 poin persentase pada 256K bisa saja nyata tetapi belum cukup bukti. Efek besar seperti 5,33 poin persentase tetap tertangkap dengan jelas."],
  ["Kenapa ablasi hanya pada enam kelas, tidak pada lima kelas?",
   "Arsitektur WasteNet identik pada kedua skema label dan hanya berbeda pada jumlah neuron keluarannya. Kepentingan relatif antarkomponen tidak berubah oleh jumlah kelas, sehingga menjalankan 40 run tambahan tidak menambah informasi apa pun."],
  ["Kenapa koneksi residual tidak diablasi pada 128K?",
   "Karena pada konfigurasi 128K setiap blok mengalami perubahan ukuran atau jumlah kanal, sehingga tidak ada blok dengan dimensi masuk dan keluar yang sama. Syarat koneksi residual tidak pernah terpenuhi, jadi koneksi itu memang tidak pernah aktif dan mengablasinya tidak mengubah apa pun. Saya sudah memverifikasi ini dengan menghitung jumlah koneksi residual aktif: nol pada 128K dan satu pada 256K."],
  ["Kenapa baseline ablasi berbeda dari baseline pada Tabel 4.1?",
   "Karena baseline sengaja dilatih ulang di dalam notebook ablasi supaya pipeline dan pengaturannya terbukti identik dengan varian yang dibandingkan. Selisihnya kurang dari satu poin persentase dan masih berada dalam rentang sebaran antar-seed. Justru karena itu setiap varian selalu dibandingkan terhadap baseline internal ablasi, bukan terhadap angka di Tabel 4.1, sehingga perbandingannya tetap adil."],
  ["Mengganti aktivasi dan memangkas expand itu kan bukan “parameter”. Bukankah pertanyaan saya soal parameter?",
   "Betul, dan saya memahami maksudnya. Parameter dalam arti bobot yang dipelajari itu jumlahnya ratusan ribu, sehingga tidak mungkin diuji satu per satu dan tidak akan bermakna. Yang lazim dilakukan dan yang informatif adalah menguji pada tingkat komponen rancangan, karena setiap komponen mewakili sekelompok parameter beserta perannya. Karena itu ablasi ini menguji empat komponen, dan untuk yang mengubah jumlah parameter saya cantumkan angka parameternya agar tetap terhubung dengan pertanyaan awal."],
  ["Apa gunanya ablasi ini untuk kesimpulan skripsi Anda?",
   "Dua hal. Pertama, ia menutup penjelasan alternatif atas temuan utama: karena arsitektur 256K terbukti tahan terhadap perubahan komponennya, kenaikan 1,00 poin persentase dari TA-KD tidak dapat dijelaskan oleh kekhususan rancangan arsitektur, melainkan berasal dari cara model dilatih. Kedua, ia memperkuat tema kapasitas yang menjadi benang merah skripsi, yaitu bahwa perilaku model berubah tergantung apakah kapasitasnya sempit atau longgar."],
  ["Kenapa hasilnya banyak yang tidak signifikan?",
   "Itu juga informasi. Hasil tidak signifikan pada 256K berarti arsitekturnya tidak bergantung pada satu komponen tertentu, dan justru itu yang membuat argumen saya kuat: perbaikan yang saya laporkan bukan efek keberuntungan rancangan. Ablasi yang semuanya berpengaruh besar malah akan membuat temuan distilasi saya sulit dipisahkan dari pengaruh arsitektur."],
  ["Apa kelemahan ablasi ini?",
   "Tiga hal saya akui terbuka. Lima seed membatasi daya uji sehingga efek kecil bisa lolos. Dua dari empat komponen, yaitu expand ratio dan hidden layer, mengubah jumlah parameter sehingga efeknya bercampur dengan berkurangnya kapasitas — karena itu jumlah parameter saya cantumkan. Dan komponen diuji satu per satu, sehingga kemungkinan interaksi antarkomponen belum tercakup."],
];
qa.forEach(([q, a]) => {
  A(p([t("T: ", { b: true, c: "1F3864" }), t(q, { b: true })], { before: 130, after: 40 }));
  A(p([t("J: ", { b: true, c: "1F3864" }), t(a)], { after: 60 }));
});

/* ------------------------------------------------ 7. JANGAN */
A(h1("7. Empat Kalimat yang Jangan Sampai Terucap"));
A(tbl([4400, 4620], [
  ["Jangan bilang", "Bilangnya begini"],
  ["“ReLU lebih baik untuk 256K.”", "“Pada 256K perbedaan antar-aktivasi tidak nyata secara statistik.”"],
  ["“Expand ratio tidak berguna.”", "“Kapasitas pada dimensi expand cenderung berlebih, jadi masih ada ruang penghematan.”"],
  ["“Ablasi membuktikan TA-KD berhasil.”", "“Ablasi menutup penjelasan alternatif dari sisi arsitektur, sehingga mendukung secara tidak langsung.”"],
  ["“Residual tidak penting.”", "“Pada 256K efek residual tidak signifikan; pada 128K residual memang tidak pernah aktif sehingga tidak diuji.”"],
]));

A(p([t("— kuasai tiga kalimat di bagian 1, sisanya tinggal mendukung —", { i: true, size: 18, c: "666666" })], { align: AlignmentType.CENTER, before: 240 }));

const doc = new Document({
  styles: { default: { document: { run: { font: F, size: 21 } } } },
  sections: [{ properties: { page: { margin: { top: 1100, bottom: 1100, left: 1100, right: 1100 } } }, children: kids }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.argv[2], b); console.log("WROTE " + process.argv[2] + " (" + b.length + " bytes)"); });
