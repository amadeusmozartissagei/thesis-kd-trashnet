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
// istilah: [term, plain explanation]
function glos(items) {
  return tbl([2400, 6620], items.map(([a, b]) => [[t(a, { b: true })], [t(b)]]), { plain: true });
}

const kids = [];
const A = (...x) => kids.push(...x);

A(p([t("Cheat Sheet Skripsi", { b: true, size: 34 })], { align: AlignmentType.CENTER, after: 40 }));
A(p([t("Knowledge Distillation Berbasis Teacher-Assistant untuk Model Ringan WasteNet pada Klasifikasi Citra Sampah TrashNet", { b: true, size: 20 })], { align: AlignmentType.CENTER, after: 40 }));
A(p([t("Hamza Pratama · 4611422068 · disusun per subbab supaya tetap cocok walau nomor halaman bergeser", { i: true, size: 18, c: "666666" })], { align: AlignmentType.CENTER, after: 200 }));

/* ------------------------------------------------ 1. INTI */
A(h1("1. Inti Skripsi dalam 30 Detik"));
A(p(t("Model kecil susah pintar. Cara umum membuatnya lebih pintar adalah knowledge distillation: model besar yang sudah pintar “mengajari” model kecil. Masalahnya, kalau gurunya terlalu jauh lebih pintar, ilmunya malah tidak nyantol — ini disebut capacity gap.")));
A(p(t("Penelitian ini menguji tiga jenis guru pada model super-kecil WasteNet: guru raksasa (EfficientNet-B4), asisten menengah (Focus-RCNet), dan guru diri sendiri (self-distillation). Hasilnya: hanya asisten menengah yang berhasil, dan hanya pada student 256K.")));
A(p([t("Kalimat kunci: ", { b: true }), t("guru terbaik bukan yang paling pintar, tapi yang jaraknya pas — dan muridnya juga harus cukup besar untuk menampung ilmunya.", { b: true })]));

/* ------------------------------------------------ 2. ANGKA */
A(h1("2. Angka yang Wajib Hafal"));
const L = (...xs) => xs.map((s) => t(s));
A(tbl([3400, 2000, 3620], [
  ["Hal", "Angka", "Catatan kalau ditanya"],
  ["Hasil utama (TA-KD 256K)", L("+1,00 pp (K6)", "+1,40 pp (K5)"), "Menang di 5 dari 5 seed"],
  ["Nilai p uji t", L("0,024 (K6)", "0,045 (K5)"), "Signifikan, di bawah 0,05"],
  ["Akurasi model terbaik", L("0,7873 (K6)", "0,8235 (K5)"), "WasteNet-256K dengan TA-KD"],
  ["Ukuran model", L("256.002 parameter", "≈ 84 juta FLOPs"), "128K: 128.086 par, ≈ 51 juta FLOPs"],
  ["Guru besar", "0,9446", "EfficientNet-B4, 17,56 juta parameter"],
  ["Asisten", "0,8507", "Focus-RCNet, 520,6 ribu parameter"],
  ["Dataset", L("2.521 citra", "6 kelas"), "Latih 1.764 / validasi 378 / uji 379"],
  ["Protokol", "5 seed, 70:15:15", "Split berbeda tiap seed"],
  ["Ablasi", "40 kali pelatihan", "SiLU→ReLU di 128K turun 5,33 pp (p=0,004)"],
  ["Lingkungan komputasi", L("Kaggle Notebook", "1× NVIDIA Tesla T4"), "Python + PyTorch, 100 epoch, mixed precision"],
], { centre: false }));

/* ------------------------------------------------ 3. KAMUS */
A(new Paragraph({ children: [new PageBreak()] }));
A(h1("3. Kamus Istilah Sulit (bahasa sehari-hari)"));

A(h2("Konsep inti penelitian"));
A(glos([
  ["Knowledge distillation", "Melatih model kecil dengan meniru cara berpikir model besar, bukan sekadar meniru jawaban benarnya. Seperti murid yang belajar dari cara guru menjelaskan, bukan cuma dari kunci jawaban."],
  ["Teacher / Student", "Teacher = model besar yang sudah pintar. Student = model kecil yang sedang dilatih."],
  ["Hard label", "Jawaban pasti dari kunci jawaban: “ini kaca”. Cuma benar atau salah, tidak ada abu-abu."],
  ["Soft label", "Tingkat keyakinan guru: “90% kaca, 7% plastik, 3% logam”. Angka 7% dan 3% itu justru yang berharga."],
  ["Dark knowledge", "Informasi tersembunyi di dalam soft label tentang kelas mana yang saling mirip. Guru tidak cuma bilang “ini kaca”, tapi juga menyiratkan “kaca ini agak mirip plastik”."],
  ["Capacity gap", "Jarak kepintaran antara guru dan murid. Kalau terlalu jauh, penjelasan guru terlalu rumit untuk dicerna — seperti profesor mengajar anak SD."],
  ["Teacher assistant (TA-KD)", "Menaruh model berkemampuan menengah di antara guru besar dan murid, supaya ilmunya diterjemahkan dulu ke tingkat yang bisa dicerna."],
  ["Self-distillation / Born-Again Networks", "Murid belajar dari dirinya sendiri versi sebelumnya, seperti belajar dari catatan sendiri semester lalu."],
  ["Temperature (T)", "Pengatur seberapa “dihaluskan” keyakinan guru. Makin besar T, makin terlihat kemiripan antarkelas, tidak cuma jawaban juaranya."],
  ["Alpha (α)", "Pengatur porsi belajar: berapa banyak dari kunci jawaban, berapa banyak dari guru. α = 0,5 berarti seimbang."],
  ["Logits", "Angka mentah keluaran model sebelum diubah menjadi persentase keyakinan."],
]));

A(h2("Arsitektur model"));
A(glos([
  ["Parameter", "Angka-angka yang dipelajari model selama latihan (bobot, bias, dan parameter batch normalization). Makin banyak, model makin besar dan makin butuh memori."],
  ["FLOPs", "Banyaknya operasi hitung untuk menebak satu gambar. Makin kecil, makin ringan dan cepat dijalankan."],
  ["Depthwise-separable convolution", "Memecah satu operasi berat menjadi dua operasi ringan yang hasilnya mirip. Trik hemat khas model ponsel."],
  ["Inverted residual", "Pola blok yang melebarkan jumlah kanal dulu, memprosesnya, lalu menciutkannya lagi."],
  ["Expand ratio", "Seberapa lebar kanal dilebarkan di dalam blok. Nilai 2,0 berarti dilebarkan dua kali lipat."],
  ["Koneksi residual", "Jalan pintas yang menambahkan input langsung ke output blok, supaya informasi tidak hilang dan latihan lebih stabil."],
  ["SiLU dan ReLU", "Fungsi aktivasi, semacam saklar yang menentukan sinyal mana diteruskan. ReLU memotong semua nilai negatif jadi nol; SiLU memotongnya lebih halus."],
  ["Batch normalization", "Menyetel ulang skala angka di tengah jaringan supaya proses latihan stabil."],
  ["Global average pooling", "Meringkas seluruh peta fitur menjadi satu angka per kanal sebelum lapisan penentu kelas."],
]));

A(h2("Data dan pelatihan"));
A(glos([
  ["Epoch", "Satu kali putaran model melihat seluruh data latih."],
  ["Seed", "Angka awal pengacak. Seed berbeda menghasilkan pembagian data dan kondisi awal berbeda. Dipakai 5 seed supaya hasil tidak bergantung satu keberuntungan."],
  ["Stratified split", "Membagi data dengan menjaga proporsi tiap kelas tetap sama di bagian latih, validasi, dan uji."],
  ["Data augmentation", "Memperbanyak variasi data latih dengan memutar, membalik, atau mengubah warna gambar."],
  ["Overfitting", "Model hafal data latih tetapi payah pada data baru."],
  ["Checkpoint", "Simpanan kondisi model pada titik terbaiknya selama latihan."],
  ["Cosine annealing", "Cara menurunkan kecepatan belajar secara perlahan mengikuti bentuk kurva, supaya akhir latihan lebih halus."],
]));

A(h2("Evaluasi dan statistik"));
A(glos([
  ["Confusion matrix", "Tabel yang menunjukkan model menebak apa untuk kelas apa, sehingga kelihatan salahnya di mana."],
  ["Accuracy", "Persentase tebakan benar dari seluruh data uji."],
  ["Precision", "Dari semua yang ditebak “kaca”, berapa persen yang benar-benar kaca."],
  ["Recall", "Dari semua kaca yang sebenarnya ada, berapa persen yang berhasil ditemukan."],
  ["F1-score", "Gabungan seimbang antara precision dan recall menjadi satu angka."],
  ["Macro F1", "Rata-rata F1 semua kelas dengan bobot sama, supaya kelas kecil seperti trash tidak tenggelam oleh kelas besar."],
  ["Simpangan baku", "Ukuran seberapa berbeda-beda hasil antar-seed. Kecil berarti stabil."],
  ["Poin persentase (pp)", "Selisih langsung antar-persentase. Dari 76% ke 78% berarti naik 2 poin persentase."],
  ["p-value", "Peluang hasil seperti itu muncul kebetulan padahal sebenarnya tidak ada efek. Makin kecil makin meyakinkan; batas umum 0,05."],
  ["Signifikan", "Nilai p di bawah 0,05, artinya perbedaannya kecil kemungkinannya hanya kebetulan."],
  ["Paired t-test", "Membandingkan dua metode pada pasangan yang sama (seed yang sama), supaya perbedaan pembagian data tidak mengaburkan hasil."],
  ["Wilcoxon signed-rank", "Versi uji berpasangan yang tidak mengandaikan data berdistribusi normal."],
  ["McNemar", "Uji yang melihat per gambar: berapa yang tadinya salah menjadi benar, dan sebaliknya."],
  ["Kontrol berpasangan", "Skenario pembanding yang identik persis kecuali satu hal yang sedang diuji."],
  ["Ablation study", "Melepas atau mengganti satu komponen model untuk melihat seberapa penting komponen itu."],
  ["Baseline", "Titik acuan, yaitu model tanpa perlakuan yang sedang diuji."],
  ["Over-parameterized", "Punya lebih banyak parameter daripada yang sebenarnya dibutuhkan."],
]));

/* ------------------------------------------------ 4. PETA */
A(new Paragraph({ children: [new PageBreak()] }));
A(h1("4. Peta Isi per Subbab"));
A(p([t("Nomor halaman sengaja tidak dicantumkan karena bisa bergeser tiap revisi. Lihat DAFTAR ISI di skripsi setelah field diperbarui.", { i: true, size: 18, c: "666666" })]));

const bab = (judul, items) => {
  A(h2(judul));
  items.forEach(([no, isi]) => A(bul([t(no + "  ", { b: true }), t(isi)])));
};

bab("BAB 1 — Pendahuluan", [
  ["1.1 Latar Belakang", "Sampah perlu dipilah otomatis; modelnya harus ringan agar bisa jalan di perangkat kecil, tapi model ringan sulit akurat. KD jadi solusi, namun tidak selalu berhasil karena capacity gap."],
  ["1.2 Batasan Masalah", "Dibatasi pada dataset TrashNet, student WasteNet 128K & 256K, guru EfficientNet-B4 dan asisten Focus-RCNet."],
  ["1.3 Rumusan Masalah", "Tiga hal: cara membangun WasteNet dengan KD, pengaruh kapasitas guru, dan pembuktian statistiknya."],
  ["1.4 Tujuan", "Cerminan dari tiga rumusan masalah di atas."],
  ["1.6 Kebaruan", "Belum ada yang menguji pengaruh kapasitas guru pada student sekecil 128–256 ribu parameter untuk citra sampah."],
]);
bab("BAB 2 — Kajian Pustaka", [
  ["2.1 Tinjauan Pustaka", "Lima penelitian acuan: Focus-RCNet, Hinton (KD), Cho & Hariharan (capacity gap), Mirzadeh (TA-KD), Furlanello (BAN)."],
  ["2.2.6 Knowledge Distillation", "Rumus gabungan: sebagian belajar dari label asli, sebagian dari soft label guru, diatur T dan α."],
  ["2.2.7 TA-KD", "Dasar teori penyisipan asisten menengah."],
  ["2.2.8 Self-Distillation", "Dasar teori Born-Again Networks; manfaatnya sebagian dari efek regularisasi."],
  ["2.2.11 Dataset TrashNet", "2.527 citra 6 kelas; kelas trash paling sedikit dan paling heterogen."],
  ["2.2.12 Evaluasi", "Accuracy, macro F1, jumlah parameter, dan FLOPs."],
  ["2.2.13 Uji Statistik", "Dasar teori uji t berpasangan, Wilcoxon, dan McNemar."],
]);
bab("BAB 3 — Metode Penelitian", [
  ["3.7.3 Data Splitting", "70:15:15 terstratifikasi, berbeda tiap seed. Inilah yang diklaim sebagai validasi silang berulang, pengganti k-fold."],
  ["3.7.5 Arsitektur Model", "WasteNet: depthwise-separable + inverted residual, kernel 3×3 dan 1×1."],
  ["3.7.7 Training Model", "SGD momentum 0,9, 100 epoch, cosine annealing, checkpoint terbaik dari data validasi. T dan α dipilih lewat uji pendahuluan seed 42 lalu dibekukan."],
  ["3.7.9 Uji Statistik", "Uji t berpasangan + Wilcoxon (antar-seed) dan McNemar (per sampel), taraf 0,05."],
  ["3.7.10 Skenario Pengujian", "R1–R9 skenario distilasi, R10–R11 ablasi arsitektur. Tabel 3.3 memuat nilai T, α, dan laju pembelajaran."],
]);
bab("BAB 4 — Hasil dan Pembahasan", [
  ["4.1 Protokol Data", "1.764 / 378 / 379 citra; partisi sama tiap seed sehingga perbandingan bisa berpasangan."],
  ["4.2 Guru, Asisten, Baseline", "Guru 0,9446 · asisten 0,8507 · student CE 0,7652 (128K) dan 0,7784 (256K). Cara melatih asisten tidak berpengaruh signifikan."],
  ["4.3.1 Distilasi Langsung", "Dari guru besar: tidak membantu di kedua ukuran student."],
  ["4.3.2 TA-KD", "Berhasil di 256K (akurasi tertinggi), gagal di 128K."],
  ["4.3.3 Self-Distillation", "Naik sedikit tapi tidak signifikan; variasi antar-seed mengecil (efek regularisasi)."],
  ["4.4 Uji Signifikansi", "Hanya TA-KD 256K yang signifikan. McNemar: K5 signifikan (p=0,014), K6 mendekati (p=0,085)."],
  ["4.5 Analisis per Kelas", "Perbaikan terkonsentrasi di glass, metal, plastic — kelas yang saling mirip. Bukti dark knowledge."],
  ["4.6 K5 vs K6", "K5 selalu lebih tinggi; kelas trash menyeret K6. Temuan utama konsisten di keduanya."],
  ["4.7 Efisiensi", "TA-KD menaikkan akurasi tanpa menambah satu pun parameter atau FLOPs."],
  ["4.8 Ablasi", "Aktivasi paling menentukan, dan hanya di 128K. 256K tahan terhadap ablasi. Expand ratio berlebih."],
  ["4.9 Perbandingan", "Akurasi di bawah Focus-RCNet (0,92), tapi model jauh lebih kecil dan protokol evaluasinya berbeda."],
  ["4.10 Pembahasan", "Sintesis: kapasitas guru yang menentukan, bukan jadwal pelatihan; kapasitas student juga harus cukup. Ditutup daftar keterbatasan: data kecil, lima seed, perbaikan sekitar satu poin persentase, dan hyperparameter lengan 128K tidak sepadan dengan 256K."],
]);
bab("BAB 5 — Penutup", [
  ["5.1 Simpulan", "Empat butir yang menjawab tiga rumusan masalah plus temuan ablasi."],
  ["5.2 Saran", "Uji di perangkat nyata, perbesar data dan seed, telusuri rentang kapasitas asisten, uji lintas arsitektur, manfaatkan temuan expand."],
]);

/* ------------------------------------------------ 5. TANYA JAWAB */
A(h1("5. Pertanyaan yang Mungkin Ditanya"));
const qa = [
  ["Kenapa akurasinya cuma 78%, kalah dari Focus-RCNet yang 92%?",
   "Kelas kapasitasnya beda jauh: model saya 256 ribu parameter, mereka 526 ribu, dan FLOPs saya 5 kali lebih kecil. Protokolnya juga beda — mereka memperbesar data lewat augmentasi dan melaporkan satu angka, saya pakai data asli dan rata-rata 5 seed. Buktinya, saat saya melatih ulang Focus-RCNet dengan protokol saya hasilnya 0,8507, bukan 0,92. Jadi selisihnya dari protokol, bukan dari implementasi."],
  ["Kenapa tidak pakai k-fold cross validation?",
   "Protokol saya sudah setara: 5 seed dengan pembagian data berbeda tiap seed, yang dikenal sebagai validasi silang berulang. Dietterich (1998) juga memperingatkan k-fold dengan uji t punya risiko kesalahan tipe I yang tinggi. Ini sudah disetujui pembimbing."],
  ["Wilcoxon-nya 0,062, berarti tidak signifikan dong?",
   "Dengan 5 seed, nilai p dua sisi terkecil yang mungkin dihasilkan Wilcoxon memang 0,0625. Jadi 0,062 itu hasil terbaik yang bisa dicapai uji ini, dan muncul justru karena menang di kelima seed. Uji ini saya pakai untuk memastikan arah dan konsistensi, sedangkan penentuan signifikansi bersandar pada uji t dan McNemar."],
  ["WasteNet itu kan sudah ada, kenapa pakai nama itu?",
   "Namanya kebetulan sama. WasteNet milik White dkk. (2020) berbasis DenseNet dan berukuran puluhan juta parameter. Milik saya disusun dari blok depthwise-separable dengan inverted residual, hanya 128–256 ribu parameter. Perbedaan ini saya tulis eksplisit di BAB 2."],
  ["Kenapa banyak hasil yang tidak signifikan?",
   "Itu justru temuannya. Kalau semua skema berhasil, tidak ada yang bisa disimpulkan tentang pengaruh kapasitas guru. Yang gagal dan yang berhasil sama-sama membentuk pola: hanya guru berkapasitas menengah yang membantu."],
  ["Kenapa TA-KD berhasil di 256K tapi gagal di 128K?",
   "Student juga perlu ruang untuk menampung ilmu tambahan. Pada 128K kapasitasnya sudah habis untuk mempelajari tugas dasarnya, sehingga tidak tersisa ruang untuk memanfaatkan informasi dari guru."],
  ["Asisten Anda dilatih cross-entropy, tidak didistilasi dari EfficientNet-B4 seperti Mirzadeh. Apakah ini masih TA-KD?",
   "Yang saya uji adalah hipotesis capacity gap dari Mirzadeh dkk., yaitu student kecil lebih terbantu oleh guru berkapasitas menengah daripada guru raksasa. Itu diuji langsung dengan membandingkan asisten 520 ribu parameter melawan EfficientNet-B4 17,56 juta pada student yang sama. Rantai berjenjangnya sendiri tetap saya uji sebagai ablasi: Focus-RCNet saya distilasi dari EfficientNet-B4, baik satu tahap maupun dua tahap, dan hasilnya tidak signifikan (p = 0,62 dan p = 0,77, Tabel 4.2). Karena setara, saya memakai asisten yang dilatih cross-entropy sebab lebih sederhana. Hal ini juga terlihat pada Gambar 2.3 panel (b): panah dari guru besar ke asisten sengaja digambar putus-putus karena hanya diuji sebagai ablasi."],
  ["Kenapa laju pembelajaran lengan 128K berbeda dari lengan 256K?",
   "Laju pembelajaran dan resolusi masukan asisten pada setiap lengan ditentukan lewat uji pendahuluan pada data validasi. Untuk varian 128K nilai yang terpilih mengikuti hasil uji pendahuluan skema lima kelas, sehingga lengan 128K dan 256K memang tidak sepadan. Saya menuliskannya terbuka sebagai keterbatasan di BAB 4: simpulan bahwa distilasi tidak membantu 128K belum dapat dipisahkan sepenuhnya dari pengaruh pemilihan hyperparameter tersebut. Temuan utama pada 256K tidak terpengaruh, sebab lengan itu memakai laju pembelajaran yang sama persis dengan kontrolnya."],
  ["Dari mana nilai T = 4 dan α = 0,5?",
   "Dari uji pendahuluan memakai seed 42 pada data validasi saja. Kombinasi terbaik lalu dibekukan dan dipakai seragam di lima seed, sehingga data uji tidak pernah ikut menentukan parameter. Nilainya ada di Tabel 3.3."],
  ["Angka ResNet50, DenseNet121 itu Anda yang menjalankan?",
   "Bukan. Itu kutipan dari Tabel 5 penelitian Zheng dkk. (2023) yang menguji arsitektur tersebut pada TrashNet. Di tabel saya ada kolom Sumber yang memisahkan mana kutipan dan mana hasil saya sendiri."],
  ["Kenapa ablasi hanya di 6 kelas, tidak di 5 kelas?",
   "Arsitektur WasteNet identik pada kedua skema label, hanya berbeda jumlah neuron keluarannya. Jadi kepentingan relatif antarkomponen tidak berubah, dan menjalankan keduanya tidak menambah informasi."],
  ["Kenapa koneksi residual tidak diablasi pada 128K?",
   "Karena pada konfigurasi 128K tidak ada blok yang dimensi masuk dan keluarnya sama, sehingga koneksi residual memang tidak pernah aktif. Mengablasinya tidak akan mengubah apa pun."],
  ["Apa kelemahan penelitian ini?",
   "Datanya kecil, hanya 2.521 citra, dan seed-nya hanya lima sehingga daya uji statistiknya terbatas. Besar perbaikannya juga sekitar satu poin persentase, jadi penerapannya perlu mempertimbangkan kebutuhan ketelitian. Selain itu hyperparameter lengan 128K mengikuti hasil uji pendahuluan skema lima kelas sehingga tidak sepadan dengan lengan 256K. Semua ini sudah saya tulis terbuka di BAB 4 dan BAB 5."],
  ["Apa kontribusi utamanya?",
   "Menunjukkan bahwa pada student ultra-ringan, pemilihan guru menentukan keberhasilan distilasi, dan guru terbaik adalah yang berkapasitas menengah — bukan yang terbesar maupun diri sendiri. Perbaikannya diperoleh tanpa menambah satu pun parameter."],
];
qa.forEach(([q, a]) => {
  A(p([t("T: ", { b: true, c: "1F3864" }), t(q, { b: true })], { before: 130, after: 40 }));
  A(p([t("J: ", { b: true, c: "1F3864" }), t(a)], { after: 60 }));
});

A(p([t("— semoga lancar sidangnya —", { i: true, size: 18, c: "666666" })], { align: AlignmentType.CENTER, before: 240 }));

const doc = new Document({
  styles: { default: { document: { run: { font: F, size: 21 } } } },
  sections: [{ properties: { page: { margin: { top: 1100, bottom: 1100, left: 1100, right: 1100 } } }, children: kids }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.argv[2], b); console.log("WROTE " + process.argv[2] + " (" + b.length + " bytes)"); });
