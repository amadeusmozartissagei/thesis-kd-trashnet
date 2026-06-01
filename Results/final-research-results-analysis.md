# Analisis Hasil Akhir Penelitian Focus-RCNet dan EfficientNet-Lite0 pada TrashNet

## Ringkasan Eksekutif

Penelitian ini mereproduksi Focus-RCNet pada dataset TrashNet, memverifikasi
knowledge distillation (KD) dari EfficientNet-B4 ke Focus-RCNet, kemudian
membandingkannya secara terkontrol dengan EfficientNet-Lite0. Sebuah follow-up
terpisah juga dijalankan untuk menguji EfficientNet-Lite0 pada profil native yang
lebih sesuai untuk deployment.

Temuan utama:

1. EfficientNet-B4 teacher berhasil dilatih dengan validation accuracy `95.52%`.
2. Pada controlled comparison `380x380`, Focus-RCNet Two-Stage KD adalah student
   terbaik: accuracy `86.30%`, `520,630` parameter, dan `0.526` GFLOPs.
3. EfficientNet-Lite0 controlled memperoleh peningkatan dari KD, tetapi tetap
   tertinggal dari Focus-RCNet: Two-Stage KD mencapai `79.58%`.
4. Pada optional Native follow-up, EfficientNet-Lite0 Two-Stage KD mencapai
   accuracy student tertinggi, yaitu `90.91%`, dengan `0.768` GFLOPs.
5. EfficientNet-Lite0 Native Direct KD turun menjadi `86.82%`. Penurunan ini
   konsisten dengan negative transfer ketika KD diterapkan langsung dari
   ImageNet initialization, bukan indikasi bug implementasi.
6. Hasil Native harus dilaporkan terpisah dari controlled comparison karena
   memakai student input `224x224` dan pretrained ImageNet weights.

## Sumber Hasil

Dokumen ini dirangkum dari artifact notebook berikut:

- [Notebook 1: Teacher Training](notebook1_teacher_training.ipynb)
- [Notebook 2: Phase 1 Focus-RCNet](notebook2-phase1-focus-rcnet.ipynb)
- [Notebook 3: Phase 2 EfficientNet-Lite0 Controlled](notebook3-phase2-efficientnet-lite0.ipynb)
- [Notebook 3B: EfficientNet-Lite0 Native Follow-Up](notebook3b-lite0-native-followup.ipynb)
- [Experimental Protocol](../Resource/project-resource-experimental-protocol-spec.md)
- [PDF paper Focus-RCNet](../Resource/Focus-RCNet%20a%20lightweight%20recyclable%20waste%20classification%20algorithm%20based%20on%20focus%20and%20knowledge%20distillation.pdf)
- [Official Springer article](https://link.springer.com/article/10.1186/s42492-023-00146-3)

## Batas Interpretasi Eksperimen

Penelitian ini memiliki dua track yang berbeda. Pemisahan ini penting agar
kesimpulan tidak melebih-lebihkan hasil.

### Track Utama: Controlled Comparison

Track utama digunakan untuk membandingkan arsitektur student secara fair.
Focus-RCNet dan EfficientNet-Lite0 memakai kondisi yang sama:

| Komponen | Nilai |
|----------|-------|
| Dataset mirror | TrashNet, `2527` gambar |
| Split | Train `1768`, validation `759` |
| Split source | Indeks checkpoint teacher, sama untuk seluruh eksperimen |
| Input | `380x380` |
| Student initialization | Random initialization |
| Optimizer | SGD, momentum `0.9`, weight decay `1e-4` |
| Learning rate | `0.05` |
| Epoch | `100` |
| Batch size | `8` |
| Scheduler | CosineAnnealingLR |
| KD temperature | `T=4` |
| KD alpha | `0.5` |
| Two-Stage KD | `50` epoch, LR `0.005` |

### Track Tambahan: Lite0 Native Profile

Track Native bersifat deployment-oriented. Track ini tidak digunakan untuk
menggantikan atau membatalkan controlled comparison.

| Komponen | Nilai |
|----------|-------|
| Student | EfficientNet-Lite0 pretrained ImageNet |
| Student input | `224x224` |
| Teacher | EfficientNet-B4 checkpoint Notebook 1 |
| Teacher input selama KD | `380x380` |
| Dual-resolution KD | Satu gambar augmentasi menghasilkan teacher view `380x380` dan student view `224x224` |
| Learning rate | `0.01` |
| Epoch | `100` |
| Batch size | `8` |
| Two-Stage KD | `50` epoch, LR `0.001` |

## Referensi Paper Asli

Paper asli oleh Zheng et al. melaporkan Focus-RCNet sebagai model lightweight
untuk klasifikasi sampah daur ulang. Teacher yang digunakan adalah
EfficientNet-B4. Paper menjelaskan penggunaan soft target dengan temperature
scaling dan kombinasi soft loss serta hard loss.

Hasil utama paper:

| Model pada Paper | Accuracy | FLOPs | Parameters |
|------------------|----------|-------|------------|
| EfficientNet-B4 teacher | 97% | 4.490 G | 17.559 M |
| Focus-RCNet baseline | 90% | 0.4188 G | 0.525802 M |
| Focus-RCNet-KD | 92% | 0.4188 G | 0.525802 M |

Paper menyatakan KD meningkatkan accuracy Focus-RCNet dari `90%` menjadi `92%`,
atau `+2.00` percentage points. Paper juga menggunakan `200` epoch, batch size
`16`, LR awal `0.05`, momentum `0.9`, dan weight decay `1e-4`.

### Ambiguity pada Pelaporan Paper

Paper memiliki beberapa detail yang perlu dibaca dengan hati-hati ketika hasil
reproduksi dibandingkan secara numerik:

| Bagian Paper | Angka atau Deskripsi | Catatan Interpretasi |
|--------------|----------------------|----------------------|
| Narasi KD dan Table 5 | Focus-RCNet `90%`, Focus-RCNet-KD `92%` | Digunakan sebagai pembanding utama reproduksi KD |
| Table 4 ablation | Baseline `88.07%`, +Focus `91.20%`, +Focus+SimAM `92.20%` | Table ablation melaporkan angka yang tidak identik dengan baseline `90%` pada Table 5 |
| Dataset section | Total `2528` gambar | Mirror Kaggle reproduksi memiliki `2527` gambar |
| Preprocessing section | Resize `224x224`, kemudian crop `380x380` | Urutan preprocessing tidak sepenuhnya jelas |
| Scheduler section | Penurunan LR tiap 90 cycle dan cosine annealing | Kombinasi scheduler tidak dijelaskan secara lengkap |

Karena itu, pembanding paper digunakan sebagai target referensi, bukan sebagai
jaminan bahwa seluruh kondisi reproduksi dapat disamakan secara bit-for-bit.

Catatan penting:

- Paper menyebut dataset berisi `2528` gambar, termasuk `483` gambar plastic.
  Mirror Kaggle yang digunakan dalam reproduksi berisi `2527` gambar dengan
  `482` gambar plastic.
- Paper tidak menjelaskan nilai temperature `T` dan bobot alpha KD secara
  eksplisit. Penelitian ini menggunakan asumsi reproduksi `T=4` dan
  `alpha=0.5`.
- Paper menjelaskan temperature scaling dan penggabungan soft serta hard loss.
  Implementasi penelitian ini menggunakan KL divergence `batchmean`, faktor
  `T^2`, FP32 untuk soft-target operations, `teacher.eval()`, frozen teacher
  parameters, dan `torch.no_grad()`.
- Nilai FLOPs paper dan reproduksi perlu dibandingkan dengan hati-hati karena
  metode penghitungan FLOPs tidak dijelaskan secara identik. Reproduksi ini
  menghitung convolution dan linear multiply-add sebagai dua FLOPs.

## Notebook 1: EfficientNet-B4 Teacher

EfficientNet-B4 dilatih sebagai teacher dan checkpoint terbaik digunakan kembali
pada seluruh eksperimen KD.

| Metric | Hasil |
|--------|-------|
| Best validation accuracy | `95.52%` |
| Best epoch | `61` |
| Parameters | `17,559,374` |
| Controlled GFLOPs | `8.784` |
| Controlled latency | `20.291 ms` |

Teacher reproduction berada `1.48` percentage points di bawah angka teacher
paper (`97%`), tetapi tetap memiliki kualitas supervisi yang kuat. Pada Notebook
3B, preflight teacher pada input `380x380` menghasilkan kembali accuracy
`95.52%`, sehingga checkpoint dan preprocessing teacher terverifikasi konsisten.

## Notebook 2: Reproduksi Focus-RCNet

### Arsitektur Reproduksi

Implementasi Focus-RCNet mengikuti konfigurasi stage pada paper:

```text
Focus CBS 1x1:  3 -> 24
Stage 1:       24 -> 48,  4 Sandglass blocks, SimAM
Stage 2:       48 -> 96,  3 Sandglass blocks, SimAM
Stage 3:       96 -> 192, 2 Sandglass blocks, SimAM
Stage 4:      192 -> 384, 2 Sandglass blocks, SimAM
Conv5 1x1:    384 -> 512
Classifier:   GAP -> Dropout -> Linear(512, 6)
```

Parameter count hasil reproduksi adalah `520,630`, sedangkan paper melaporkan
`525,802`. Selisihnya `-5,172` parameter atau `-0.98%`. Selisih kecil ini dicatat
sebagai keterbatasan reproduksi custom architecture.

### Hasil Phase 1

| Experiment | Model | Accuracy | Best Epoch | Delta vs Baseline |
|------------|-------|----------|------------|-------------------|
| Exp 1 | EfficientNet-B4 Teacher | 95.52% | 61 | - |
| Exp 2 | Focus-RCNet Baseline | 85.24% | 71 | - |
| Exp 3 | Focus-RCNet KD Scratch | 85.64% | 87 | +0.40 pp |
| Exp 4 | Focus-RCNet Two-Stage KD | 86.30% | 31 | +1.05 pp |

Exp 3 adalah reproduksi KD paper. Exp 4 merupakan extension penelitian: model
diadaptasi melalui CE terlebih dahulu, kemudian di-fine-tune menggunakan KD.

### Perbandingan Reproduksi Focus-RCNet dengan Paper

| Item | Paper | Reproduksi | Selisih Reproduksi |
|------|-------|------------|--------------------|
| Teacher accuracy | 97.00% | 95.52% | -1.48 pp |
| Focus-RCNet baseline accuracy | 90.00% | 85.24% | -4.76 pp |
| Focus-RCNet KD accuracy | 92.00% | 85.64% | -6.36 pp |
| KD uplift dari baseline | +2.00 pp | +0.40 pp | -1.60 pp |
| Focus-RCNet parameters | 525,802 | 520,630 | -5,172 (-0.98%) |
| Focus-RCNet FLOPs | 0.4188 G | 0.526 G | Metode hitung tidak identik |

Reproduksi belum menyamai accuracy paper. Akan tetapi, KD Scratch tidak lagi
menurunkan performa setelah arsitektur dikoreksi: accuracy meningkat dari
`85.24%` menjadi `85.64%`. Two-Stage KD memperbesar peningkatan menjadi
`86.30%`.

Gap terhadap paper dapat dipengaruhi oleh beberapa faktor:

1. Paper memakai `200` epoch dan batch size `16`; reproduksi dibatasi menjadi
   `100` epoch dan batch size `8`.
2. Mirror dataset berbeda satu gambar plastic.
3. Paper tidak mempublikasikan `T`, alpha KD, seed, detail implementasi seluruh
   layer, dan metode FLOPs secara lengkap.
4. Paper menyebut penurunan LR per 90 cycle dan cosine annealing secara
   bersamaan, sehingga detail scheduler tidak sepenuhnya unambiguous.

## Notebook 3: EfficientNet-Lite0 Controlled Comparison

EfficientNet-Lite0 diuji pada kondisi controlled yang sama dengan Focus-RCNet:
input `380x380`, random initialization, dan split dataset identik.

### Hasil Phase 2

| Experiment | Model | Accuracy | Best Epoch | Delta vs Lite0 Baseline |
|------------|-------|----------|------------|-------------------------|
| Exp 5 | EfficientNet-Lite0 Baseline | 76.15% | 86 | - |
| Exp 6 | EfficientNet-Lite0 KD Scratch | 78.26% | 92 | +2.11 pp |
| Exp 7 | EfficientNet-Lite0 Two-Stage KD | 79.58% | 38 | +3.43 pp |

KD bekerja secara sehat pada Lite0 controlled. Baik KD Scratch maupun Two-Stage
KD meningkatkan accuracy. Namun, Lite0 controlled tetap tertinggal dari
Focus-RCNet controlled.

## Tabel Akhir Controlled Comparison

Tabel ini adalah pembanding utama untuk menjawab pertanyaan arsitektur student
mana yang lebih efektif pada protokol yang sama.

| ID | Experiment | Model | Accuracy | F1 Macro | AUC Macro | Parameters | GFLOPs | Latency |
|----|------------|-------|----------|----------|-----------|------------|--------|---------|
| 1 | Teacher | EfficientNet-B4 | 95.52% | 0.9434 | 0.9955 | 17,559,374 | 8.784 | 20.291 ms |
| 2 | Baseline | Focus-RCNet | 85.24% | 0.8489 | 0.9754 | 520,630 | 0.526 | 6.461 ms |
| 3 | KD Scratch | Focus-RCNet | 85.64% | 0.8433 | 0.9756 | 520,630 | 0.526 | 6.087 ms |
| 4 | Two-Stage KD | Focus-RCNet | **86.30%** | **0.8641** | 0.9734 | **520,630** | **0.526** | 6.488 ms |
| 5 | Baseline | EfficientNet-Lite0 | 76.15% | 0.7490 | 0.9481 | 3,378,694 | 2.246 | 7.364 ms |
| 6 | KD Scratch | EfficientNet-Lite0 | 78.26% | 0.7617 | 0.9542 | 3,378,694 | 2.246 | 7.078 ms |
| 7 | Two-Stage KD | EfficientNet-Lite0 | **79.58%** | **0.7890** | **0.9546** | 3,378,694 | 2.246 | 6.970 ms |

### Interpretasi Controlled Comparison

Pada protokol terkontrol, Focus-RCNet Two-Stage KD lebih unggul dari Lite0
Two-Stage KD:

| Metric | Focus-RCNet Two-Stage | Lite0 Two-Stage | Interpretasi |
|--------|------------------------|-----------------|--------------|
| Accuracy | 86.30% | 79.58% | Focus-RCNet unggul `+6.72 pp` |
| Parameters | 520,630 | 3,378,694 | Lite0 memiliki `6.49x` parameter lebih banyak |
| GFLOPs | 0.526 | 2.246 | Lite0 membutuhkan `4.27x` GFLOPs |
| Latency | 6.488 ms | 6.970 ms | Lite0 sekitar `7.43%` lebih lambat |

Kesimpulan controlled comparison: Focus-RCNet adalah student lightweight yang
lebih efisien dan lebih akurat ketika kedua model diuji dalam protokol yang sama.

## Notebook 3B: Lite0 Native Follow-Up

Controlled comparison memperlihatkan bahwa Lite0 tidak kompetitif ketika
dipaksa memakai profil `380x380` dari Focus-RCNet. Karena itu, follow-up Native
dijalankan dengan student input `224x224` dan pretrained ImageNet weights.

### Hasil Lite0 Native

| Native Track | Initialization Path | Accuracy | Best Epoch | F1 Macro | AUC Macro | GFLOPs | Latency | Peak Memory |
|--------------|---------------------|----------|------------|----------|-----------|--------|---------|-------------|
| Native A: Baseline | ImageNet -> CE | 89.59% | 79 | 0.8896 | 0.9880 | 0.768 | 7.205 ms | 146.6 MB |
| Native B: Direct KD | ImageNet -> KD | 86.82% | 73 | 0.8556 | 0.9791 | 0.768 | 7.109 ms | 146.6 MB |
| Native C: Two-Stage KD | ImageNet -> CE -> KD | **90.91%** | 7 | **0.9053** | **0.9884** | 0.768 | 7.142 ms | 146.6 MB |

### Mengapa Native Direct KD Menurun?

Native B turun `-2.77` percentage points dari Native A. Output training memberi
indikasi bahwa soft-target pressure sangat besar sejak epoch awal:

```text
Epoch 1 Native B
Soft loss : 23.6389
Hard loss :  2.6271
Alpha     :  0.5
```

Interpretasi paling masuk akal adalah classification head Lite0 baru menggantikan
head ImageNet dan belum beradaptasi ke enam kelas TrashNet. Pada kondisi tersebut,
soft-target objective mendominasi arah optimisasi sejak awal. Native C mengurangi
masalah ini: model terlebih dahulu belajar TrashNet melalui CE, kemudian
fine-tuning KD dilakukan dengan LR lebih kecil. Penjelasan kausal ini masih perlu
dikonfirmasi melalui ablation alpha atau KD warm-up apabila penelitian ingin
diperluas.

Penurunan Native B dicatat sebagai negative transfer, bukan bug, karena:

1. Teacher preflight pada `380x380` tepat `95.52%`.
2. Dual-resolution loader terverifikasi menghasilkan student shape
   `[3, 224, 224]` dan teacher shape `[3, 380, 380]`.
3. Teacher berada pada eval mode, frozen, dan tidak memperoleh gradient.
4. KD memakai temperature scaling, KL divergence `batchmean`, faktor `T^2`, dan
   soft-target operations FP32.
5. Seluruh loss finite dan tiga checkpoint Native berhasil diverifikasi.
6. Two-Stage KD berhasil meningkatkan accuracy, sehingga jalur KD berfungsi.

## Tabel Analisis Controlled vs Native

Tabel berikut menjelaskan dampak profile Native pada Lite0. Ini adalah analisis
profile-specific, bukan controlled architectural comparison dengan Focus-RCNet.

| Lite0 Variant | Track | Student Input | Initialization | Accuracy | F1 Macro | AUC Macro | GFLOPs | Latency |
|---------------|-------|---------------|----------------|----------|----------|-----------|--------|---------|
| Exp 5: Baseline | Controlled | `380x380` | Random | 76.15% | 0.7490 | 0.9481 | 2.246 | 7.364 ms |
| Exp 6: KD Scratch | Controlled | `380x380` | Random -> KD | 78.26% | 0.7617 | 0.9542 | 2.246 | 7.078 ms |
| Exp 7: Two-Stage KD | Controlled | `380x380` | Random -> CE -> KD | 79.58% | 0.7890 | 0.9546 | 2.246 | 6.970 ms |
| Native A: Baseline | Native | `224x224` | ImageNet -> CE | 89.59% | 0.8896 | 0.9880 | 0.768 | 7.205 ms |
| Native B: Direct KD | Native | `224x224` | ImageNet -> KD | 86.82% | 0.8556 | 0.9791 | 0.768 | 7.109 ms |
| Native C: Two-Stage KD | Native | `224x224` | ImageNet -> CE -> KD | **90.91%** | **0.9053** | **0.9884** | **0.768** | 7.142 ms |

Perbandingan Lite0 Two-Stage controlled dan Native:

| Metric | Lite0 Controlled Two-Stage | Lite0 Native Two-Stage | Perubahan Native |
|--------|----------------------------|------------------------|------------------|
| Accuracy | 79.58% | 90.91% | `+11.33 pp` |
| GFLOPs | 2.246 | 0.768 | `-65.81%` |
| Latency | 6.970 ms | 7.142 ms | `+2.47%` pada pengukuran T4 ini |
| Parameters | 3,378,694 | 3,378,694 | Tidak berubah |

Native memberikan kombinasi accuracy dan FLOPs yang jauh lebih baik untuk Lite0.
Latency pada GPU T4 tidak turun secara proporsional terhadap FLOPs, sehingga
latency tetap harus diukur pada perangkat deployment target sebelum membuat
klaim produksi.

## Posisi Lite0 Native terhadap Focus-RCNet dan Paper

Lite0 Native C dapat ditempatkan sebagai follow-up deployment profile, tetapi
bukan sebagai pengganti hasil controlled.

| Metric | Focus-RCNet Two-Stage Controlled | Lite0 Native Two-Stage | Selisih Native |
|--------|----------------------------------|------------------------|----------------|
| Accuracy | 86.30% | 90.91% | `+4.61 pp` |
| Parameters | 520,630 | 3,378,694 | Lite0 Native `6.49x` lebih besar |
| GFLOPs | 0.526 | 0.768 | Lite0 Native `1.46x` lebih besar |
| Latency | 6.488 ms | 7.142 ms | Lite0 Native `10.08%` lebih lambat pada T4 |

Dibandingkan angka paper Focus-RCNet-KD `92%`, Lite0 Native C berada `-1.09`
percentage points di bawahnya. Hasil ini menarik untuk deployment-oriented
analysis, tetapi tidak membuktikan Lite0 lebih unggul secara arsitektural karena
profil training dan input berbeda.

## Kesimpulan Penelitian

Kesimpulan yang dapat dipertahankan secara metodologis:

1. Reproduksi Focus-RCNet berhasil menghasilkan arsitektur lightweight yang
   dekat dengan paper dari sisi parameter count, tetapi accuracy masih berada di
   bawah laporan paper.
2. KD Scratch Focus-RCNet memberi peningkatan kecil `+0.40 pp`; Two-Stage KD
   sebagai extension meningkatkan gain menjadi `+1.05 pp`.
3. Dalam controlled comparison, Focus-RCNet Two-Stage KD adalah pilihan student
   paling efektif: lebih akurat dan lebih ringan daripada Lite0 controlled.
4. Lite0 sangat sensitif terhadap profile training. Pada controlled `380x380`,
   Lite0 kurang kompetitif. Pada Native `224x224` dengan ImageNet initialization,
   performanya meningkat tajam.
5. Native Direct KD tidak optimal. Untuk Lite0 Native, CE adaptation sebelum KD
   merupakan strategi yang lebih efektif.
6. Lite0 Native Two-Stage KD adalah kandidat deployment-oriented dengan accuracy
   student tertinggi dalam penelitian ini (`90.91%`), sedangkan Focus-RCNet tetap
   menjadi lightweight winner pada controlled comparison.

## Keterbatasan dan Langkah Lanjutan

Keterbatasan utama:

1. Seluruh angka training masih berasal dari satu seed (`42`).
2. Dataset kecil dan imbalanced; kelas trash hanya berisi `137` gambar.
3. Mirror dataset berbeda satu gambar dari deskripsi paper.
4. Hyperparameter KD paper tidak sepenuhnya tersedia.
5. Latency diukur pada GPU Tesla T4, bukan perangkat edge target.
6. Absolute FLOPs paper dan reproduksi tidak dihitung dengan prosedur identik.

Langkah lanjutan yang paling bernilai:

1. Ulangi Native A dan Native C dengan dua atau tiga seed tambahan untuk menguji
   kestabilan gain `+1.32 pp`.
2. Jika waktu eksperimen tersedia, lakukan ablation kecil untuk Native Direct KD:
   alpha lebih kecil atau KD warm-up setelah beberapa epoch CE.
3. Ukur latency dan memory pada perangkat deployment target.
4. Laporkan hasil controlled dan Native dalam tabel terpisah pada skripsi.

## Referensi Paper

Zheng, D., Wang, R., Duan, Y., Pang, P. C. I., dan Tan, T. (2023).
*Focus-RCNet: a lightweight recyclable waste classification algorithm based on
focus and knowledge distillation*. Visual Computing for Industry, Biomedicine,
and Art, 6, 19. [https://doi.org/10.1186/s42492-023-00146-3](https://doi.org/10.1186/s42492-023-00146-3)
