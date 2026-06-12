# Final Research Planning Summary

## 1. Posisi dan Tujuan Penelitian

Penelitian ini tetap berangkat dari reimplementation paper Focus-RCNet:

> Focus-RCNet: a lightweight recyclable waste classification algorithm based on focus and knowledge distillation.

Tujuan utama penelitian bukan hanya mengejar angka paper secara persis, tetapi menyusun eksperimen yang lebih rapi untuk mengevaluasi:

- Reimplementation Focus-RCNet pada dataset TrashNet.
- Pengaruh knowledge distillation terhadap model ringan.
- Perbandingan logits-based KD dan self-distillation.
- Kelayakan model sangat kecil, terutama WasteNet-128K, untuk arah deployment ESP32.

Fokus final penelitian:

- Focus-RCNet menjadi reimplementation utama dari paper.
- EfficientNet-B4 menjadi teacher utama dan referensi performa.
- WasteNet-128K menjadi target utama deployment karena jumlah parameter di bawah 200 ribu.
- WasteNet-256K menjadi pembanding kapasitas, bukan target utama deployment.
- Hasil final wajib menggunakan protokol baru dengan independent test split.

## 2. Status dan Masalah Eksperimen Lama

Seluruh eksperimen lama diposisikan sebagai preliminary/exploratory result, bukan klaim final. Eksperimen lama tetap berguna sebagai justifikasi desain eksperimen final, termasuk perlunya pilot tuning alpha, evidence bahwa teacher-assistant KD perlu retuning, dan evidence untuk Focus-RCNet teacher-assistant selection.

Eksperimen sebelumnya memberi sinyal penting:

- Focus-RCNet two-stage KD meningkatkan akurasi dibanding baseline pada controlled comparison.
- Direct KD sensitif terhadap student dan bobot soft target.
- WasteNet-128K baseline CE menjadi kandidat kuat untuk target di bawah 200 ribu parameter.
- WasteNet teacher-assistant two-stage KD dengan konfigurasi lama tidak mempertahankan akurasi, sehingga perlu tuning ulang.
- Run 200 epoch batch size 16 mendekati protokol paper, tetapi memperlihatkan gejala overfitting/overconfidence.

Sinyal dari archive yang mempengaruhi desain final:

- Focus-RCNet direct KD 200e-bs16 dengan `T=4, alpha=0.5` mencapai sekitar 88,41%, lebih tinggi daripada baseline 200e-bs16 sekitar 86,30%.
- Focus-RCNet two-stage KD 200e-bs16 dengan `T=4, alpha=0.5` mencapai sekitar 86,03%, sehingga tidak otomatis menjadi teacher assistant terbaik.
- WasteNet direct KD dengan `T=4, alpha=0.5` turun sedikit dibanding CE: WasteNet-128K sekitar 79,45% -> 78,66%, WasteNet-256K sekitar 79,31% -> 77,34%.
- WasteNet teacher-assistant KD dengan `T=4, alpha=0.5` turun besar: WasteNet-128K sekitar 69,04%, WasteNet-256K sekitar 68,12%.
- Karena itu, final rerun perlu pilot tuning sebelum 5-seed final, terutama untuk WasteNet dan terutama untuk TA-KD.

Masalah utama eksperimen lama adalah belum memakai independent test split. Protokol train/test saja berisiko menghasilkan optimistic bias ketika validation/test dipakai berulang untuk memilih model, epoch, temperature, alpha, atau konfigurasi training lain, lalu hasil pada set yang sama tetap dilaporkan sebagai hasil akhir.

Karena itu, eksperimen final harus memakai train/validation/test yang terpisah.

## 3. Pertanyaan Penelitian dan Klaim Final

Pertanyaan penelitian final dapat disusun sebagai:

1. Apakah reimplementation Focus-RCNet dapat mendekati hasil paper pada TrashNet?
2. Apakah logits-based KD meningkatkan performa model ringan dibanding CE?
3. Untuk Focus-RCNet, apakah direct KD atau two-stage KD lebih layak menjadi teacher assistant?
4. Apakah teacher-assistant KD membantu student sangat kecil seperti WasteNet setelah `T`, `alpha`, dan stage-2 LR dituning?
5. Apakah self-distillation CORD lebih cocok untuk model kecil dibanding external teacher KD?
6. Bagaimana trade-off akurasi, jumlah parameter, latency, dan risiko overfitting?

Klaim final yang paling aman:

- Focus-RCNet adalah reimplementation utama dari paper.
- WasteNet-128K adalah kandidat utama untuk arah ESP32 karena parameter di bawah 200 ribu.
- Logits-based KD dan self-distillation dibandingkan secara adil dengan split, seed, dan metrik yang sama.
- Hasil lama membuktikan perlunya independent test split dan hyperparameter tuning.
- Jangan menjadikan hasil WasteNet TA-KD lama sebagai klaim positif; gunakan sebagai alasan retuning.

## 4. Protokol Evaluasi Final

Gunakan stratified split:

```text
70% train
15% validation
15% independent test
```

Aturan penggunaan data:

- Train set hanya untuk training.
- Validation set untuk checkpoint selection, early stopping pada pilot, dan hyperparameter tuning.
- Test set hanya untuk final reporting.
- Test set tidak boleh dipakai untuk memilih model, epoch, temperature, alpha, atau konfigurasi training lain.

Recommended seed:

```python
SEEDS = [42, 123, 777, 2026, 3407]
```

Minimal klaim utama dijalankan dengan 5 seed berbeda.

Independent test set tetap digunakan untuk evaluasi akhir. K-Fold Cross Validation tidak digunakan karena keterbatasan komputasi dan ruang lingkup penelitian S1. Hyperparameter tuning dilakukan menggunakan validation set pada skema train-validation-test (70%-15%-15%).

Strategi validasi:

- Pilot tuning dengan 1 seed pada validation set.
- Pilih 2-3 konfigurasi terbaik berdasarkan validation metric.
- Jalankan konfigurasi final dengan 5 seed.
- Evaluasi independent test set satu kali untuk final reporting.

### 4.1 R0: Final Data Protocol Setup

R0 adalah tahap persiapan protokol data, bukan eksperimen training model. R0 wajib selesai sebelum R1-R10 karena semua eksperimen final harus memakai split, class mapping, dan aturan evaluasi yang sama.

Tujuan R0:

- Mengunci dataset TrashNet yang dipakai untuk final experiment.
- Mengecek jumlah gambar per kelas dan mendeteksi file rusak/duplikat jika ada.
- Menetapkan class mapping final, misalnya urutan label dan indeks kelas.
- Membuat stratified split 70/15/15 untuk train, validation, dan independent test.
- Menyimpan split manifest agar semua run dapat direproduksi.
- Memastikan test set tidak dipakai untuk tuning, checkpoint selection, atau pemilihan konfigurasi.

Output minimal R0:

```text
class_mapping.json
dataset_inventory.csv
split_manifest_seed_42.csv
split_manifest_seed_123.csv
split_manifest_seed_777.csv
split_manifest_seed_2026.csv
split_manifest_seed_3407.csv
split_summary.csv
```

Isi minimal split manifest:

```text
image_path,label,class_id,split,seed
```

Aturan split:

- Untuk setiap seed, semua model pada seed tersebut harus memakai split yang sama.
- Validation set hanya dipakai untuk tuning dan checkpoint selection.
- Independent test set hanya dipakai setelah konfigurasi final dibekukan.
- R0 tidak menghasilkan accuracy, F1, atau metric performa model karena belum ada training.

## 5. Model yang Digunakan

### 5.1 Referensi dan Teacher

| Model | Peran |
| --- | --- |
| EfficientNet-B4 | Teacher utama dan referensi performa |
| Focus-RCNet | Reimplementation utama dari paper dan teacher assistant |

### 5.2 Student Ringan

| Model | Peran |
| --- | --- |
| WasteNet-128K | Kandidat utama deployment ESP32 karena parameternya di bawah 200 ribu |
| WasteNet-256K | Pembanding kapasitas, bukan target utama ESP32 |

### 5.3 Model Sekunder

EfficientNet-Lite0 dapat tetap dipakai sebagai pembanding deployment-oriented, tetapi tidak perlu menjadi fokus final jika scope mulai melebar.

## 6. Jenis Knowledge Distillation yang Diuji

Eksperimen final dibagi menjadi baseline CE, logits-based KD, dan self-distillation.

### 6.1 Baseline CE

Baseline CE adalah training tanpa distillation.

Tujuan:

- Menjadi pembanding utama setiap model.
- Menjawab apakah KD benar-benar memberi peningkatan.
- Menjadi checkpoint awal untuk two-stage KD.

### 6.2 Logits-Based KD / Vanilla KD

Jenis KD klasik dengan soft logits teacher.

Loss:

```text
L = alpha * KL(student_logits / T, teacher_logits / T) * T^2
    + (1 - alpha) * CrossEntropyLoss
```

Subjenis:

- Direct KD: EfficientNet-B4 -> student.
- Two-stage KD: CE pretraining -> KD fine-tuning.
- Teacher-assistant KD: EfficientNet-B4 -> Focus-RCNet -> WasteNet.

Strategi resolusi input untuk KD WasteNet:

- Teacher EfficientNet-B4 dilatih dan di-infer pada 380x380 agar soft target optimal.
- Student WasteNet tetap dilatih pada 160x160.
- Setiap batch di-resize dua kali: 380x380 untuk teacher inference, 160x160 untuk student training.
- Eksperimen lama di archive memakai resolusi tunggal 160x160 untuk keduanya, sehingga teacher kurang optimal. Final run memperbaiki ini dengan dual-resolution.

### 6.3 Self-Distillation / CORD

Self-distillation tidak bergantung pada teacher eksternal besar. Knowledge ditransfer dari model yang sama, misalnya dari classifier/layer terdalam ke classifier/layer dangkal.

Metode acuan final adalah CORD self-distillation. CORD diposisikan sebagai jalur self-distillation untuk menguji apakah model kecil dapat memperoleh regularisasi/knowledge transfer dari struktur internalnya sendiri tanpa teacher eksternal seperti EfficientNet-B4.

Prinsip penggunaan CORD dalam penelitian ini:

- Tidak memakai teacher eksternal besar.
- Diterapkan pada WasteNet-128K sebagai target utama deployment.
- Diterapkan juga pada WasteNet-256K sebagai pembanding kapasitas.
- Dibandingkan langsung dengan CE baseline, direct KD, dan teacher-assistant KD.
- Detail loss, branch/head, dan representasi yang didistilasi harus mengikuti formulasi CORD yang dipilih.

Adaptasi untuk penelitian ini:

- CORD diadaptasi ke arsitektur WasteNet untuk waste classification pada TrashNet.
- Adaptasi perlu menyesuaikan titik distillation dengan struktur depthwise-separable WasteNet.
- Jika CORD membutuhkan auxiliary head, projection head, atau intermediate representation, posisinya harus dibuat minimal agar parameter dan latency tetap relevan untuk arah ESP32.

Tujuan:

- Menguji apakah model kecil dapat memperbaiki generalisasi tanpa teacher besar.
- Membandingkan external teacher KD dengan self-distillation.
- Mengurangi ketergantungan pada EfficientNet-B4 saat target akhirnya adalah model kecil.

## 7. Rancangan Eksperimen Utama

Final claim memakai 10 experiment groups, yaitu R1-R10. R0 tetap dicatat dalam rancangan karena R0 adalah fondasi data/protokol, tetapi R0 bukan training run dan tidak dihitung sebagai experiment group final. Pilot tuning boleh dilakukan sebelum final rerun, tetapi pilot bukan klaim final. Setelah konfigurasi dipilih dari validation set, konfigurasi dibekukan lalu semua experiment groups final dijalankan dengan 5 seed.

Catatan penting: R2 bukan asumsi bahwa two-stage KD pasti paling baik. R2 adalah proses memilih konfigurasi Focus-RCNet KD terbaik, misalnya direct KD atau two-stage KD, berdasarkan validation metric.

| ID | Eksperimen final | Teacher | Model akhir | Tujuan |
| --- | --- | --- | --- | --- |
| R0 | Final data protocol setup | None | None | Mengunci dataset, class mapping, split manifest, dan aturan evaluasi |
| R1 | EfficientNet-B4 teacher | None | EfficientNet-B4 | Teacher utama dan referensi performa |
| R2 | Focus-RCNet KD teacher-assistant selection | EfficientNet-B4 | Focus-RCNet | Memilih Focus-RCNet KD terbaik sebagai teacher assistant |
| R3 | WasteNet-128K baseline CE | None | WasteNet-128K | Baseline utama under 200K parameter |
| R4 | WasteNet-128K direct KD | EfficientNet-B4 | WasteNet-128K | Menguji big-teacher direct KD ke tiny student |
| R5 | WasteNet-128K teacher-assistant KD | Focus-RCNet | WasteNet-128K | Menguji EfficientNet-B4 -> Focus-RCNet -> WasteNet-128K |
| R6 | WasteNet-256K baseline CE | None | WasteNet-256K | Baseline pembanding kapasitas |
| R7 | WasteNet-256K direct KD | EfficientNet-B4 | WasteNet-256K | Menguji direct KD pada kapasitas lebih besar |
| R8 | WasteNet-256K teacher-assistant KD | Focus-RCNet | WasteNet-256K | Pembanding TA-KD terhadap WasteNet-128K |
| R9 | WasteNet-128K CORD self-distillation | Self / internal representation | WasteNet-128K | Pembanding tanpa external teacher untuk target ESP32 |
| R10 | WasteNet-256K CORD self-distillation | Self / internal representation | WasteNet-256K | Pembanding self-distillation pada kapasitas lebih besar |

Dependency chain:

```text
R0 Final data protocol setup
  -> R1-R10 semua experiment groups

R1 EfficientNet-B4 teacher
  -> R2 Focus-RCNet teacher-assistant selection
      -> R5 WasteNet-128K TA-KD
      -> R8 WasteNet-256K TA-KD

R1 EfficientNet-B4 teacher
  -> R4 WasteNet-128K direct KD
  -> R7 WasteNet-256K direct KD

R3 WasteNet-128K baseline CE
  -> R5 WasteNet-128K TA-KD
  -> R9 WasteNet-128K CORD

R6 WasteNet-256K baseline CE
  -> R8 WasteNet-256K TA-KD
  -> R10 WasteNet-256K CORD
```

Jika semua experiment groups final dijalankan dengan 5 seed:

```text
R0 protocol setup = 1 setup stage
10 experiment groups x 5 seeds = 50 final runs
```

## 8. Pilot Hyperparameter Sebelum Final Run

Temperature, alpha, dan loss weight dicari sebelum final 5-seed run. Jangan memilih konfigurasi berdasarkan independent test. Pilot boleh memakai 1 seed terlebih dahulu, atau 2 seed jika compute memungkinkan. Early stopping boleh dipakai selama pilot untuk menghemat compute, misalnya patience 15-20 epoch berdasarkan validation loss.

Prinsip utama:

- Pilot memakai validation set.
- Independent test set tetap dikunci dan tidak disentuh.
- Setelah konfigurasi dipilih, buat satu frozen config per experiment group.
- Jangan memilih `T/alpha` berbeda-beda per seed.
- Baseline CE dan EfficientNet-B4 teacher tidak memakai `T/alpha`.

### 8.1 Pilot Logits-Based KD

Target tuning logits KD:

```text
R2 Focus-RCNet KD teacher-assistant selection
R4 WasteNet-128K direct KD
R5 WasteNet-128K TA-KD
R7 WasteNet-256K direct KD
R8 WasteNet-256K TA-KD
```

Rekomendasi berdasarkan archive:

- R2 tidak perlu grid besar dulu. Bandingkan Focus-RCNet direct KD dan Focus-RCNet two-stage KD dengan anchor `T=4, alpha=0.5`, lalu pilih teacher assistant berdasarkan validation metric.
- Untuk R4, jalankan small grid pada WasteNet-128K direct KD:

```python
TEMPERATURES_W128_DIRECT = [2, 4, 6]
ALPHAS_W128_DIRECT = [0.05, 0.1, 0.3]
ANCHOR_W128_DIRECT = {"temperature": 4, "alpha": 0.5}
```

- Untuk R5, jalankan small grid pada WasteNet-128K TA-KD, tetapi tambahkan CE fine-tune control untuk memisahkan masalah fine-tuning dari masalah KD:

```python
TEMPERATURES_W128_TA = [2, 4, 6]
ALPHAS_W128_TA = [0.05, 0.1, 0.3]
ANCHOR_W128_TA = {"temperature": 4, "alpha": 0.5}
STAGE2_LR_CANDIDATES = [0.001, 0.0005]
```

- Untuk R7 dan R8, jangan full grid dulu. Pakai konfigurasi terbaik W128 sebagai starting point, tambah runner-up W128, dan anchor `T=4, alpha=0.5`.
- Alpha kecil diprioritaskan untuk WasteNet karena archive menunjukkan `alpha=0.5` dapat membuat soft loss terlalu dominan.

### 8.2 Pilot CORD Self-Distillation

CORD self-distillation tidak bergantung pada EfficientNet-B4 atau Focus-RCNet. Pilot dimulai dari WasteNet-128K karena ini klaim utama deployment.

Grid awal yang dibatasi:

```python
CORD_LOSS_WEIGHTS = [0.1, 0.2]
```

Jika implementasi CORD memakai softened logits atau probability matching, tambahkan:

```python
SELF_DISTILL_TEMPERATURES = [2, 4]
```

Urutan self-distillation:

1. Pilot R9 WasteNet-128K CORD.
2. Pilih loss weight dan komponen CORD berdasarkan validation metric.
3. Terapkan konfigurasi terbaik atau konfigurasi terdekat ke R10 WasteNet-256K sebagai secondary comparison.
4. Bekukan konfigurasi sebelum final 5-seed run.

## 9. Strategi Eksekusi Final

Flow yang disepakati:

1. Jalankan R0: buat dataset inventory, class mapping, stratified split 70/15/15, dan split manifest untuk semua seed.
2. Jalankan pilot seed, misalnya seed 42, tanpa menyentuh independent test.
3. Train anchor untuk pilot: EfficientNet-B4 teacher, kandidat Focus-RCNet KD, dan WasteNet CE baseline.
4. Pilih Focus-RCNet teacher assistant dengan membandingkan direct KD vs two-stage KD berdasarkan validation metric.
5. Pilot W128 direct KD dengan small grid `T/alpha`.
6. Pilot W128 TA-KD dengan small grid `T/alpha`, CE fine-tune control, dan stage-2 LR lebih kecil.
7. Pilot W128 CORD dengan grid loss weight kecil.
8. Untuk W256, gunakan konfigurasi terbaik W128, runner-up W128, dan anchor `T=4, alpha=0.5` sebagai limited secondary pilot.
9. Freeze satu konfigurasi final untuk setiap experiment group.
10. Jalankan R1-R10 final dengan 5 seed memakai frozen config.
11. Evaluasi independent test satu kali untuk final reporting.

Aturan penting:

- Test set tidak boleh disentuh selama pilot tuning.
- Jika suatu konfigurasi dipilih karena test result, hasilnya tidak valid sebagai klaim final.
- Simpan semua konfigurasi, seed, split index, checkpoint metadata, dan prediction CSV.

Early stopping dibedakan berdasarkan fase:

- Pilot tuning: early stopping boleh dipakai, misalnya patience 15-20 epoch berdasarkan validation loss.
- Final 5-seed run: early stopping tidak digunakan. Semua run dijalankan sampai epoch terakhir dan checkpoint terbaik dipilih berdasarkan best validation accuracy, agar setiap seed mendapat kesempatan training yang sama dan hasilnya fair untuk dibandingkan.

## 10. Metrik Evaluasi dan Artefak

Laporkan per seed dan agregat:

- Accuracy.
- Precision macro.
- Recall macro.
- F1 macro.
- AUC macro OVR.
- Per-class precision/recall/F1.
- Confusion matrix.
- Train loss dan validation loss.
- Train accuracy dan validation accuracy.
- Best epoch.
- Parameters.
- FLOPs atau MACs.
- Checkpoint size.
- Inference latency.

Format agregasi:

```text
mean
standard deviation
best seed
95% confidence interval jika memungkinkan
```

Prediction CSV minimal:

```text
image_path,label,prediction,
prob_class_0,prob_class_1,prob_class_2,
prob_class_3,prob_class_4,prob_class_5,
seed,model_id,split
```

Setiap run harus menyimpan indikator:

```text
best_epoch
train_acc_at_best
val_acc_at_best
test_acc_final
train_val_gap
val_loss_min_epoch
final_train_acc
final_val_acc
final_train_loss
final_val_loss
```

## 11. Statistical Testing

Untuk perbandingan antar model pada test set yang sama, gunakan:

- McNemar test: membandingkan apakah dua model berbeda secara signifikan pada gambar yang sama.
- Bootstrap confidence interval: estimasi interval kepercayaan 95% untuk accuracy dan F1 macro dengan resampling 1000-2000 kali dari prediction CSV.
- Paired mean difference across 5 seeds: hitung selisih rata-rata metrik antar dua model di setiap seed, lalu laporkan mean +/- std dari selisih tersebut.

Jika banyak pasangan model dibandingkan, terapkan koreksi Holm-Bonferroni agar p-value tidak terlalu optimis.

Semua test ini membutuhkan prediction CSV yang sudah didefinisikan di atas.

## 12. Analisis Overfitting, Underfitting, dan KD Degradation

Berdasarkan archive, gejala overfitting hanya muncul pada konfigurasi 200 epoch batch size 16. Validation loss minimum terjadi di epoch 76, best accuracy di epoch 175, dan train accuracy mencapai 99,38%. Konfigurasi 100 epoch batch size 8 atau batch size 16 tidak menunjukkan gejala tersebut.

### 12.1 Overfitting

Indikasi overfitting:

- Train accuracy sangat tinggi.
- Validation accuracy stagnan atau turun.
- Validation loss naik.
- Gap train-validation besar.
- Best validation loss terjadi jauh lebih awal daripada best validation accuracy.

Interpretasi yang aman:

> Model menunjukkan gejala overfitting atau overconfidence, terutama ketika train accuracy mendekati 99-100% sementara validation loss meningkat.

### 12.2 Underfitting

Indikasi underfitting:

- Train accuracy rendah.
- Validation accuracy juga rendah.
- Train loss dan validation loss masih tinggi.
- Model belum belajar representasi yang cukup.

### 12.3 KD Degradation

Indikasi KD degradation:

- Checkpoint CE awal lebih baik daripada hasil setelah KD.
- Best epoch KD sangat awal.
- Train accuracy ikut turun atau tidak stabil.
- Soft loss mendominasi hard CE loss.

Ini penting untuk WasteNet karena hasil lama TA-KD turun dari sekitar 79% ke sekitar 68-69%.

## 13. Prioritas Jika Compute Terbatas

Jika compute cukup, jalankan seluruh R1-R10:

```text
R0 wajib selesai dulu
R1-R10 x 5 seeds = 50 final runs
```

Jika compute terbatas, prioritas final claim adalah:

```text
R0 Final data protocol setup
R1 EfficientNet-B4 teacher
R2 Focus-RCNet teacher-assistant selection
R3 WasteNet-128K CE
R4 WasteNet-128K direct KD
R5 WasteNet-128K TA-KD
R9 WasteNet-128K CORD
```

WasteNet-256K tetap berguna, tetapi posisinya secondary capacity comparison:

```text
R6 WasteNet-256K CE
R7 WasteNet-256K direct KD
R8 WasteNet-256K TA-KD
R10 WasteNet-256K CORD
```

## 14. Ringkasan Keputusan

- Gunakan split 70/15/15 untuk klaim final.
- Jalankan R0 terlebih dahulu untuk mengunci dataset, class mapping, split manifest, dan aturan evaluasi.
- Gunakan minimal 5 seed untuk eksperimen utama.
- Final claim ideal memakai R1-R10.
- Final claim minimal difokuskan pada R1, R2, R3, R4, R5, dan R9.
- Bagi penelitian menjadi dua jenis KD: logits-based KD dan self-distillation CORD.
- Lakukan pilot tuning sebelum final 5-seed run, lalu freeze konfigurasi.
- Tune `temperature` dan `alpha` hanya untuk logits KD dengan validation set.
- Tune loss weight CORD secara terpisah dari logits KD.
- Pilih Focus-RCNet teacher assistant dari direct KD vs two-stage KD, jangan mengunci two-stage secara asumtif.
- Fokus deployment pada WasteNet-128K karena parameter di bawah 200 ribu.
- Jadikan overfitting/underfitting dan KD degradation sebagai bagian resmi dari analisis, bukan catatan sampingan.
