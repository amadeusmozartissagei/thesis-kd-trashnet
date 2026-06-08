# Final Research Planning Summary

## 1. Posisi Penelitian

Penelitian ini tetap berangkat dari reimplementation paper Focus-RCNet:

> Focus-RCNet: a lightweight recyclable waste classification algorithm based on focus and knowledge distillation.

Tujuan utama bukan hanya mengejar angka paper secara persis, tetapi menyusun eksperimen yang lebih rapi untuk mengevaluasi:

- Reimplementation Focus-RCNet pada dataset TrashNet.
- Pengaruh knowledge distillation terhadap model ringan.
- Perbandingan logits-based KD dan self-distillation.
- Kelayakan model sangat kecil, terutama WasteNet-128K, untuk arah deployment ESP32.

Eksperimen lama dengan split 70/30 dipertahankan sebagai exploratory result. Klaim final sebaiknya memakai protokol baru dengan independent test split.

## 2. Masalah dari Eksperimen Lama

Eksperimen sebelumnya memberi banyak sinyal penting:

- Focus-RCNet two-stage KD meningkatkan akurasi dibanding baseline pada controlled comparison.
- Direct KD sensitif terhadap student dan bobot soft target.
- WasteNet-128K baseline CE menjadi kandidat kuat untuk target di bawah 200 ribu parameter.
- WasteNet teacher-assistant two-stage KD dengan konfigurasi lama tidak mempertahankan akurasi, sehingga perlu tuning ulang.
- Run 200 epoch batch size 16 mendekati protokol paper, tetapi memperlihatkan gejala overfitting/overconfidence.

Catatan interpretasi:

- Split 70/30 tidak otomatis salah.
- Optimistic bias muncul ketika validation set dipakai berulang untuk memilih model, epoch, temperature, alpha, dan tetap dilaporkan sebagai hasil akhir.
- Karena itu, eksperimen final harus memakai train/validation/test yang terpisah.

## 3. Protokol Evaluasi Final

Gunakan stratified split:

```text
70% train
15% validation
15% independent test
```

Aturan penggunaan data:

- Train set hanya untuk training.
- Validation set untuk checkpoint selection, early stopping, dan hyperparameter tuning.
- Test set hanya untuk final reporting.
- Test set tidak boleh dipakai untuk memilih model, epoch, temperature, alpha, atau konfigurasi training lain.

Recommended seed:

```python
SEEDS = [42, 123, 777, 2026, 3407]
```

Minimal klaim utama dijalankan dengan 5 seed berbeda.

## 4. K-Fold dan Independent Test

Jika memakai cross-validation, jangan lakukan k-fold pada seluruh dataset.

Skema yang disarankan:

1. Kunci independent test set sebesar 15%.
2. Sisanya 85% disebut development set.
3. Lakukan Stratified K-Fold hanya pada development set.
4. Gunakan hasil k-fold untuk memilih konfigurasi.
5. Setelah konfigurasi final dipilih, evaluasi satu kali pada independent test set.

Untuk scope S1, k-fold penuh bisa sangat mahal. Alternatif yang lebih realistis:

- Pilot tuning dengan 1 seed.
- Pilih 2-3 konfigurasi terbaik dari validation set.
- Jalankan konfigurasi final dengan 5 seed.
- Laporkan independent test result.

## 5. Pembagian Jenis Knowledge Distillation

Eksperimen final dibagi berdasarkan jenis KD.

### 5.1 Baseline CE

Baseline tanpa distillation.

Tujuan:

- Menjadi pembanding utama setiap model.
- Menjawab apakah KD benar-benar memberi peningkatan.
- Menjadi checkpoint awal untuk two-stage KD.

### 5.2 Logits-Based KD / Vanilla KD

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

### 5.3 Self-Distillation / CORD-CORSD

Self-distillation tidak bergantung pada teacher eksternal besar. Knowledge ditransfer dari model yang sama, misalnya dari classifier/layer terdalam ke classifier/layer dangkal.

Catatan istilah:

- Pastikan paper/metode yang dipakai adalah CORD atau CORSD.
- Kandidat yang relevan untuk image classification adalah CORSD: Class-Oriented Relational Self Distillation.
- CORSD memakai relational knowledge, structured pairs, relation networks, dan auxiliary classifiers.

Tujuan:

- Menguji apakah model kecil dapat memperbaiki generalisasi tanpa teacher besar.
- Membandingkan external teacher KD dengan self-distillation.
- Mengurangi ketergantungan pada EfficientNet-B4 saat target akhirnya adalah model kecil.

## 6. Model yang Diprioritaskan

### 6.1 Referensi dan Teacher

| Model | Peran |
| --- | --- |
| EfficientNet-B4 | Teacher utama dan referensi performa |
| Focus-RCNet | Reimplementation utama dari paper dan teacher assistant |

### 6.2 Student Ringan

| Model | Peran |
| --- | --- |
| WasteNet-128K | Kandidat utama deployment ESP32 karena parameternya di bawah 200 ribu |
| WasteNet-256K | Pembanding kapasitas, bukan target utama ESP32 |

### 6.3 Model Sekunder

EfficientNet-Lite0 dapat tetap dipakai sebagai pembanding deployment-oriented, tetapi tidak perlu menjadi fokus final jika scope mulai melebar.

## 7. Rancangan Eksperimen Utama

### 7.1 Focus-RCNet Track

| ID | Eksperimen | Tujuan |
| --- | --- | --- |
| F1 | Focus-RCNet CE | Baseline reimplementation paper |
| F2 | Focus-RCNet direct KD | Replikasi logits KD dari EfficientNet-B4 |
| F3 | Focus-RCNet two-stage KD | Extension: CE pretraining lalu KD |
| F4 | Focus-RCNet self-distillation | Bandingkan external KD vs self-distillation |

### 7.2 WasteNet-128K Track

| ID | Eksperimen | Tujuan |
| --- | --- | --- |
| W128-1 | WasteNet-128K CE | Baseline utama model under 200K |
| W128-2 | WasteNet-128K direct KD | Uji KD dari EfficientNet-B4 ke tiny student |
| W128-3 | WasteNet-128K teacher-assistant KD | Uji EfficientNet-B4 -> Focus-RCNet -> WasteNet |
| W128-4 | WasteNet-128K self-distillation | Uji CORD/CORSD pada target ESP32 |

### 7.3 WasteNet-256K Track

| ID | Eksperimen | Tujuan |
| --- | --- | --- |
| W256-1 | WasteNet-256K CE | Pembanding kapasitas |
| W256-2 | WasteNet-256K direct KD | Uji apakah kapasitas lebih besar lebih cocok untuk KD |
| W256-3 | WasteNet-256K teacher-assistant KD | Pembanding TA-KD terhadap 128K |
| W256-4 | WasteNet-256K self-distillation | Pembanding self-distillation terhadap 128K |

Jika compute terbatas, prioritas utama adalah Focus-RCNet dan WasteNet-128K.

## 8. Hyperparameter Temperature dan Alpha

Temperature dan alpha tetap perlu dicoba, terutama karena hasil lama menunjukkan `T=4, alpha=0.5` tidak selalu cocok.

### 8.1 Grid Awal untuk Logits-Based KD

```python
TEMPERATURES = [2, 4, 6, 8]
ALPHAS = [0.05, 0.1, 0.3, 0.5]
```

Untuk WasteNet, prioritas alpha kecil:

```python
TEMPERATURES_WASTENET = [2, 4, 6]
ALPHAS_WASTENET = [0.05, 0.1, 0.3]
```

Alasan:

- Student kecil mudah terdorong terlalu kuat oleh soft target teacher.
- Hasil lama menunjukkan alpha 0.5 dapat merusak performa WasteNet.

### 8.2 Grid Awal untuk Self-Distillation

Jika CORD/CORSD yang dipakai memiliki auxiliary logits distillation:

```python
SELF_DISTILL_TEMPERATURES = [2, 4]
SELF_DISTILL_WEIGHTS = [0.1, 0.3]
```

Jika memakai CORSD penuh, hyperparameter tambahan perlu dibatasi, misalnya:

```python
RELATION_LOSS_WEIGHTS = [0.01, 0.05]
AUX_LOSS_WEIGHTS = [0.1, 0.2]
```

Jangan grid semua kombinasi sekaligus. Mulai dari pilot kecil.

## 9. Strategi Tuning yang Aman

Urutan tuning:

1. Jalankan baseline CE.
2. Jalankan KD default sebagai anchor.
3. Pilot grid temperature-alpha dengan 1 seed.
4. Pilih top 2 atau top 3 berdasarkan validation metric.
5. Jalankan konfigurasi terpilih dengan 5 seed.
6. Evaluasi final pada independent test set.

Aturan penting:

- Test set tidak boleh disentuh selama tuning.
- Jika suatu konfigurasi dipilih karena test result, hasilnya tidak valid sebagai klaim final.
- Simpan semua konfigurasi, seed, dan checkpoint metadata.

## 10. Metrik Evaluasi Konsisten

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

## 11. Analisis Overfitting dan Underfitting

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

### 11.1 Overfitting

Indikasi overfitting:

- Train accuracy sangat tinggi.
- Validation accuracy stagnan atau turun.
- Validation loss naik.
- Gap train-validation besar.
- Best validation loss terjadi jauh lebih awal daripada best validation accuracy.

Interpretasi yang aman:

> Model menunjukkan gejala overfitting atau overconfidence, terutama ketika train accuracy mendekati 99-100% sementara validation loss meningkat.

### 11.2 Underfitting

Indikasi underfitting:

- Train accuracy rendah.
- Validation accuracy juga rendah.
- Train loss dan validation loss masih tinggi.
- Model belum belajar representasi yang cukup.

### 11.3 KD Degradation

Indikasi KD degradation:

- Checkpoint CE awal lebih baik daripada hasil setelah KD.
- Best epoch KD sangat awal.
- Train accuracy ikut turun atau tidak stabil.
- Soft loss mendominasi hard CE loss.

Ini penting untuk WasteNet karena hasil lama TA-KD turun dari sekitar 79% ke sekitar 68-69%.

## 12. Narasi Final yang Disarankan

Pertanyaan penelitian final dapat disusun sebagai:

1. Apakah reimplementation Focus-RCNet dapat mendekati hasil paper pada TrashNet?
2. Apakah logits-based KD meningkatkan performa model ringan dibanding CE?
3. Apakah two-stage KD lebih stabil daripada direct KD?
4. Apakah teacher-assistant KD membantu student sangat kecil seperti WasteNet?
5. Apakah self-distillation CORD/CORSD lebih cocok untuk model kecil dibanding external teacher KD?
6. Bagaimana trade-off akurasi, jumlah parameter, latency, dan risiko overfitting?

Klaim final yang paling aman:

- Focus-RCNet adalah reimplementation utama dari paper.
- WasteNet-128K adalah kandidat utama untuk arah ESP32 karena parameter di bawah 200 ribu.
- Logits-based KD dan self-distillation dibandingkan secara adil dengan split, seed, dan metrik yang sama.
- Hasil lama membuktikan perlunya independent test split dan hyperparameter tuning.

## 13. Prioritas Eksekusi

Jika waktu dan compute terbatas, jalankan urutan berikut:

1. Buat split 70/15/15 dan simpan indeks untuk 5 seed.
2. Train EfficientNet-B4 teacher untuk 5 seed.
3. Train Focus-RCNet CE dan Focus-RCNet two-stage KD.
4. Train WasteNet-128K CE.
5. Jalankan pilot KD temperature-alpha untuk WasteNet-128K.
6. Jalankan WasteNet-128K best logits KD dengan 5 seed.
7. Implementasi self-distillation CORD/CORSD sederhana untuk WasteNet-128K.
8. Jika masih ada waktu, tambahkan WasteNet-256K sebagai pembanding kapasitas.

## 14. Status Eksperimen Lama

Eksperimen lama tetap berguna, tetapi posisinya:

| Eksperimen lama | Status |
| --- | --- |
| Notebook 1 teacher | Preliminary teacher reference |
| Notebook 2 Focus-RCNet | Exploratory controlled comparison |
| Notebook 2B alpha ablation | Diagnostic |
| Notebook 3 Lite0 controlled | Secondary comparison |
| Notebook 3B Lite0 native | Deployment-oriented exploratory |
| Notebook 3C Lite0 alpha ablation | Diagnostic |
| Notebook 4 WasteNet baseline/direct KD | Preliminary WasteNet screening |
| Notebook 5 WasteNet TA-KD | Negative ablation / evidence for retuning |
| 200e-bs16 follow-up | Paper-protocol follow-up with overfitting discussion |

## 15. Ringkasan Keputusan

- Gunakan 70/15/15 untuk klaim final.
- Gunakan minimal 5 seed untuk eksperimen utama.
- Bagi penelitian menjadi dua jenis KD: logits-based KD dan self-distillation.
- Tetap tuning temperature dan alpha, tetapi hanya memakai validation set.
- Fokus deployment pada WasteNet-128K karena parameter di bawah 200 ribu.
- Jangan menjadikan hasil WasteNet TA-KD lama sebagai klaim positif.
- Jadikan overfitting/underfitting sebagai bagian resmi dari analisis, bukan catatan sampingan.

