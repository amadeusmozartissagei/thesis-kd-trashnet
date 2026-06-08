DRAFT PENELITIAN

Reimplementation Focus-RCNet dan Evaluasi Strategi Knowledge Distillation

Klasifikasi sampah daur ulang pada dataset TrashNet

Ringkasan metode, hasil eksperimen, analisis trade-off, dan agenda validasi lanjutan

| Nama peneliti | Hamza Pratama |
| --- | --- |
| Status dokumen | Draft kerja penelitian - versi Markdown revisi |
| Cakupan | Eksperimen lokal pada folder Results |
| Dataset | TrashNet resmi: 2.527 citra, 6 kelas |
| Tanggal penyusunan | 2 Juni 2026 |

# Ringkasan Eksekutif

Penelitian ini mengevaluasi reimplementation arsitektur Focus-RCNet untuk klasifikasi enam kelas sampah daur ulang pada dataset TrashNet. Fokus utama eksperimen adalah mengukur manfaat knowledge distillation (KD), membandingkan direct KD dengan two-stage KD, dan menilai trade-off antara akurasi dan efisiensi model ringan.

Rangkaian eksperimen menghasilkan tiga temuan utama. Pertama, two-stage KD memberikan peningkatan paling stabil pada Focus-RCNet dan profil EfficientNet-Lite0 yang relevan untuk deployment. Kedua, direct KD sensitif terhadap profil student dan bobot soft target. Ketiga, perubahan protokol Focus-RCNet baseline dari 100 epoch dan batch size 8 menjadi 200 epoch dan batch size 16 menaikkan akurasi validasi sebesar 1,06 percentage point (pp), tetapi belum mencapai akurasi 90% yang dilaporkan paper.

Kesimpulan utama: Two-stage KD adalah strategi yang paling stabil pada eksperimen lokal. Klaim yang dapat dipertahankan bukan bahwa reimplementation mengalahkan paper, melainkan bahwa strategi pelatihan dua tahap memberi peningkatan konsisten pada dua arsitektur student dengan profil komputasi berbeda.

Catatan kurasi eksperimen: tidak ada notebook yang perlu dihapus dari arsip. Namun, untuk naskah utama, `notebook3-phase2-efficientnet-lite0.ipynb` lebih tepat diposisikan sebagai pembanding sekunder terkontrol, `notebook3c-lite0-native-direct-kd-alpha-ablation.ipynb` sebagai diagnostik ablation/lampiran, dan `focus-rcnet-baseline-ce-e200-bs16.ipynb` sebagai follow-up protokol paper, bukan bagian dari controlled comparison utama.

Tabel 1. Ringkasan hasil utama dan posisi narasi

| Kelompok | Konfigurasi terbaik | Val. Acc. | Peningkatan | Posisi dalam naskah |
| --- | --- | --- | --- | --- |
| Focus-RCNet controlled | Two-stage KD | 86,30% | +1,06pp vs baseline | Hasil utama reimplementation |
| Focus-RCNet follow-up | Baseline CE, 200e-bs16 | 86,30% | +1,06pp vs 100e-bs8 | Follow-up protokol; tidak digabung ke controlled comparison |
| Lite0 native | Two-stage KD | 90,91% | +1,32pp vs baseline | Pembanding deployment-oriented utama |
| Lite0 controlled | Two-stage KD | 79,58% | +3,43pp vs baseline | Pembanding sekunder terkontrol |

Sumber: Output notebook eksperimen pada folder Results.

# 1. Pendahuluan

## 1.1 Latar Belakang

Klasifikasi sampah otomatis membutuhkan model yang cukup akurat untuk membedakan material dengan karakteristik visual serupa, tetapi tetap ringan untuk dipakai pada perangkat dengan sumber daya terbatas. Fokus penelitian Zheng et al. (2023) adalah merancang Focus-RCNet, sebuah arsitektur ringan yang menggabungkan Focus module, sandglass block, SimAM attention, dan KD dari teacher EfficientNet-B4.

Paper melaporkan akurasi 90% untuk Focus-RCNet tanpa KD dan 92% setelah KD. Reimplementation lokal digunakan untuk memeriksa apakah kecenderungan tersebut dapat diamati kembali, sekaligus mengeksplorasi strategi pelatihan yang tidak dibahas secara rinci pada paper.

## 1.2 Tujuan Penelitian

Mereimplementasikan arsitektur utama Focus-RCNet berdasarkan deskripsi paper.

Mengukur kontribusi direct KD dan two-stage KD terhadap student ringan.

Membandingkan Focus-RCNet dengan EfficientNet-Lite0 dalam dua posisi: Lite0 controlled sebagai pembanding sekunder terkontrol dan Lite0 native sebagai pembanding deployment-oriented.

Mengevaluasi profil native EfficientNet-Lite0 untuk skenario deployment-oriented.

Mengidentifikasi faktor yang berpotensi menjelaskan gap hasil reimplementation terhadap paper.

## 1.3 Pertanyaan Penelitian

Apakah KD meningkatkan akurasi student ringan dibandingkan pelatihan cross entropy biasa?

Apakah two-stage KD lebih konsisten dibandingkan direct KD dari inisialisasi awal?

Bagaimana trade-off akurasi, parameter, FLOPs, dan latency pada Focus-RCNet dan EfficientNet-Lite0?

Apakah penyesuaian protokol 200 epoch dan batch size 16 memperkecil gap reimplementation Focus-RCNet terhadap paper?

# 2. Acuan Paper dan Validasi Dataset

## 2.1 Hasil yang Dilaporkan Paper

Paper Focus-RCNet melaporkan teacher EfficientNet-B4 dengan akurasi 97%, Focus-RCNet tanpa KD dengan akurasi 90%, serta Focus-RCNet-KD dengan akurasi 92%. Model student dilaporkan memiliki sekitar 525.802 parameter dan 0,418G FLOPs. Angka tersebut menjadi referensi, bukan target yang dianggap identik secara eksperimental, karena kode sumber dan detail KD lengkap tidak tersedia.

Tabel 2. Referensi performa dari paper Focus-RCNet

| Model | Accuracy | Params | FLOPs | Peran |
| --- | --- | --- | --- | --- |
| EfficientNet-B4 | 97% | 17,559M | 4,490G | Teacher |
| Focus-RCNet | 90% | 0,526M | 0,418G | Student tanpa KD |
| Focus-RCNet-KD | 92% | 0,526M | 0,418G | Student dengan KD |

Sumber: Zheng et al. (2023), Tabel 5.

## 2.2 Validasi Manual Dataset TrashNet

Arsip dataset-resized.zip dari repository TrashNet resmi dihitung ulang secara manual dengan hanya memasukkan file gambar pada path dataset-resized/<kelas>/<gambar>. Hasil validasi menunjukkan 2.527 citra. Kelas plastic berisi 482 citra, bukan 483 citra seperti yang tertulis pada paper. Perbedaan satu citra pada paper kemungkinan merupakan kesalahan penulisan.

Tabel 3. Distribusi kelas pada arsip TrashNet resmi

| Kelas | Jumlah | Train lokal | Validation lokal |
| --- | --- | --- | --- |
| cardboard | 403 | 282 | 121 |
| glass | 501 | 350 | 151 |
| metal | 410 | 287 | 123 |
| paper | 594 | 416 | 178 |
| plastic | 482 | 337 | 145 |
| trash | 137 | 96 | 41 |
| Total | 2.527 | 1.768 | 759 |

Sumber: Perhitungan manual arsip TrashNet resmi dan output split notebook lokal.

Catatan dataset: Paper tidak menyebut secara eksplisit apakah eksperimen memakai archive dataset-resized sekitar 42,8 MB atau archive original sekitar 3,5 GB. Reimplementation menggunakan dataset-resized yang didistribusikan langsung oleh repository resmi.

## 2.3 Perbedaan Protokol terhadap Paper

Reimplementation mengikuti struktur utama paper, tetapi tidak merupakan reproduksi identik. Beberapa parameter paper eksplisit berbeda, sedangkan sejumlah detail implementasi tidak dijelaskan secara lengkap.

Tabel 4. Perbedaan eksplisit antara paper dan reimplementation

| Aspek | Paper | Reimplementation utama | Implikasi |
| --- | --- | --- | --- |
| Dataset | 2.528 citra | 2.527 citra resmi | Paper typo pada kelas plastic |
| Split | 70/30 | Stratified 70/30, seed 42 | Isi validation dapat berbeda |
| Epoch | 200 | 100; follow-up 200 | Durasi optimisasi berbeda |
| Batch size | 16 | 8; follow-up 16 | Dinamika gradient dan BatchNorm berubah |
| Scheduler | Cosine dan redaksi step LR ambigu | CosineAnnealingLR | Kurva LR mungkin berbeda |
| Input pipeline | Deskripsi resize/crop ambigu | Resize langsung 380x380 | Tidak identik |
| AMP | Tidak dijelaskan | Aktif | Presisi numerik berbeda |
| Detail KD | Tidak lengkap | T=4, alpha=0,5, KLDiv x T^2 | Asumsi implementasi lokal |

Sumber: Paper Focus-RCNet dan konfigurasi notebook lokal.

# 3. Metodologi Eksperimen

## 3.1 Dataset dan Preprocessing

Seluruh eksperimen memakai TrashNet dengan enam kelas. Pembagian data utama adalah 70% train dan 30% validation secara stratified. Split awal dibuat dengan seed 42 dan dipertahankan melalui checkpoint teacher untuk eksperimen berikutnya. Run follow-up 200e-bs16 meregenerasi split stratified dengan seed yang sama karena checkpoint teacher tidak di-attach pada sesi Kaggle tersebut.

Pipeline augmentasi train terdiri atas Resize 380x380, horizontal flip, vertical flip, RandomBrightnessContrast, CoarseDropout, normalisasi ImageNet, dan konversi tensor. Pipeline validation hanya memakai Resize 380x380, normalisasi ImageNet, dan konversi tensor.

## 3.2 Model yang Dievaluasi

Tabel 5. Ringkasan arsitektur model

| Model | Input | Params | Inisialisasi | Peran |
| --- | --- | --- | --- | --- |
| EfficientNet-B4 | 380x380 | 17.559.374 | ImageNet | Teacher |
| Focus-RCNet | 380x380 | 520.630 | Random | Student utama |
| EfficientNet-Lite0 controlled | 380x380 | 3.378.694 | Random | Pembanding terkontrol sekunder |
| EfficientNet-Lite0 native | 224x224 | 3.378.694 | ImageNet | Pembanding deployment-oriented |

Sumber: Verifikasi parameter pada notebook eksperimen.

## 3.3 Arsitektur Focus-RCNet

Implementasi Focus-RCNet lokal memakai Focus module untuk memindahkan informasi spasial ke dimensi channel, sandglass block bergaya MobileNeXt, SimAM parameter-free attention setelah setiap stage, convolution 1x1 menuju 512 channel, global average pooling, dropout 0,2, dan fully connected classifier.

Tabel 6. Progresi feature map Focus-RCNet lokal

| Tahap | Output shape | Channel |
| --- | --- | --- |
| Input | 3 x 380 x 380 | 3 |
| Focus | 24 x 190 x 190 | 24 |
| Stage 1 | 48 x 95 x 95 | 48 |
| Stage 2 | 96 x 48 x 48 | 96 |
| Stage 3 | 192 x 24 x 24 | 192 |
| Stage 4 | 384 x 12 x 12 | 384 |
| Conv5 | 512 x 12 x 12 | 512 |

Sumber: Forward-pass verification pada Notebook 2.

## 3.4 Knowledge Distillation

Teacher EfficientNet-B4 dibekukan selama KD. Student dilatih dengan kombinasi soft loss dan hard loss. Implementasi lokal menggunakan persamaan berikut:

L_total = alpha x KLDiv(student/T, teacher/T) x T^2 + (1 - alpha) x CE(student, label)

Konfigurasi default adalah temperature T=4 dan alpha=0,5. Direct KD berarti student langsung dilatih dengan loss gabungan. Two-stage KD berarti student terlebih dahulu dilatih dengan cross entropy, kemudian checkpoint terbaik dilanjutkan melalui fine-tuning KD dengan learning rate lebih kecil.

## 3.5 Matriks Eksperimen

Tabel 7. Matriks eksperimen yang telah dijalankan dan posisi narasi

| ID | Model | Mode | Input | Epoch | Batch | Init. | Posisi narasi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exp 1 | EfficientNet-B4 | Teacher CE | 380 | 100 | 8 | ImageNet | Referensi teacher |
| Exp 2 | Focus-RCNet | Baseline CE | 380 | 100 | 8 | Random | Inti Focus-RCNet |
| Exp 3 | Focus-RCNet | Direct KD | 380 | 100 | 8 | Random | Inti Focus-RCNet |
| Exp 4 | Focus-RCNet | Two-stage KD | 380 | 100+50 | 8 | Random | Inti Focus-RCNet |
| A1 | Focus-RCNet | Direct KD alpha=0,10 | 380 | 100 | 8 | Random | Ablation diagnostik |
| A2 | Focus-RCNet | Direct KD alpha=0,05 | 380 | 100 | 8 | Random | Ablation diagnostik |
| Follow-up | Focus-RCNet | Baseline CE | 380 | 200 | 16 | Random | Follow-up protokol |
| Exp 5 | Lite0 controlled | Baseline CE | 380 | 100 | 8 | Random | Pembanding sekunder |
| Exp 6 | Lite0 controlled | Direct KD | 380 | 100 | 8 | Random | Pembanding sekunder |
| Exp 7 | Lite0 controlled | Two-stage KD | 380 | 100+50 | 8 | Random | Pembanding sekunder |
| Native A | Lite0 native | Baseline CE | 224 | 100 | 8 | ImageNet | Deployment-oriented |
| Native B | Lite0 native | Direct KD | 224 | 100 | 8 | ImageNet | Deployment-oriented |
| Native C | Lite0 native | Two-stage KD | 224 | 100+50 | 8 | ImageNet | Deployment-oriented |
| B1 | Lite0 native | Direct KD alpha=0,10 | 224 | 100 | 8 | ImageNet | Ablation/lampiran |
| B2 | Lite0 native | Direct KD alpha=0,05 | 224 | 100 | 8 | ImageNet | Ablation/lampiran |

Sumber: Notebook 1, 2, 2B, 3, 3B, 3C, dan follow-up 200e-bs16.

# 4. Hasil Eksperimen

## 4.1 Teacher EfficientNet-B4

Teacher EfficientNet-B4 mencapai akurasi validasi terbaik 95,52% pada epoch 61. Teacher ini menjadi sumber soft target bagi seluruh eksperimen KD lokal. Nilainya berada 1,48pp di bawah angka teacher 97% pada paper.

Tabel 8. Hasil teacher lokal

| Model | Best epoch | Val. Acc. | Params | Waktu |
| --- | --- | --- | --- | --- |
| EfficientNet-B4 | 61 | 95,52% | 17.559.374 | 72,1 menit |

Sumber: notebook1_teacher_training.ipynb.

## 4.2 Phase 1: Focus-RCNet Controlled Comparison

Pada Focus-RCNet, direct KD dari random initialization memberikan peningkatan kecil sebesar 0,40pp. Two-stage KD memberikan hasil tertinggi, yaitu 86,30%, atau meningkat 1,06pp dibandingkan baseline.

Tabel 9. Hasil Focus-RCNet pada protokol utama

| ID | Mode | Best epoch | Val. Acc. | Delta vs baseline | Waktu |
| --- | --- | --- | --- | --- | --- |
| Exp 2 | Baseline CE | 71 | 85,24% | - | 23,9 m |
| Exp 3 | Direct KD | 87 | 85,64% | +0,40pp | 37,9 m |
| Exp 4 | Two-stage KD | 31 | 86,30% | +1,06pp | 18,9 m |

Sumber: notebook2-phase1-focus-rcnet.ipynb.

## 4.3 Ablation Alpha pada Focus-RCNet Direct KD

Penurunan alpha dari 0,50 menjadi 0,10 dan 0,05 tidak meningkatkan hasil. Pada implementasi lokal, alpha adalah bobot soft loss teacher. Ketika bobot soft loss diperkecil hingga 0,05, hasil kembali sama dengan baseline CE.

Tabel 10. Ablation alpha Focus-RCNet direct KD

| Run | Alpha | Best epoch | Val. Acc. | Delta vs baseline | Interpretasi |
| --- | --- | --- | --- | --- | --- |
| A2 | 0,05 | 97 | 85,24% | +0,00pp | Manfaat KD hilang |
| A1 | 0,10 | 82 | 85,51% | +0,27pp | Peningkatan kecil |
| Exp 3 | 0,50 | 87 | 85,64% | +0,40pp | Terbaik pada direct KD |

Sumber: notebook2b-focus-rcnet-kd-alpha-ablation.ipynb.

## 4.4 Follow-up Focus-RCNet Baseline: 200 Epoch dan Batch Size 16

Follow-up dijalankan untuk mendekati protokol eksplisit paper. Epoch dan batch size diubah bersamaan dari 100e-bs8 menjadi 200e-bs16. Faktor lain dipertahankan. Hasil terbaik meningkat dari 85,24% menjadi 86,30%, yaitu tambahan delapan prediksi benar pada validation set berisi 759 citra.

Tabel 11. Perbandingan baseline Focus-RCNet sebelum dan sesudah follow-up

| Baseline CE | Epoch | Batch | Best epoch | Val. Acc. | Benar / 759 | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| Protokol awal | 100 | 8 | 71 | 85,24% | 647 | - |
| Follow-up | 200 | 16 | 175 | 86,30% | 655 | +1,06pp |
| Paper | 200 | 16 | - | 90,00% | - | - |

Sumber: Notebook 2, follow-up Kaggle 200e-bs16, dan paper Focus-RCNet.

Interpretasi: Follow-up menunjukkan bahwa durasi dan batch size protokol paper relevan. Namun, peningkatan +1,06pp belum cukup untuk disebut signifikan secara statistik karena baru tersedia satu run per konfigurasi dan prediction-level paired test belum dilakukan.

Kurva follow-up juga menunjukkan gejala overfitting. Validation loss minimum terjadi pada epoch 76 dengan nilai 0,5614, sedangkan akurasi terbaik baru muncul pada epoch 175 dengan validation loss 0,6593 dan train accuracy 99,38%. Model menjadi lebih yakin pada sebagian prediksi yang salah.

## 4.5 Pembanding Sekunder: EfficientNet-Lite0 Controlled

EfficientNet-Lite0 controlled memakai input 380x380 dan random initialization agar protokolnya sejajar dengan Focus-RCNet. Eksperimen ini tetap berguna sebagai kontrol metodologis, tetapi kurang natural untuk EfficientNet-Lite0 karena tidak memakai profil native 224x224 dan ImageNet initialization. Karena itu, hasilnya dipertahankan sebagai pembanding sekunder, bukan klaim utama deployment.

Tabel 12. Hasil EfficientNet-Lite0 controlled sebagai pembanding sekunder

| ID | Mode | Best epoch | Val. Acc. | Delta vs baseline | Waktu |
| --- | --- | --- | --- | --- | --- |
| Exp 5 | Baseline CE | 86 | 76,15% | - | 28,1 m |
| Exp 6 | Direct KD | 92 | 78,26% | +2,11pp | 45,3 m |
| Exp 7 | Two-stage KD | 38 | 79,58% | +3,43pp | 22,6 m |

Sumber: notebook3-phase2-efficientnet-lite0.ipynb.

## 4.6 Follow-up EfficientNet-Lite0 Native

Profil native menggunakan input 224x224 dan ImageNet initialization. Profil ini lebih dekat ke skenario deployment EfficientNet-Lite0. Baseline native mencapai 89,59%. Direct KD dengan alpha 0,50 justru menurunkan hasil menjadi 86,82%, sementara two-stage KD meningkatkan akurasi menjadi 90,91%.

Tabel 13. Hasil EfficientNet-Lite0 native

| Run | Mode | Val. Acc. | F1 | AUC | GFLOPs | Latency | Peak mem. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native A | Baseline CE | 89,59% | 0,8896 | 0,9880 | 0,768 | 7,205 ms | 146,6 MB |
| Native B | Direct KD | 86,82% | 0,8556 | 0,9791 | 0,768 | 7,109 ms | 146,6 MB |
| Native C | Two-stage KD | 90,91% | 0,9053 | 0,9884 | 0,768 | 7,142 ms | 146,6 MB |

Sumber: notebook3b-lite0-native-followup.ipynb.

## 4.7 Ablation Alpha pada Lite0 Native Direct KD

Ablation alpha menunjukkan bahwa penurunan alpha direct KD memulihkan sebagian besar penurunan performa Lite0 native. Alpha 0,10 mencapai hasil yang sama dengan baseline, tetapi tetap belum melampaui two-stage KD. Nilai informasinya bersifat diagnostik: eksperimen ini menjelaskan sensitivitas direct KD, tetapi cukup ditempatkan ringkas atau sebagai lampiran karena tidak mengubah kesimpulan utama.

Tabel 14. Ablation alpha Lite0 native direct KD

| Run | Alpha | Val. Acc. | Precision | Recall | F1 | AUC | Best epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native B | 0,50 | 86,82% | - | - | 0,8556 | 0,9791 | 73 |
| B2 | 0,05 | 89,06% | 0,8842 | 0,8785 | 0,8811 | 0,9858 | 62 |
| B1 | 0,10 | 89,59% | 0,8902 | 0,8866 | 0,8882 | 0,9853 | 74 |

Sumber: notebook3c-lite0-native-direct-kd-alpha-ablation.ipynb.

## 4.8 Ringkasan Controlled Comparison dan Status Narasi

Tabel berikut merangkum tujuh eksperimen dalam protokol controlled comparison dari Notebook 3. Untuk narasi penelitian, baris Focus-RCNet menjadi hasil inti reimplementation, sedangkan baris Lite0 controlled dibaca sebagai pembanding sekunder karena profilnya dibuat sejajar dengan Focus-RCNet, bukan sebagai konfigurasi EfficientNet-Lite0 yang paling natural.

Tabel 15. Controlled comparison dan status narasi

| ID | Model | Mode | Acc. | F1 | AUC | Params | GFLOPs | Latency | Status narasi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | EfficientNet-B4 | Teacher | 95,52% | 0,9434 | 0,9955 | 17.559.374 | 8,784 | 20,291 ms | Referensi teacher |
| 2 | Focus-RCNet | Baseline | 85,24% | 0,8489 | 0,9754 | 520.630 | 0,526 | 6,461 ms | Inti |
| 3 | Focus-RCNet | Direct KD | 85,64% | 0,8433 | 0,9756 | 520.630 | 0,526 | 6,087 ms | Inti |
| 4 | Focus-RCNet | Two-stage | 86,30% | 0,8641 | 0,9734 | 520.630 | 0,526 | 6,488 ms | Inti |
| 5 | Lite0 | Baseline | 76,15% | 0,7490 | 0,9481 | 3.378.694 | 2,246 | 7,364 ms | Sekunder |
| 6 | Lite0 | Direct KD | 78,26% | 0,7617 | 0,9542 | 3.378.694 | 2,246 | 7,078 ms | Sekunder |
| 7 | Lite0 | Two-stage | 79,58% | 0,7890 | 0,9546 | 3.378.694 | 2,246 | 6,970 ms | Sekunder |

Sumber: notebook3-phase2-efficientnet-lite0.ipynb.

# 5. Pembahasan

## 5.1 Two-stage KD Memberikan Hasil Paling Konsisten

Two-stage KD mengungguli direct KD dan baseline pada eksperimen Focus-RCNet dan pada profil Lite0 native. Pada Focus-RCNet, strategi ini meningkatkan hasil sebesar 1,06pp dibandingkan baseline. Pada Lite0 native, strategi ini menaikkan hasil sebesar 1,32pp dan mencapai akurasi student lokal tertinggi, yaitu 90,91%. Lite0 controlled juga menunjukkan pola kenaikan yang sama, tetapi posisinya hanya sebagai pembanding sekunder.

Secara interpretatif, tahap awal cross entropy memberi fondasi representasi kelas yang stabil. KD kemudian berfungsi sebagai fine-tuning regularizer yang memindahkan informasi teacher tanpa mendominasi pembelajaran sejak awal.

## 5.2 Direct KD Sensitif terhadap Profil Student

Direct KD dari random initialization memberikan peningkatan pada Focus-RCNet dan Lite0 controlled, tetapi direct KD dengan alpha 0,50 merugikan Lite0 native yang sudah memiliki ImageNet initialization. Hasil ini menunjukkan bahwa bobot soft target teacher tidak dapat diperlakukan sebagai konstanta universal.

Ablation alpha Lite0 native memperkuat kesimpulan tersebut. Penurunan alpha dari 0,50 menjadi 0,10 menaikkan akurasi dari 86,82% menjadi 89,59%. Namun, nilai ini hanya menyamai baseline dan masih berada di bawah two-stage KD.

## 5.3 Trade-off Akurasi dan Efisiensi

Focus-RCNet lokal memiliki 520.630 parameter, sekitar 33,7 kali lebih kecil daripada teacher EfficientNet-B4. Dalam controlled comparison, Focus-RCNet two-stage KD mencapai 86,30% dengan 0,526G FLOPs dan latency 6,488 ms. Lite0 native two-stage KD mencapai akurasi lebih tinggi, yaitu 90,91%, tetapi membutuhkan sekitar 6,5 kali lebih banyak parameter daripada Focus-RCNet.

Catatan FLOPs: Angka FLOPs lokal tidak boleh dibandingkan langsung dengan paper tanpa menyamakan konvensi penghitungan. Notebook lokal menghitung multiply-add sebagai dua FLOPs, sedangkan paper tampaknya menggunakan konvensi berbeda.

## 5.4 Gap terhadap Paper

Reimplementation Focus-RCNet belum mencapai angka paper. Baseline lokal terbaik setelah follow-up adalah 86,30%, masih 3,70pp di bawah baseline paper 90%. Direct KD lokal terbaik adalah 85,64%, masih berada di bawah Focus-RCNet-KD paper 92%.

Gap tersebut kemungkinan berasal dari kombinasi faktor: preprocessing yang tidak dapat direplikasi persis, kemungkinan perbedaan split, perbedaan sumber resolusi gambar, detail internal arsitektur yang tidak lengkap, detail KD yang tidak dilaporkan, teacher lokal yang lebih lemah 1,48pp, serta scheduler paper yang dijelaskan secara ambigu.

Batas klaim: Belum ada bukti bahwa dataset-resized merupakan penyebab utama gap. Ukuran file archive yang lebih besar tidak otomatis menghasilkan akurasi lebih tinggi. Diperlukan eksperimen terkontrol pada source image original untuk menguji hipotesis tersebut.

## 5.5 Makna Follow-up 200e-bs16

Percobaan follow-up menunjukkan bahwa mendekati protokol paper memberi manfaat praktis. Namun, epoch dan batch size diubah secara bersamaan sehingga kontribusi masing-masing faktor tidak dapat diisolasi. Hasil ini cukup untuk menyimpulkan bahwa protokol gabungan relevan, bukan bahwa batch size 16 atau 200 epoch secara individual merupakan penyebab peningkatan.

# 6. Keterbatasan Penelitian

Seluruh angka performa utama masih berasal dari validation set, bukan independent test set.

Sebagian besar eksperimen dijalankan hanya dengan satu seed, yaitu seed 42.

Prediction-level output belum disimpan secara konsisten sehingga paired significance test belum dapat dilakukan.

Kode sumber paper tidak tersedia; beberapa detail preprocessing, scheduler, KD, dan internal architecture tidak dapat direplikasi identik.

Archive gambar original TrashNet belum diuji secara terkontrol terhadap dataset-resized.

Profil latency diukur pada GPU sesi Kaggle dan tidak langsung mewakili perangkat edge nyata.

# 7. Rekomendasi Eksperimen Lanjutan

Eksperimen berikutnya sebaiknya diprioritaskan berdasarkan nilai informasinya. Grid search besar belum diperlukan karena baseline, direct KD, two-stage KD, dan dua kelompok alpha ablation sudah memberi pola yang cukup jelas. Percobaan yang ada tetap disimpan sebagai arsip, tetapi Lite0 controlled dan Lite0 native alpha ablation tidak perlu diperluas dalam narasi utama kecuali sebagai bukti pendukung.

Eksperimen tambahan yang paling bernilai adalah pengujian pada RealWaste sebagai external validation. RealWaste berbeda dari TrashNet karena berisi citra sampah dari lingkungan landfill autentik, dengan kondisi visual yang lebih mendekati deployment: material bercampur, objek terdeformasi, latar lebih kompleks, dan kelas lebih rinci. Karena itu, eksperimen ini dapat menjawab apakah Focus-RCNet dan strategi KD yang stabil pada TrashNet benar-benar memiliki kemampuan generalisasi lintas domain, atau hanya kuat pada distribusi TrashNet.

Pengujian RealWaste sebaiknya dilakukan bertahap. Tahap pertama memakai kelas yang overlap dengan TrashNet, yaitu cardboard, glass, metal, paper, plastic, dan kemungkinan miscellaneous trash sebagai padanan trash. Tahap kedua memakai seluruh 9 kelas RealWaste dengan final layer baru dan fine-tuning. Jika performa turun tajam pada tahap pertama, klaim penelitian perlu dibatasi sebagai reimplementation pada TrashNet. Jika fine-tuning atau two-stage KD memperbaiki hasil pada RealWaste, maka KD dapat diposisikan sebagai strategi adaptasi domain untuk skenario deployment.

Tabel 16. Prioritas eksperimen lanjutan

| Prioritas | Eksperimen | Tujuan | Keputusan setelah hasil |
| --- | --- | --- | --- |
| 1 | Focus-RCNet direct KD 200e-bs16, T=4, alpha=0,5 | Uji apakah protokol paper juga membantu KD | Bandingkan dengan baseline follow-up 86,30% |
| 2 | Multi-seed baseline vs two-stage KD | Estimasi mean dan standard deviation | Perkuat klaim konsistensi |
| 3 | Independent test split atau cross-validation | Pisahkan model selection dari evaluasi akhir | Laporkan test accuracy |
| 4 | External validation pada RealWaste | Uji generalisasi Focus-RCNet/KD dari TrashNet ke citra landfill nyata | Jika drop besar, batasi klaim ke TrashNet; jika fine-tuning/KD membaik, bahas sebagai domain adaptation |
| 5 | Ablation dataset-original vs dataset-resized | Uji dampak source resolution | Validasi hipotesis dataset |
| 6 | Ekspor prediction-level CSV | Lakukan paired test dan analisis error | Uji signifikansi dan kasus gagal |

Sumber: Sintesis hasil eksperimen lokal, UCI RealWaste, dan Single et al. (2023).

# 8. Kesimpulan

Reimplementation berhasil menguji kembali konsep inti Focus-RCNet dan KD pada TrashNet, tetapi belum mereplikasi angka akurasi paper secara penuh. Focus-RCNet lokal terbaik pada eksperimen utama mencapai 86,30% melalui two-stage KD. Follow-up baseline 200e-bs16 juga mencapai 86,30%, naik 1,06pp dari baseline awal, tetapi masih berada 3,70pp di bawah baseline paper.

Kontribusi eksperimen lokal yang paling kuat adalah temuan bahwa two-stage KD memberikan peningkatan pada Focus-RCNet sebagai student utama dan EfficientNet-Lite0 native sebagai pembanding deployment-oriented. Lite0 controlled tetap mendukung pola yang sama, tetapi sebaiknya dibaca sebagai pembanding sekunder. Direct KD tetap berguna dalam kondisi tertentu, tetapi sensitif terhadap inisialisasi student dan bobot soft loss. Dengan demikian, strategi pelatihan perlu diperlakukan sebagai bagian penting dari desain model ringan, bukan sekadar pengaturan tambahan.

# Lampiran A. Peta Notebook Eksperimen

Tabel 17. Notebook dan perannya

| Notebook | Peran | Output utama | Status narasi |
| --- | --- | --- | --- |
| notebook1_teacher_training.ipynb | Training teacher EfficientNet-B4 | Teacher 95,52% | Referensi teacher |
| notebook2-phase1-focus-rcnet.ipynb | Focus-RCNet baseline, direct KD, two-stage KD | Best 86,30% | Inti |
| notebook2b-focus-rcnet-kd-alpha-ablation.ipynb | Ablation alpha Focus-RCNet direct KD | alpha 0,50 terbaik | Diagnostik ringkas |
| notebook3-phase2-efficientnet-lite0.ipynb | Lite0 controlled dan matriks evaluasi | Matriks 7 eksperimen | Pembanding sekunder |
| notebook3b-lite0-native-followup.ipynb | Profil native Lite0 | Two-stage 90,91% | Deployment-oriented |
| notebook3c-lite0-native-direct-kd-alpha-ablation.ipynb | Ablation alpha Lite0 native | alpha 0,10 memulihkan baseline | Lampiran/diagnostik |
| focus-rcnet-baseline-ce-e200-bs16.ipynb | Follow-up protokol paper | Baseline 86,30% | Follow-up protokol |

Sumber: Folder Results pada workspace penelitian.

# Lampiran B. Referensi Utama

Zheng, D., Wang, R., Duan, Y., Pang, P. C.-I., & Tan, T. (2023). Focus-RCNet: a lightweight recyclable waste classification algorithm based on focus and knowledge distillation. Visual Computing for Industry, Biomedicine, and Art, 6:19. https://doi.org/10.1186/s42492-023-00146-3

Thung, G., & Yang, M. TrashNet dataset repository. https://github.com/garythung/trashnet

Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. Information, 14(12), 633. https://doi.org/10.3390/info14120633

UCI Machine Learning Repository. RealWaste dataset. https://archive.ics.uci.edu/dataset/908/realwaste

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.

Yang, L., Zhang, R.-Y., Li, L., & Xie, X. (2021). SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks. ICML.
