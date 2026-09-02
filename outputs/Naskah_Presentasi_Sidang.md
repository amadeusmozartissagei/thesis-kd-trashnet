# Naskah Presentasi Sidang — Hamza Pratama

Total anggaran bicara **12:55** dari jatah 15 menit, jadi tersisa **2:05** sebagai buffer. Naskah ini juga tertanam di speaker notes tiap slide (Presenter View).

Kolom **selesai** = patokan jam berjalan. Kalau tertinggal lebih dari 1 menit, percepat slide 2–7; jangan pernah memangkas slide 8–10.

| # | Slide | Durasi | Selesai di menit |
|---|-------|--------|------------------|
| 01 | Sampul | 0:20 | 0:20 |
| 02 | Masalah | 0:40 | 1:00 |
| 03 | Celah riset | 0:45 | 1:45 |
| 04 | Rumusan & tujuan | 0:35 | 2:20 |
| 05 | Dataset | 0:40 | 3:00 |
| 06 | Alur & strategi distilasi | 1:10 | 4:10 |
| 07 | Skenario pengujian | 0:45 | 4:55 |
| 08 | Tangga hasil R1–R9 | 1:15 | 6:10 |
| 09 | Uji berpasangan R8 | 1:30 | 7:40 |
| 10 | Capacity gap | 1:10 | 8:50 |
| 11 | ΔF1 per kelas | 0:55 | 9:45 |
| 12 | Ablasi arsitektur | 1:10 | 10:55 |
| 13 | Simpulan | 1:00 | 11:55 |
| 14 | Keterbatasan & saran | 0:50 | 12:45 |
| 15 | Terima kasih | 0:10 | 12:55 |

## Slide 01 — Sampul  ·  0:20  ·  selesai di menit 0:20

Bismillah. Selamat pagi, Bapak dan Ibu dewan penguji.

Perkenalkan, saya Hamza Pratama, NIM 4611422068, dari Program Studi Teknik Informatika. Pada kesempatan ini saya akan memaparkan hasil penelitian skripsi saya yang berjudul "Knowledge Distillation Berbasis Teacher-Assistant untuk Model Ringan WasteNet pada Klasifikasi Citra Sampah TrashNet". Izin memulai.

> **Catatan:** Tarik napas dulu. Jangan langsung ngebut.

## Slide 02 — Masalah  ·  0:40  ·  selesai di menit 1:00

Penelitian ini berangkat dari satu ketegangan praktis.

Pemilahan sampah secara manual masih rentan tidak konsisten, sehingga klasifikasi otomatis dibutuhkan. Masalahnya, model CNN yang akurat umumnya berat, sedangkan perangkat sasaran seperti smart bin atau perangkat tepi memiliki komputasi yang terbatas.

Knowledge distillation menawarkan jalan tengah: memindahkan dark knowledge dari model besar ke model kecil, sehingga model kecil dapat belajar lebih dari sekadar label benar atau salah.

> **Catatan:** Kalimat kunci: akurasi harus bertemu efisiensi.

## Slide 03 — Celah riset  ·  0:45  ·  selesai di menit 1:45

Penelitian terdahulu, khususnya Focus-RCNet oleh Zheng dan kawan-kawan tahun 2023, melaporkan akurasi sekitar 92 persen pada TrashNet dengan distilasi dari EfficientNet-B4.

Namun fokusnya belum pada perbandingan sistematis kapasitas guru untuk student yang sangat ringan. Di sinilah celahnya.

Penelitian ini menguji tiga jenis guru — guru besar, asisten menengah, dan guru berkapasitas sama — secara terkontrol, pada student di rentang 128 sampai 256 ribu parameter, dengan dua ruang label, dan divalidasi secara statistik.

> **Catatan:** Jangan bilang 'penelitian terdahulu kurang'. Bilang 'fokusnya berbeda'.

## Slide 04 — Rumusan & tujuan  ·  0:35  ·  selesai di menit 2:20

Dari celah tersebut disusun tiga rumusan masalah.

Pertama, bagaimana membangun WasteNet ringan untuk klasifikasi citra sampah dengan knowledge distillation.
Kedua, bagaimana pengaruh kapasitas guru terhadap performa student.
Ketiga, apakah perbedaan performanya signifikan dibandingkan baseline tanpa distilasi.

Ketiganya akan saya jawab satu per satu pada bagian simpulan.

> **Catatan:** Janji ini WAJIB ditepati di slide 13 — penguji sering mengecek.

## Slide 05 — Dataset  ·  0:40  ·  selesai di menit 3:00

Data yang dipakai adalah TrashNet: 2.527 citra dalam enam kelas.

Setelah deduplikasi dengan SHA-256, enam duplikat yang berkonflik label dibuang, sehingga tersisa 2.521 citra. Pembagian stratified 70/15/15 dikunci lebih dahulu, sebelum pelatihan apa pun dijalankan.

Selain skema enam kelas, saya juga menguji skema lima kelas tanpa kelas trash, karena kelas trash bersifat heterogen dan jumlahnya paling sedikit, yaitu hanya 137 citra.

> **Catatan:** Kalau Pak Anggyi mendalami data/leakage → lompat ke slide cadangan 16.

## Slide 06 — Alur & strategi distilasi  ·  1:10  ·  selesai di menit 4:10

Ada tiga peran model di sepanjang kurva kapasitas.

EfficientNet-B4, 17,6 juta parameter, sebagai guru besar sekaligus batas atas performa.
Focus-RCNet, 520 ribu parameter, sebagai asisten menengah.
WasteNet 128K dan 256K sebagai student, yaitu target deployment.

Dari ketiganya diuji tiga strategi: distilasi langsung dari guru besar; TA-KD, yaitu distilasi melalui asisten menengah; dan self-distillation, yaitu guru berkapasitas sama dari generasi sebelumnya.

Fungsi kerugiannya menggabungkan cross-entropy dengan KL-divergence yang dibobot alfa dan disuhu T.

Satu hal yang perlu saya sampaikan terus terang. Pada Mirzadeh dan kawan-kawan, asisten lebih dahulu didistilasi dari guru besar. Dalam penelitian ini asisten dilatih dengan cross-entropy, lalu berperan sebagai guru — itu sebabnya panah ini saya gambar abu-abu. Yang saya adopsi adalah gagasan intinya, yaitu guru berkapasitas menengah sebagai jembatan. Rantai penuh versi Mirzadeh tetap saya jalankan sebagai ablasi, dan perbedaannya tidak signifikan.

> **Catatan:** Tunjuk kotak diagramnya satu per satu, jangan baca teksnya.  
> PARAGRAF TERAKHIR JANGAN DILEWATI — sampaikan duluan sebelum ditanya.  
> Kalau didalami → slide cadangan 24: pada 256K selisihnya cuma +0,11 pp (p=0,83), jadi penyimpangan ini tidak menciptakan kemenangan R8.

## Slide 07 — Skenario pengujian  ·  0:45  ·  selesai di menit 4:55

Seluruhnya ada sebelas skenario. R1 sampai R9 membentuk benchmark dari baseline, distilasi langsung, teacher-assistant, hingga self-distillation. R10 dan R11 adalah ablasi komponen arsitektur.

Yang perlu saya tekankan: seluruh skenario memakai pembagian data, jumlah epoch, dan optimizer yang identik — 100 epoch, SGD dengan momentum 0,9, cosine annealing, dan lima seed.

Jadi ketika dua lengan dibandingkan, satu-satunya yang berbeda memang faktor yang sedang diuji.

> **Catatan:** Ini fondasi klaim validitas lo. Ucapkan pelan dan jelas.

## Slide 08 — Tangga hasil R1–R9  ·  1:15  ·  selesai di menit 6:10

Ini hasil lengkapnya, akurasi uji rata-rata dari lima seed.

Guru besar mencapai 0,9446 pada skema enam kelas — itu batas atasnya. Asisten 0,8507. Student baseline tanpa distilasi 0,7652 untuk varian 128K dan 0,7784 untuk 256K.

Student terbaik adalah R8, yaitu WasteNet-256K dengan TA-KD: 0,7873 pada enam kelas dan 0,8235 pada lima kelas.

Satu catatan penting. Angka pada tabel ini tidak sah dibandingkan bebas antar-baris, karena setiap baris memiliki kondisi yang berbeda. Yang sah adalah membandingkan setiap lengan dengan kontrol berpasangannya, dan itulah yang saya tampilkan pada slide berikutnya.

> **Catatan:** JANGAN membacakan seluruh tabel. Sebut 4 angka saja lalu lanjut.

## Slide 09 — Uji berpasangan R8  ·  1:30  ·  selesai di menit 7:40

Ini bukti utama penelitian saya.

Setiap garis mewakili satu seed. Titik di kiri adalah kontrol tanpa distilasi, titik di kanan adalah lengan TA-KD. Kedua lengan identik — pembagian data sama, inisialisasi sama, jumlah epoch sama — yang berbeda hanya distilasi dinyalakan atau dimatikan.

Hasilnya, kelima seed naik tanpa kecuali, dan itu terjadi di kedua skema label. Selisihnya 1,00 poin persen pada enam kelas dengan p sama dengan 0,024, dan 1,40 poin persen pada lima kelas dengan p sama dengan 0,045.

Saya menyadari kenaikan ini tidak besar. Yang saya klaim bukan besarnya, melainkan konsistensinya: lima dari lima seed, pada dua ruang label yang berbeda, dan konsisten di lebih dari satu uji statistik.

> **Catatan:** SLIDE PALING PENTING. Pelan. Paragraf terakhir hafalkan.  
> Kalau ditanya Wilcoxon 0,062 → itu p terkecil yang mungkin untuk n=5, jadi hasil TERBAIK, bukan kegagalan. Slide cadangan 18.

## Slide 10 — Capacity gap  ·  1:10  ·  selesai di menit 8:50

Slide ini merangkum temuan inti penelitian.

Kelima panel memakai sumbu-y yang sama dan diurutkan menurut kapasitas guru.

Panel satu dan tiga memakai guru besar EfficientNet-B4: polanya acak, ada yang naik dan ada yang turun, tidak signifikan.
Panel empat memakai guru berkapasitas sama melalui self-distillation: arahnya positif, tetapi belum konsisten.
Panel lima memakai asisten menengah pada student 256K: kelimanya naik.

Jadi yang menentukan bukan jadwal pelatihannya, melainkan kapasitas gurunya. Guru yang terlalu besar justru gagal — ini gejala capacity gap yang dilaporkan Cho dan Hariharan. Guru harus berkapasitas menengah.

> **Catatan:** Hasil yang 'wash' itu KONTRIBUSI, bukan kegagalan. Sampaikan dengan percaya diri.  
> Kalau lo terdengar defensif di sini, penguji akan mengulik terus.

## Slide 11 — ΔF1 per kelas  ·  0:55  ·  selesai di menit 9:45

Kalau kenaikan tadi hanya kebetulan, sebarannya akan acak antar-kelas. Ternyata tidak.

Gain terkonsentrasi di glass, metal, dan plastic — trio kelas yang paling sering tertukar satu sama lain. Itu justru yang diharapkan apabila yang berpindah memang informasi kemiripan antar-kelas, bukan sekadar derau.

Sebaliknya, kelas trash sedikit menurun, dan hal itu masuk akal: trash adalah kelas buangan yang heterogen, sehingga struktur soft-label dari guru kurang berguna di sana. Ini sekaligus menjelaskan mengapa skema lima kelas menunjukkan gain yang lebih bersih.

> **Catatan:** Ini yang membedakan 'angka naik' dari 'ada mekanisme nyata'.

## Slide 12 — Ablasi arsitektur  ·  1:10  ·  selesai di menit 10:55

Atas masukan pada seminar proposal, saya menambahkan ablasi komponen arsitektur, seluruhnya 40 run.

Hasilnya, satu-satunya komponen yang berpengaruh nyata adalah fungsi aktivasi, dan pengaruhnya bergantung pada kapasitas. Mengganti SiLU dengan ReLU menurunkan varian 128K sebesar 5,33 poin persen dengan p sama dengan 0,004, tetapi pada varian 256K tidak signifikan.

Yang menarik, expand ratio dapat diturunkan sehingga jumlah parameter berkurang drastis tanpa kehilangan akurasi — artinya arsitektur ini masih over-parameter pada sisi tersebut.

Implikasinya bagi temuan utama: karena varian 256K terbukti tahan terhadap seluruh ablasi, kenaikan dari TA-KD tidak dapat dijelaskan oleh kekhususan arsitektur, melainkan memang berasal dari cara pelatihannya.

> **Catatan:** Paragraf terakhir yang paling penting — ablasi menyambung ke temuan utama.  
> Detail per seed ada di slide cadangan 20.

## Slide 13 — Simpulan  ·  1:00  ·  selesai di menit 11:55

Sekarang saya jawab ketiga rumusan masalah tadi.

Pertama, WasteNet ringan berhasil dibangun dan dilatih dengan knowledge distillation — dua varian, 51 dan 84 mega-FLOPs, dengan student terbaik 0,7873 dan 0,8235.

Kedua, kapasitas guru menentukan keberhasilan distilasi: guru besar dan guru berkapasitas sama tidak menolong; hanya asisten menengah yang menaikkan performa student, dan hanya ketika kapasitas student memadai.

Ketiga, peningkatan tersebut signifikan secara statistik dan konsisten pada lima dari lima seed.

Ditambah satu temuan dari ablasi: fungsi aktivasi bersifat kritis hanya ketika kapasitas langka.

> **Catatan:** Sebut 'pertama, kedua, ketiga' dengan tegas — penguji mencocokkan dengan slide 4.

## Slide 14 — Keterbatasan & saran  ·  0:50  ·  selesai di menit 12:45

Saya juga ingin menyampaikan keterbatasan penelitian ini secara terbuka.

Pertama, asisten yang saya pakai berjarak 2,03 kali terhadap student 256K, tetapi 4,06 kali terhadap student 128K. Karena itu hasil pada 128K sebaiknya dibaca sebagai "asisten yang tepat belum ditemukan", bukan "128K tidak dapat didistilasi".

Kedua, learning rate dan resolusi pada lengan 128K mengikuti uji pendahuluan skema lima kelas, sehingga belum sepenuhnya sepadan dengan lengan 256K.

Keduanya sudah saya tuliskan pada BAB 4 dan menjadi dasar saran penelitian lanjutan.

> **Catatan:** Menyebut kelemahan duluan = kredibilitas naik, dan mematikan pertanyaan jebakan.

## Slide 15 — Terima kasih  ·  0:10  ·  selesai di menit 12:55

Demikian pemaparan saya. Terima kasih atas waktu Bapak dan Ibu. Saya persilakan untuk pertanyaan, saran, maupun masukan.

> **Catatan:** Berhenti bicara. Jangan mengisi keheningan.

---

# Slide cadangan (16–23) — kapan dipakai

## Slide 16 — Protokol data & anti-leakage

PAKAI BILA: ditanya dataset, preprocessing, augmentasi, atau kebocoran data.
Poin: split dikunci di Notebook R0 sebelum training; augmentasi hanya di train; test set tidak pernah dipakai memilih model, epoch, T, maupun alfa.

## Slide 17 — Ringkasan uji efek KD

PAKAI BILA: diminta rangkuman seluruh lengan dalam satu tabel.
Poin: hanya R8 yang menang sekaligus konsisten 5/5 seed.

## Slide 18 — Tiga uji statistik + McNemar

PAKAI BILA: ditanya kenapa tiga uji, atau kenapa Wilcoxon tidak signifikan, atau diminta angka McNemar.
Poin 1: unit analisisnya berbeda — paired-t & Wilcoxon per seed, McNemar per sampel.
Poin 2: dengan n=5, p dua sisi terkecil Wilcoxon = 0,0625. Jadi 0,062 adalah hasil TERKUAT yang mungkin, dan muncul justru karena unggul 5/5.
Poin 3: McNemar K5 signifikan (p=0,014), K6 belum (p=0,085) — sampaikan apa adanya, jangan ditutupi.

## Slide 19 — Error bar vs berpasangan

PAKAI BILA: ada yang bilang "simpangan bakunya besar" atau "rentangnya tumpang tindih".
Poin: variasi antar-seed berasal dari perbedaan split, bukan dari efek KD. Desain berpasangan menghilangkan sumber variasi itu — karena itulah uji berpasangan yang dipakai.

## Slide 20 — Ablasi per seed

PAKAI BILA: ditanya detail ablasi, atau kenapa sebagian varian mengubah jumlah parameter.
Poin: hanya ReLU pada 128K yang kalah di SELURUH seed. Varian expand & hidden-layer mengubah parameter sehingga confounded — jumlah parameter selalu dilaporkan berdampingan.

## Slide 21 — Arsitektur & efisiensi

PAKAI BILA: ditanya WasteNet itu arsitektur apa, kernelnya berapa, FLOPs-nya, atau definisi parameter.
Poin: depthwise-separable + inverted residual; kernel 3x3 spasial dan 1x1 antar-kanal. Parameter = bobot + bias + gamma/beta BatchNorm. WasteNet White dkk. (2020) adalah model lain yang kebetulan senama.

## Slide 22 — Validasi silang berulang

PAKAI BILA: diminta menambahkan k-fold cross validation.
Poin: yang dijalankan adalah 5x hold-out berulang dengan split BERBEDA tiap seed = repeated / Monte-Carlo CV, jadi manfaat inti k-fold sudah didapat. Dietterich (1998) justru memperingatkan k-fold + paired-t punya Type I error meningkat dan menganjurkan McNemar — dan McNemar sudah dipakai.

## Slide 23 — Capacity gap K5

PAKAI BILA: diminta bukti bahwa polanya tidak khusus skema enam kelas.
Poin: pola yang sama berulang di lima kelas. Jujur sebutkan ada satu seed yang hasilnya seri pada lengan TA-KD 128K.

## Slide 24 — Penyimpangan dari Mirzadeh

PAKAI BILA: ditanya "asisten Anda dilatih CE, apakah itu masih TA-KD?" atau "kenapa tidak persis Mirzadeh?"
Poin 1: yang diadopsi gagasan intinya — guru berkapasitas menengah sebagai jembatan capacity gap.
Poin 2: cara melatih asisten diuji sebagai ablasi. Di tingkat asisten tidak signifikan (p=0,62 dan p=0,97; varian dua tahap p=0,77).
Poin 3 (paling penting): di tingkat student pada lengan 256K — lengan yang membawa klaim utama — selisihnya hanya +0,11 pp dengan p=0,83. Jadi pilihan ini TIDAK menciptakan kemenangan R8.
Poin 4 (jujur): pada 128K rantai penuh justru +1,85 pp, tetapi 4/5 seed dan p=0,07, belum signifikan — dan lengan 128K memang lengan yang keterbatasan hyperparameter-nya sudah diakui.
