---
name: hyperparameter-confound-128k
description: Lengan TA-KD 128K tidak sebanding dengan 256K (lr & resolusi asisten beda) — diketahui, diputuskan TIDAK di-rerun
metadata:
  type: project
---

**Temuan 2026-08-10** (muncul saat menjawab pertanyaan user soal ablasi Mirzadeh). Lengan TA-KD yang dilaporkan tidak sebanding antar ukuran student:

| | lengan KD | kontrol CE | lr cocok? |
|---|---|---|---|
| 256K (R8) | T4 α0,5 **lr 1e-4** av380 | lr 1e-4 | ya |
| 128K (R5) | T4 α0,5 **lr 1e-5** av160 | lr 1e-4 | **tidak, beda 10×** |

Sebabnya: pilot K6 untuk R5 hanya menyapu lr {5e-4, 1e-3} dengan av380. Kombinasi lr1e-5+av160 tidak pernah ada di grid K6 — **diimpor dari pilot K5** saat harmonisasi (di K5 kombinasi itu menang sah). Akibatnya klaim "TA-KD hanya menolong 256K" ikut terconfound hyperparameter, bukan murni kapasitas.

Bukti pendukung: satu-satunya run 128K K6 di lr5e-4 (varian asisten terdistilasi, `R5/seed_*/` top-level) mengalahkan kontrol +1,11 pp p=0,033. Penyeimbang: di **validasi** ketiga lengan 128K K6 praktis seri (0,8111 / 0,8122 / 0,8116), jadi belum tentu berbalik.

**Yang TIDAK terancam:** R8 menang lawan kontrol pada kedua jenis asisten DAN kedua lr — asisten CE +1,00 pp (p=0,024), asisten terdistilasi +1,11 pp (p=0,042). Temuan utama aman.

**KEPUTUSAN USER 2026-08-10: TIDAK di-rerun.** Notebook Batch A (`notebook5b_r5b_...`), generator `work/build_notebook5b.py`, dan seluruh rencana Batch A/B/C **dihapus atas permintaan user** — jangan dibangun ulang, jangan tawarkan rerun lagi kecuali user yang minta.

**SUDAH DITULIS DI SKRIPSI 2026-08-10** — paragraf keterbatasan BAB 4 Pembahasan kini memuat 3 kalimat yang mengakui ketidaksepadanan hyperparameter lengan 128K vs 256K. Jadi ini bukan lagi risiko tersembunyi.

**Cara memakai catatan ini:** ini keterbatasan yang perlu diakui, bukan pekerjaan yang tertunda. Kalau penguji menanyakan kenapa lr lengan 128K berbeda, jawab apa adanya: hyperparameter tiap lengan dipilih lewat pilot, dan untuk 128K nilainya diimpor dari skema K5 sehingga tidak sepadan dengan lengan 256K — sehingga hasil "128K tidak tertolong" tidak bisa dipisahkan sepenuhnya dari pengaruh hyperparameter. Lihat [[bab4-writing-status]], [[pending-thesis-factcheck]].
