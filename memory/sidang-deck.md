---
name: sidang-deck
description: Deck sidang 12 Agustus 2026 — 15 slide utama + 8 cadangan, dibangun ulang dari deck sempro; figur uji berpasangan
metadata:
  type: project
---

**Deck sidang dibuat 2026-08-12** (sidang Rabu 12 Agustus 2026, 10.00 WIB) →
`outputs/sidang-hamza-knowledge-distillation-wastenet.pptx`, **15 slide utama + 8 cadangan**.
Builder: scratchpad `build_sidang.py` (python-pptx **1.0.2 kini TERPASANG** di box ini —
tidak perlu lagi unpack/repack zip manual seperti dicatat di [[reference-pptx-toolchain]]).

**Cara membangunnya:** buka deck sempro sebagai basis (mewarisi tema/font/ukuran), buang 7 slide
rencana (alur presentasi, tiga peran model, prosedur, teknik analisis, evaluasi, kontribusi,
referensi), edit slide yang dipertahankan, tambah slide hasil, lalu susun ulang `sldIdLst`
(+ `prs.part.drop_rel` untuk slide yang dibuang). Slide 18 lama (ΔF1 per kelas) DIPROMOSIKAN
jadi slide utama 11 — pill CADANGAN-nya dicopot.

**Sistem desain deck (diekstrak dari sempro, dipakai ulang di slide baru):** Helvetica Neue;
judul 29,25pt · kicker/label 11,25pt bold #555555 · intro 12pt #555555 · card head 20,25pt bold ·
body 12–16,5pt; kartu #EDEDED, aksen #FF6B35, highlight baris tabel #FFE9E0; judul di (0,44″, 0,38″),
nomor slide di (12,33″, 6,86″), kanvas 13,33×7,5″.

**3 JEBAKAN yang kena saat membangun (jangan terulang):**
1. **Pill CADANGAN tidak pernah tersalin** karena pencariannya pakai substring "cadangan" —
   judul slide cadangan buatanku sendiri mengandung kata itu, jadi dikira pill-nya sudah ada.
   Cari lewat **nama shape `tag`**, bukan teks.
2. **Judul melimpah 2 baris menabrak subjudul/pill.** LibreOffice mensubstitusi Helvetica Neue
   dengan font yang lebih lebar, jadi lebih cepat wrap daripada di PowerPoint asli. Judul slide
   cadangan dibatasi lebar 10,8″ (bukan 12,45″) dan judul panjang dipendekkan.
3. Baris R10/R11 perlu ditambahkan ke tabel skenario (deepcopy `tr` terakhir) supaya cocok dengan
   BAB 3; tinggi baris diturunkan ke 0,355″ agar 12 baris tetap muat.

**Figur baru** di `outputs/figs_sidang/` (generator scratchpad `make_paired_charts.py`):
slope chart berpasangan R8 K6+K5, kontras 5 panel (K6 & K5), dan errorbar-vs-berpasangan.
Bentuknya sengaja slope chart, bukan bar+error bar: sd antar-seed ±0,89 pp > Δ 1,00 pp sehingga
error bar tumpang tindih dan tampak "tidak beda" — jebakan nyata di depan penguji.
Warna = diverging biru #2a78d6 (naik) / merah #e34948 (turun), lolos validator CVD (ΔE 21,6).

**⚠️ CACAT YANG DITEMUKAN USER & SUDAH DIPERBAIKI (2026-08-12):** slide "Alur & strategi distilasi"
warisan sempro menggambar panah SOLID EffNet-B4 → Focus-RCNet dan legenda "TA-KD — EfficientNet-B4
→ Focus-RCNet → WasteNet", yaitu **rantai Mirzadeh yang TIDAK dijalankan** untuk R5/R8 (asisten
dilatih CE). Ini cacat yang sama persis dengan Gambar 2.3 di naskah yang sudah digambar ulang.
Perbaikan: panah dijadikan abu-abu + keterangan "dilatih CE, bukan distilasi", legenda diubah,
naskah slide 6 diberi paragraf pengungkapan proaktif, dan ditambah **slide cadangan 24
"penyimpangan dari Mirzadeh dkk."**. Pelajaran: aset warisan sempro harus dicek ulang terhadap
apa yang BENAR-BENAR dijalankan, bukan hanya ditata ulang.

**Angka rantai Mirzadeh penuh (asisten terdistilasi) di tingkat student, K6, 5 seed:**
R8 256K 0,7884 vs 0,7873 → **+0,11 pp, 2/5, p = 0,83 (wash — jadi penyimpangan ini TIDAK
menciptakan kemenangan R8)**; R5 128K 0,7763 vs 0,7578 → +1,85 pp, 4/5, p = 0,07 (lebih tinggi
tapi belum signifikan; lengan 128K = lengan ber-confound hyperparameter, lihat
[[hyperparameter-confound-128k]]). Sumber: `R{5,8}/seed_*/metrics_*.csv` (level seed_*, BUKAN
subfolder `ta_*`) — lihat "TRAP R5/R8" di RESULTS_SUMMARY.

**Angka yang BARU dihitung untuk deck ini** (belum ada di RESULTS_SUMMARY): **b/c McNemar R8** —
K6 b=64, c=45, χ²=2,97, p=0,085, n=1.895 · K5 b=60, c=35, χ²=6,06, p=0,014, n=1.790. Cocok
dengan χ²/p yang sudah dilaporkan di naskah. Skrip: scratchpad `mcnemar.py`.

**NASKAH BICARA** ditanam ke speaker notes seluruh 23 slide + diekspor ke
`outputs/Naskah_Presentasi_Sidang.md` (generator scratchpad `add_notes.py` — jalankan SESUDAH
`build_sidang.py`, karena builder itu menulis ulang deck dari nol dan menghapus notes).
Anggaran **12:35 dari 15 menit** (buffer 2:25), tiap slide diberi penanda menit kumulatif.
Naskah murni kalau dibaca ≈ 8 menit pada 135 kata/menit — sisanya jeda dan menunjuk grafik.
Slide cadangan diberi catatan "PAKAI BILA ...", bukan naskah.

QA render: `outputs/qa_sidang/` (LibreOffice → PDF → pdftoppm), termasuk `contact_sheet.jpg`.
