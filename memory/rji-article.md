---
name: rji-article
description: Artikel jurnal RJI (English) dari skripsi — lokasi file, scope yang dipilih, dan toolchain build-nya
metadata:
  type: project
---

Artikel untuk Recursive Journal of Informatics dibuat 2026-08-14 dari template
`C:\Users\Pratama\Downloads\smpro\TEMPLATE_RJI_2023.docx`. Output:
`C:\Users\Pratama\Downloads\smpro\artikel-rji-hamza-takd-wastenet.docx`.
Penulis: Hamza Pratama (1, corresponding) + Kholiq Budiman (2).

Keputusan scope dari user: **bahasa Inggris penuh**, dan **hanya tangga utama R1–R8**.
R9 (BAN self-distillation) dan ablasi R10/R11 sengaja TIDAK dimasukkan. K6 jadi angka
utama; K5 cuma satu kalimat robustness check di subbab uji statistik.

**Why:** artikel harus ringkas untuk format jurnal; skripsi lengkap tetap di
[[bab4-writing-status]].

**How to apply:** kalau diminta revisi artikel, jangan pakai docx-js — dokumen dibangun
ulang lewat script generator XML di scratchpad (`build_article.py` + `docxlib.py`,
figur dari `make_figs.py`), yang mengganti body `word/document.xml` template sambil
mempertahankan header/footer branding RJI. Sitasi IEEE bernomor (16 rujukan), bukan
APA seperti skripsi ([[thesis-citations]]).
