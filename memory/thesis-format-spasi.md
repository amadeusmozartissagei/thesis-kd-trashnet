---
name: thesis-format-spasi
description: Aturan spasi skripsi untuk sidang (1,5 + pengecualian) dan cara menerapkannya di sempro-hamza-1.docx
metadata:
  type: project
---

Ketentuan sidang (ditetapkan user 2026-08-12): **spasi baris paragraf isi = 1,5** (line=360, lineRule=auto).
Pengecualian yang dipakai — 1 spasi (line=240): isi tabel, paragraf pembawa gambar, keterangan
tabel/gambar, entri daftar pustaka, ABSTRAK & ABSTRACT. Pola ini identik dengan skripsi acuan
yang sudah lolos sidang (`Downloads/smpro/[REFERENSI]laporan-skripsi-post-sidang-rev-1.docx`) —
pakai dokumen itu sebagai wasit kalau ragu soal format.

**Why:** Ctrl+A lalu set 1,5 di Word menimpa SEMUA paragraf, termasuk tabel/gambar/dapus, sehingga
tabel membengkak dan tiap gambar dapat ruang kosong 50% tinggi gambar (lineRule=auto ikut menskalakan
tinggi baris gambar).

**How to apply:**
- Spasi Daftar Isi/Tabel/Gambar HARUS diset di **style** (TOC1/TOC2/TOC3/TabelGambar di `styles.xml`),
  bukan direct formatting — Word membuang direct formatting entri TOC setiap kali field di-refresh.
- Nomor halaman TOC bisa dihitung ulang tanpa user: Word COM (`Documents.Open` → `StoryRanges.Fields.Update`
  → `TablesOfContents/TablesOfFigures.Update` → `Repaginate` → `Save`), lalu `ExportAsFixedFormat` untuk QA.
  Cek Word tidak sedang buka file dulu (lihat [[feedback-docx-editing]]).
- Di antara akhir field TOC dan heading DAFTAR TABEL ada paragraf `<w:br w:type="page"/>` yang
  **redundan** (field TOC sudah memutus halaman sendiri) → menghasilkan halaman kosong; sudah dihapus.

Kondisi per 2026-08-12: 79–80 halaman, nomor DAFTAR ISI/TABEL/GAMBAR sudah diverifikasi cocok
dengan PDF hasil ekspor. Lihat [[bab4-writing-status]] dan [[thesis-writing-setup]].

**2026-08-18:** paragraf pertama Lampiran 1 Biodata (sel kanan tabel 2 kolom di sebelah pas foto, paraId 030ED129) ternyata masih 1,0 spasi — penguji minta 1,5 seperti paragraf di bawahnya. Sudah diubah ke `line=360 auto` DAN teksnya diringkas (537→398 karakter, "Jawa Tengah" yang berulang dibuang) atas izin user supaya tetap muat sejajar tinggi foto (foto ≈ 2843 twip ≈ 7,9 baris pada 1,5 spasi; hasil akhir pas 8 baris). Catatan: sel tabel memang boleh 1 spasi menurut aturan, tapi biodata ini dibaca sebagai paragraf biasa, jadi ikut 1,5.
