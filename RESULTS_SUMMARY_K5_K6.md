# TrashNet KD — Ringkasan Hasil Final (K5 vs K6)

Studi knowledge distillation untuk klasifikasi sampah TrashNet, dua label space:
**K6** = 6 kelas (cardboard, glass, metal, paper, plastic, trash); **K5** = 5 kelas (`trash` dihapus).
Protokol identik kecuali jumlah kelas (terverifikasi dari config).

**Protokol final (semua eksperimen):** 5 seed (42/123/777/2026/3407), 100 epoch, SGD (momentum 0.9,
weight decay 1e-4), cosine annealing, tanpa early stopping; checkpoint dipilih pada best val accuracy;
test set dievaluasi hanya setelah checkpoint final. Student WasteNet @160px; Focus-RCNet & EfficientNet-B4 @380px.

---

## Tabel 1 — Tangga utama R1–R8 (test set, mean ± sd, 5 seed)

R5/R8 = arm TA-KD (lihat Tabel 3 untuk kontrol CE-nya).

| Exp | Model (≈params K6/K5) | Peran | K6 Acc | K6 F1 | K5 Acc | K5 F1 |
|-----|----------------------|-------|--------|-------|--------|-------|
| R1 | EfficientNet-B4 (17.56M) | Guru besar (batas atas) | 0.9446 ± 0.0070 | 0.9336 | 0.9581 ± 0.0048 | 0.9573 |
| R2 | Focus-RCNet (520.6K/520.1K) | Assistant, CE | 0.8507 ± 0.0164 | 0.8332 | 0.8737 ± 0.0152 | 0.8721 |
| R3 | WasteNet-128K (128.1K/128.1K) | Student kecil, CE (floor) | 0.7652 ± 0.0204 | 0.7371 | 0.8067 ± 0.0200 | 0.8040 |
| R4 | WasteNet-128K | Direct KD ← EffNet | 0.7646 ± 0.0127 | 0.7447 | 0.8061 ± 0.0181 | 0.8037 |
| R5 | WasteNet-128K | TA-KD ← assistant | 0.7578 ± 0.0322 | 0.7277 | 0.8117 ± 0.0200 | 0.8083 |
| R6 | WasteNet-256K (256.0K/255.8K) | Student sedang, CE (floor) | 0.7784 ± 0.0128 | 0.7545 | 0.7994 ± 0.0259 | 0.7974 |
| R7 | WasteNet-256K | Direct KD ← EffNet | 0.7736 ± 0.0178 | 0.7485 | 0.8056 ± 0.0156 | 0.8007 |
| **R8** | WasteNet-256K | **TA-KD ← assistant** | **0.7873 ± 0.0089** | **0.7588** | **0.8235 ± 0.0107** | **0.8219** |

Student terbaik di kedua studi = **R8** (256K TA-KD).

---

## Tabel 2 — Cara melatih assistant Focus-RCNet (test acc, vs CE)

| Varian | K6 Acc | Δ vs CE (p) | K5 Acc | Δ vs CE (p) |
|--------|--------|-------------|--------|-------------|
| R2 ce_baseline | 0.8507 ± 0.0164 | — | 0.8737 ± 0.0152 | — |
| R2.1 direct_kd | 0.8554 ± 0.0231 | +0.47pp (p=0.62) | 0.8732 ± 0.0135 | −0.06pp (p=0.97) |
| R2.2 twostage_kd | 0.8528 ± 0.0078 | +0.21pp (p=0.77) | 0.8765 ± 0.0096 | +0.28pp (p=0.77) |

→ Cara latih assistant tidak berpengaruh signifikan (semua n.s.) → assistant CE (R2) sah dipakai sebagai guru R5/R8.

---

## Tabel 3 — Efek KD (paired test berpasangan per-seed, test acc)

Kontrol berpasangan: tiap baris membandingkan dua arm yang identik kecuali satu faktor.

| Perbandingan | Faktor yang diuji | K6: Δ, p, menang | K5: Δ, p, menang |
|--------------|-------------------|------------------|------------------|
| R4 − R3 | direct KD (128K) vs no-KD | −0.05pp, p=0.94, 2/5 | −0.06pp, p=0.96, 3/5 |
| R5-TA − R5-CE | TA-KD (128K) vs CE-control | −0.74pp, p=0.20, 1/5 | −0.50pp, p=0.23, 1/5 |
| R7 − R6 | direct KD (256K) vs no-KD | −0.47pp, p=0.68, 3/5 | +0.61pp, p=0.68, 4/5 |
| **R8-TA − R8-CE** | **TA-KD (256K) vs CE-control** | **+1.00pp, p=0.024, 5/5** | **+1.40pp, p=0.045, 5/5** |

Konfirmasi McNemar (pooled, n=test×5): **R8 K6 χ²=2.97 p=0.085; R8 K5 χ²=6.06 p=0.014.**
Hanya R8 (256K TA-KD) yang signifikan & konsisten 5/5; sisanya n.s.

---

## Temuan utama

1. **KD dari guru besar (EfficientNet-B4) gagal** — R4 (128K) dan R7 (256K) wash vs baseline no-KD,
   terlepas dari jadwal: single-stage (R4/R7) maupun two-stage (R2.2) sama-sama wash. Ini gejala
   *capacity gap* — guru terlalu besar dibanding student (Cho & Hariharan, ICCV 2019).
2. **TA-KD (guru = assistant menengah Focus-RCNet) menolong — tapi hanya di student 256K** (R8:
   +1.0pp K6 / +1.4pp K5, signifikan, 5/5). Pada student 128K (R5) tetap wash (terlalu kecil/jenuh).
3. **Yang menentukan = identitas/kapasitas guru, bukan jadwal latih.** Two-stage tidak menolong saat
   gurunya tetap EffNet (R2.2 wash); assistant menolong meski hanya saat kapasitas student cukup.
4. **5-vs-6:** K5 (5 kelas) > K6 (6 kelas) di semua baris (+1.3 s.d. +5.4pp test acc; gap F1 lebih besar
   karena kelas `trash` sulit menyeret macro-F1 K6). Deskriptif, bukan uji berpasangan.

## Terminologi (penting, jangan ketuker)
- **two-stage** = *jadwal* (CE dulu → KD fine-tune). Dipakai mekanis di R2.2 dan R5/R8.
- **TA-KD** = *identitas guru* (assistant menengah, bukan guru besar). Hanya R5/R8.
- R5/R8 = two-stage **dan** assistant; yang membuatnya "TA-KD" adalah gurunya = assistant.
- R2.2 = two-stage **tapi** guru = EffNet besar → BUKAN TA-KD (Focus-RCNet di situ = student).

## Posisi metode & rujukan
- Metode inti: **TA-KD** (R5/R8) — Mirzadeh et al., *Improved Knowledge Distillation via Teacher Assistant*, AAAI 2020.
- Baseline KD standar: direct KD (R4/R7) — Hinton, Vinyals, Dean, 2015.
- Bukti capacity-gap (guru besar gagal): Cho & Hariharan, *On the Efficacy of Knowledge Distillation*, ICCV 2019.
- **R4.1/R7.1 (two-stage KD dari EffNet ke WasteNet) di-abort** — prior kuat washing (R4/R7/R2.2);
  inisiatif two-stage dilaporkan sebagai ablation negatif yang memperkuat narasi "guru, bukan jadwal".
