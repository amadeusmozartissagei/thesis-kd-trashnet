# TrashNet KD — Ringkasan Hasil Final (K5 vs K6)

Studi knowledge distillation untuk klasifikasi sampah TrashNet, dua label space:
**K6** = 6 kelas (cardboard, glass, metal, paper, plastic, trash); **K5** = 5 kelas (`trash` dihapus).
Protokol identik kecuali jumlah kelas (terverifikasi dari config).

**Protokol final (semua eksperimen):** 5 seed (42/123/777/2026/3407), 100 epoch, SGD (momentum 0.9,
weight decay 1e-4), cosine annealing, tanpa early stopping; checkpoint dipilih pada best val accuracy;
test set dievaluasi hanya setelah checkpoint final. Student WasteNet @160px; Focus-RCNet & EfficientNet-B4 @380px.

---

## Tabel 0 — Desain eksperimen (peta semua percobaan)

Dua sumbu penentu = **guru** (EffNet besar vs assistant menengah) × **ukuran student**. Jadwal (single/two-stage) bukan faktor penentu.

| # | Student (dilatih) | Guru (soft-label) | Jadwal | Peran | Verdict |
|---|-------------------|-------------------|--------|-------|---------|
| R0 | — | — | — | protokol data | — |
| R1 | EfficientNet-B4 | — | CE | guru besar (sumber ilmu) | batas atas ~0.95 |
| R2 | Focus-RCNet | — | CE | assistant (guru untuk R5/R8) | dipakai R5/R8 |
| R2.1 | Focus-RCNet | EffNet-B4 | single (direct) | ablation: cara latih assistant | ≈ R2 (n.s.) |
| R2.2 | Focus-RCNet | EffNet-B4 | two-stage | ablation: cara latih assistant | ≈ R2 (n.s.) |
| R3 | WasteNet-128K | — | CE | floor 128K (+ init R5) | — |
| R4 | WasteNet-128K | EffNet-B4 | single (direct) | baseline KD (Hinton) | wash vs R3 |
| R5 | WasteNet-128K | Focus-RCNet | two-stage | TA-KD 128K | wash vs kontrol |
| R5-CE | WasteNet-128K | — (KD off) | two-stage | kontrol berpasangan R5 | — |
| R6 | WasteNet-256K | — | CE | floor 256K (+ init R8) | — |
| R7 | WasteNet-256K | EffNet-B4 | single (direct) | baseline KD (Hinton) | wash vs R6 |
| **R8** | **WasteNet-256K** | **Focus-RCNet** | **two-stage** | **TA-KD 256K — METODE INTI** | **menang** |
| R8-CE | WasteNet-256K | — (KD off) | two-stage | kontrol berpasangan R8 | — |
| R9 | WasteNet-256K | WasteNet-256K (gen N−1) | single (from-scratch) | BAN self-distillation | directional (K5) / wash (K6) |

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
| R9 | WasteNet-256K | BAN self-KD (gen-1) | 0.7831 ± 0.0105 | 0.7609 | 0.8134 ± 0.0109 | 0.8116 |

Student terbaik di kedua studi = **R8** (256K TA-KD); R9 (BAN self-KD) di posisi kedua, di antara floor R6 dan R8.

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
| R9 − R6 | self-KD (BAN, 256K) vs no-KD | +0.47pp, p=0.51, 3/5 | +1.40pp, p=0.38, 4/5 |

Konfirmasi McNemar (continuity-corrected, pooled n=test×5): **R8 K6 χ²=2.97 p=0.085; R8 K5 χ²=6.06 p=0.014;
R9−R6 K5 χ²=2.14 p=0.143 (b=147/c=122); R9−R6 K6 χ²=0.20 p=0.654 (b=164/c=155).**
Hanya R8 (256K TA-KD) yang signifikan & konsisten 5/5; sisanya n.s. di **semua** uji (paired-t, Wilcoxon, McNemar).
R9 (BAN self-KD) directional di K5 (+1.4pp, 4/5, paired-t p=0.38, McNemar p=0.143) tapi wash di K6
(+0.5pp, paired-t p=0.51, McNemar p=0.654) — sama-besarnya magnitude dengan R8 di K5 tapi tidak konsisten
antar-seed, jadi tak pernah clear signifikansi. R9 vs R6 = kontrol berpasangan yang sah (dua-duanya
WasteNet-256K single-stage from-scratch, beda cuma KD on/off; self-teacher @160 matched-res).

---

## Tabel 4 — R8 per-kelas: di mana TA-KD menolong (ΔF1 = TA-KD − CE-control, mean 5 seed)

**K6 (6 kelas):**

| Kelas | TA-KD F1 | CE-control F1 | ΔF1 (pp) |
|-------|----------|---------------|----------|
| glass | 0.731 | 0.708 | +2.23 |
| metal | 0.742 | 0.725 | +1.68 |
| paper | 0.863 | 0.855 | +0.82 |
| cardboard | 0.901 | 0.895 | +0.58 |
| plastic | 0.754 | 0.751 | +0.33 |
| trash | 0.561 | 0.572 | −1.03 |

**K5 (5 kelas):**

| Kelas | TA-KD F1 | CE-control F1 | ΔF1 (pp) |
|-------|----------|---------------|----------|
| plastic | 0.812 | 0.788 | +2.34 |
| glass | 0.756 | 0.736 | +1.98 |
| paper | 0.885 | 0.873 | +1.23 |
| cardboard | 0.897 | 0.890 | +0.69 |
| metal | 0.760 | 0.754 | +0.68 |

Gain terkonsentrasi di kelas tersulit/paling membingungkan (glass, metal, plastic = trio lemah recyclable),
bukan merata → konsisten dengan transfer *dark knowledge* (kemiripan antar-kelas), bukti efek mekanistik nyata.
`trash` (K6) justru sedikit turun (−1.03pp): catch-all heterogen tanpa struktur soft-label berguna — sekaligus
menjelaskan mengapa K5 (tanpa trash) menunjukkan gain lebih bersih/besar.

## Tabel 4b — R9 (BAN self-KD) per-kelas: ΔF1 = R9 − R6 (CE floor), mean 5 seed

**K5 (5 kelas):**

| Kelas | R9 F1 | R6 F1 | ΔF1 (pp) |
|-------|-------|-------|----------|
| cardboard | 0.903 | 0.877 | +2.60 |
| metal | 0.746 | 0.725 | +2.09 |
| paper | 0.884 | 0.864 | +1.95 |
| glass | 0.754 | 0.735 | +1.87 |
| plastic | 0.771 | 0.786 | −1.44 |

**K6 (6 kelas):**

| Kelas | R9 F1 | R6 F1 | ΔF1 (pp) |
|-------|-------|-------|----------|
| glass | 0.750 | 0.712 | +3.77 |
| trash | 0.603 | 0.589 | +1.45 |
| metal | 0.727 | 0.717 | +1.05 |
| cardboard | 0.887 | 0.893 | −0.58 |
| paper | 0.854 | 0.860 | −0.60 |
| plastic | 0.744 | 0.756 | −1.22 |

Macro ΔF1 = +1.41pp (K5) / +0.65pp (K6), konsisten dengan akurasi. BAN **sebagian** meniru mekanisme R8 —
glass & metal naik di kedua label space (jejak dark-knowledge yang sama) — tapi **kurang tertarget**: berbeda
dari R8, **plastic justru turun di K5 dan K6** (−1.44 / −1.22; di R8 plastic adalah gainer terbesar), `trash`
(K6) malah naik (kebalikan R8), dan kelas mudah ikut bergeser. Jadi guru berkapasitas-sama membawa
*dark knowledge* yang lebih lemah & lebih acak ketimbang assistant menengah — menjelaskan mengapa R9 n.s.
secara agregat: bukan sekadar lebih kecil, tapi gain-nya tidak terkonsentrasi di trio kelas lemah seperti R8.

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
5. **Self-distillation (BAN, R9) tidak cukup.** Guru berkapasitas-sama (WasteNet-256K generasi sebelumnya)
   cuma ngangkat directional di K5 (+1.4pp, 4/5, p=0.38) dan wash di K6 (+0.5pp, 3/5, p=0.51) — tak pernah
   signifikan, walau variance student turun (efek regularisasi BAN). Ini melengkapi kurva *capacity-gap* dari
   sisi guru: guru besar (R7) wash → guru kapasitas-sama (R9) directional/wash → guru menengah/assistant (R8)
   menang. Yang menentukan = guru harus **menengah**, bukan besar maupun diri-sendiri.

## Terminologi (penting, jangan ketuker)
- **two-stage** = *jadwal* (CE dulu → KD fine-tune). Dipakai mekanis di R2.2 dan R5/R8.
- **TA-KD** = *identitas guru* (assistant menengah, bukan guru besar). Hanya R5/R8.
- R5/R8 = two-stage **dan** assistant; yang membuatnya "TA-KD" adalah gurunya = assistant.
- R2.2 = two-stage **tapi** guru = EffNet besar → BUKAN TA-KD (Focus-RCNet di situ = student).

## Posisi metode & rujukan
- Metode inti: **TA-KD** (R5/R8) — Mirzadeh et al., *Improved Knowledge Distillation via Teacher Assistant*, AAAI 2020.
- Baseline KD standar: direct KD (R4/R7) — Hinton, Vinyals, Dean, 2015.
- Bukti capacity-gap (guru besar gagal): Cho & Hariharan, *On the Efficacy of Knowledge Distillation*, ICCV 2019.
- Self-distillation baseline: **BAN (R9)** — Furlanello et al., *Born-Again Neural Networks*, ICML 2018.
  Guru = WasteNet-256K generasi sebelumnya (gen-1: R6 CE; gen-2 belum dijalankan), single-stage from-scratch,
  resolusi guru 160px matched. Winner per label space: K5 t4_a0p1 (T=4, α=0.1), K6 t2_a0p3 (T=2, α=0.3).
  Hasil: directional (K5) / wash (K6), n.s. di dua label space — arahan dosen pembimbing (menggantikan CORD).
- **R4.1/R7.1 (two-stage KD dari EffNet ke WasteNet) di-abort** — prior kuat washing (R4/R7/R2.2);
  inisiatif two-stage dilaporkan sebagai ablation negatif yang memperkuat narasi "guru, bukan jadwal".

---

## Provenance — run-path map (dibuat 2026-07-02; SUMBER ANGKA untuk BAB 4/5)

Tiap angka test-acc di Tabel 1 diverifikasi cocok (5-seed mean) dengan run di path berikut. **Saat menulis hasil, ambil angka dari path ini — JANGAN dari config lain.** `<study>` = `final_research_kd` (K6) / `final_research_kd_k5` (K5); path relatif ke `<study>/runs/final/`.

| Exp | Teacher/Assistant | Run path |
|-----|-------------------|----------|
| R1 | — (CE) | `R1/seed_*` |
| R2 | — (CE) | `R2/ce_baseline/seed_*` |
| R3 | — (CE) | `R3/seed_*` |
| R4 | teacher = EffNet-B4 (R1) | `R4/seed_*` |
| **R5** | **assistant = R2 CE** | **`R5/seed_*/ta_t4_a0p5_lr1em5_av160`** ⚠️ |
| R6 | — (CE) | `R6/seed_*` |
| R7 | teacher = EffNet-B4 (R1) | `R7/seed_*` |
| **R8** | **assistant = R2 CE** | **`R8/seed_*/ta_t4_a0p5_lr1em4_av380`** ⚠️ |
| R9 | teacher = R6 WasteNet-256K CE (self, prev-gen) | `R9/gen1/seed_*` |

### ⚠️ TRAP R5/R8 (wajib diperhatikan saat menulis BAB 4)
Angka R5/R8 yang dilaporkan berasal dari **subfolder `ta_t4_a0p5.../` (assistant = R2 CE)**, BUKAN dari config yang ada langsung di `R{5,8}/seed_*/config_r{5,8}_ta_kd_final_*.json`.
- Di **K6**, config di path canonical itu memakai **assistant TERDISTILASI** (`R2/direct_kd`) dan angkanya beda: K6 R5 = 0.7763 (bukan 0.7578), K6 R8 = 0.7884 (bukan 0.7873). Itu **varian ablation distilled-assistant**, bukan angka yang dilaporkan.
- Di **K5** varian distilled-assistant tidak dijalankan; hanya ada subfolder CE.
- Secara statistik distilled vs CE = wash (lihat Tabel 2), jadi ini murni soal ketepatan sitasi angka, bukan sains.
