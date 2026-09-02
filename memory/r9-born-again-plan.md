---
name: r9-born-again-plan
description: R9 = Born-Again Networks (BAN) experiment added to the KD study per advisor direction; CORD aborted
metadata:
  type: project
---

**As of 2026-06-26:** the KD study (R0–R8) is no longer closed. Advisor (dosen pembimbing) directed an extra comparison-method arm. The originally-suggested method **CORD was aborted** (too complex) and replaced with **Born-Again Networks (BAN), Furlanello et al., ICML 2018** as experiment **R9**. Advisor's mandate is loose ("nyoba aja") → a wash is a valid, reportable result, not a failure.

**What R9 is:** self-distillation, teacher = WasteNet-256K (same arch as student) from the previous generation, single-stage from-scratch (R7's schedule; only teacher identity differs). Positioned as the same-capacity endpoint of the capacity-gap curve: R7 (big teacher, wash) · **R9 (self teacher, ?)** · R8 (mid assistant, wins). Expectation: ~likely wash on 2.5k imgs + tiny model + weak teacher — which still *reinforces* the [[k6-harmonization-plan]] thesis.

**Notebook built:** `final_research_kd_k5/notebook9_r9_k5_wastenet_256k_born_again.ipynb` (derived from notebook7 R7 direct-KD; into-method code unchanged, only teacher swap + bookkeeping). Generations are presets inside one notebook (`BAN_GENERATION` 1/2), NOT separate R-numbers. gen-1 teacher = R6 CE (exists, free); gen-2 teacher = R9 gen-1 output. Output `runs/final/R9/gen{N}/seed_*`. Default = pilot/seed42/gen1.

**RESULT (R9-K5 BAN gen-1, 5-seed final, DONE):** winner config t4_a0p1 (T=4, alpha=0.1). Test acc **R9 0.8134 ± 0.0109** vs R6 floor 0.7994 ± 0.0259 vs R8 TA-KD 0.8235 ± 0.0107. **R9−R6 = +1.40pp, wins 4/5, paired-t p=0.378 (n.s.)** — non-significance driven by seed 123 (R6 anomalously high 0.838, R9 0.802). Secondary finding: **BAN halves variance** (sd 0.011 vs R6 0.026) = regularization effect. R9 sits below R8 (−1.0pp); R8 stays best student. Narrative: fills capacity-gap curve endpoint — R7 (big teacher) wash p=.68 < R9 (self) +1.4pp directional p=.38 < R8 (assistant) +1.4pp sig p=.045. R9 = clean negative/partial result that reinforces "only mid-capacity assistant clears significance."

**K6 RESULT (gen-1, 5-seed final, DONE):** winner t2_a0p3 (T=2, α=0.3 — differs from K5 t4_a0p1; K6 prefers T=2 like R7-K6). Test acc **R9 0.7831 ± 0.0105** vs R6 floor 0.7784 ± 0.0128 vs R8 0.7873 ± 0.0089. **R9−R6 = +0.47pp, wins 3/5, paired-t p=0.51 → WASH** (much weaker than K5's +1.4pp; `trash` class drags K6 KD as predicted). R9 below R8 (−0.42pp).

**FULL PICTURE both label spaces:** capacity-gap curve clean & monotone — R7 (big teacher) wash p=.68 / R9 (self teacher) K5 +1.4pp p=.38 directional, K6 +0.5pp p=.51 wash / R8 (assistant) K5 +1.4pp p=.045 sig, K6 +1.0pp p=.024 sig. Only mid-capacity assistant clears significance in BOTH. R8 stays best student in both. R9 = clean negative/partial result reinforcing "teacher must be mid-capacity, not big nor self."

**Status: R9 COMPLETE both K5+K6.** RESULTS_SUMMARY_K5_K6.md updated (Tabel 0 design, Tabel 1 ladder R9 row, Tabel 3 R9−R6 paired, Temuan utama point 5, Furlanello ref). Gen-2 + BANE = decided SKIP (wash already; plateau risk). Notebooks: K5 `final_research_kd_k5/notebook9_...`, K6 `final_research_kd/notebook9_...` both final/gen1. Study back to writing phase, now WITH the R9 BAN arm. See [[k6-harmonization-plan]].

**ADVISOR ASK: "masukin statistical test"** (generic, no method named). Study uses paired t-test + Wilcoxon (per-seed) and McNemar continuity-corrected (per-sample, pooled). **McNemar uses continuity-corrected formula (|b-c|-1)²/(b+c)** — confirmed by reproducing R8 exactly (R8 K5 χ²=6.06 p=.014, K6 χ²=2.97 p=.085). **R9 McNemar now DONE & in RESULTS_SUMMARY Tabel 3:** R9−R6 K5 χ²=2.14 p=0.143 (b=147/c=122); K6 χ²=0.20 p=0.654 (b=164/c=155) — both n.s., consistent with paired-t. All KD arms now tested uniformly (paired-t + Wilcoxon + McNemar). 

**Per-class ΔF1 (R9−R6) DONE — RESULTS_SUMMARY Tabel 4b.** Finding: BAN partially mimics R8's mechanism (glass+metal up in BOTH label spaces = same dark-knowledge trace) but LESS targeted — unlike R8, **plastic DROPS in both K5/K6** (−1.44/−1.22; plastic was R8's top gainer), trash rises in K6 (opposite of R8), easy classes drift. Macro ΔF1 +1.41 (K5) / +0.65 (K6) matches accuracy. Mechanistic explanation for R9 n.s.: same-capacity self-teacher carries weaker & noisier dark-knowledge than mid-capacity assistant — gains not concentrated on weak-class trio like R8.

**R9 ANALYSIS FULLY COMPLETE** (acc/f1 5-seed, paired-t, Wilcoxon, McNemar, per-class — both K5+K6, all in RESULTS_SUMMARY, parity with R8). ONLY REMAINING = writing into thesis prose (ready-to-paste paragraphs already drafted in chat). No more computation or runs for BAN.
