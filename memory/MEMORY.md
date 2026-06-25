# Memory Index

- [K5/K6 terminology](k5-k6-terminology.md) — K6 = 6-class study (final_research_kd/), K5 = 5-class study (final_research_kd_k5/)
- [K5 study progress](k5-study-progress.md) — **K5 COMPLETE (R0–R8).** Headline: TA-KD WORKS on 256K student (R8: +1.4pp vs matched CE-control, paired-t p=0.045, McNemar p=0.014, 5/5 seeds, gains on weak classes) — the ONE positive KD result. NO KD helps 128K (R4/R5 wash); direct big-teacher KD washes even at 256K (R7 p=0.68); only mid-capacity-assistant TA-KD at 256K helps → teacher-assistant capacity-gap thesis confirmed. R8 0.8235 = new best student. Pilot under-called it (seed-42 val tie; 5-seed test revealed it). NEXT: write up.
- [K5 run workflow](k5-r2-run-workflow.md) — Kaggle, one seed per run, edit one plain SEED line; keep it simple
- [K6 harmonization plan](k6-harmonization-plan.md) — Harmonize K6 to K5: R2=CE/R2.1=direct_kd(done)/R2.2=twostage_kd(final TODO T4α0.1); R5/R8 rerun cloned from K5 with R2 CE-baseline assistant (overwrite old direct_kd-assistant finals)
