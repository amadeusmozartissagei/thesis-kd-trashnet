---
name: thesis-citations
description: Citation style for the thesis (APA 7, plain text) + primary reference (Focus-RCNet) + APA reference list
metadata:
  type: reference
---

**DAPUS DONE (2026-06-29):** assembled 26 APA-7 entries, alphabetical, hanging indent (left=720 hanging=720), TNR12/sz24 jc=both line=240 after=120. Replaced Eriko's `<w:sdt tag=MENDELEY_BIBLIOGRAPHY>` placeholder ("initial value") between DAFTAR PUSTAKA heading (paraId 0AFB1DA2) and para 1BFA7665. Italics via `*...*`→`<w:i/>`; DOIs kept as plain text. Verified well-formed + repackaged (Python zipfile, no `zip` binary on this box — copy original docx, replace only word/document.xml). Script: scratchpad `gen_dapus.py`. The 26 = exactly the in-text-cited works (extracted from doc): Buslaev, Chicco&Jurman, Cho&Hariharan, Demšar, Dietterich, Furlanello, Gonzalez&Woods, Goodfellow, Hinton, Krizhevsky, LeCun'98, LeCun'15, Mirzadeh, Nair&Hinton, O'Shea&Nash, Pan&Yang, Perez&Wang, Powers, Russakovsky(full 12 authors), Schmidhuber, Shorten&Khoshgoftaar, Sokolova&Lapalme, Sugiyono, Tan&Le, Thung&Yang, Zheng. **Two orphan Eriko pneumonia cites swapped in-text** (user chose "ganti keduanya"): `(Marhasova et al., 2022)`→`(Sugiyono, 2015)` in BAB 3.1; `(Litjens et al., 2017)`→`(Goodfellow dkk., 2016)` in BAB 2. No pneumonia refs remain.

**(superseded) DAPUS DEFERRED plan** (2026-06-29): finish chapters, accumulate cites, assemble once. Now executed (above).

**Citation style = APA 7th edition, handled by me as PLAIN TEXT** (decided 2026-06-29). I write both in-text cites and the DAFTAR PUSTAKA as static APA-formatted text — NOT Mendeley fields (can't drive Mendeley plugin programmatically). Tradeoff: not auto-updating; if Hamza later wants Mendeley-managed bibliography he must re-insert via the plugin. In-text conventions: narrative "Hinton dkk. (2015)" / "Cho dan Hariharan (2019)"; parenthetical "(Mirzadeh dkk., 2020)" with `&` for 2 authors "(Cho & Hariharan, 2019)"; 3+ authors use "dkk." from first cite. See [[thesis-writing-setup]].

**PRIMARY REFERENCE = Focus-RCNet** (Hamza's main reference, the predecessor his study extends; he reimplements Focus-RCNet as the *assistant* and reuses its EfficientNet-B4 teacher):
Zheng, D., Wang, R., Duan, Y., Pang, P. C., & Tan, T. (2023). Focus-RCNet: A lightweight recyclable waste classification algorithm based on focus and knowledge distillation. *Visual Computing for Industry, Biomedicine, and Art, 6*(1), Article 19. https://doi.org/10.1186/s42492-023-00146-3
— TrashNet, 6 classes, KD from EfficientNet-B4 teacher, best acc ~92% (baseline 88.07% → +Focus 91.20% → +SimAM 92.20% → KD-refined). Now lead row in Tabel 1.1 + lead related-work para in BAB 1 Kebaruan.

**WEB-VERIFIED 2026-06-30 (PMC10567611):** the **92.20% is the SimAM ablation step (the single highest accuracy, BEFORE KD)**, NOT a post-KD number. KD from EfficientNet-B4 is used mainly to COMPRESS parameters (paper: "improve performance ~2% vs original design"). So don't write "KD → 92.2%"; write "Focus-RCNet best acc ~92.2% (Focus+SimAM); KD compresses the model." BAB 2 sempro was corrected accordingly (see [[pending-thesis-factcheck]]).

**APA 7 reference list for BAB 1 method/anchor refs (verify/complete in DAFTAR PUSTAKA):**
- Cho, J. H., & Hariharan, B. (2019). On the efficacy of knowledge distillation. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4794–4802.
- Furlanello, T., Lipton, Z. C., Tschannen, M., Itti, L., & Anandkumar, A. (2018). Born-again neural networks. *Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80*, 1607–1616.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.
- Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., & Ghasemzadeh, H. (2020). Improved knowledge distillation via teacher assistant. *Proceedings of the AAAI Conference on Artificial Intelligence, 34*(04), 5191–5198.
- Thung, G., & Yang, M. (2016). *Classification of trash for recyclability status* [CS229 project report]. Stanford University.
- Zheng, D., Wang, R., Duan, Y., Pang, P. C., & Tan, T. (2023). [see PRIMARY REFERENCE above].

**WASTE-ISSUE VALIDATION refs added to BAB 1 §1.1 opening para (2026-07-14):** Hamza flagged the "issue sampah" opening had no sources. Added 2 in-text cites to the first Latar Belakang para (paraId 2AFFF845, fully rebuilt) + 2 APA-7 DAPUS entries (spliced after Hinton, before Krizhevsky). Global fig = What a Waste 2.0 (Kaza dkk., 2018): ~2,01 miliar ton MSW 2016 → 3,40 miliar ton 2050, ~13,5% didaur ulang (web-verified World Bank). Indonesia = SIPSN KLHK: ">50 juta ton/tahun" (hedged vs 56.63/64/69.9 juta variants across years). Both REAL:
- Kaza, S., Yao, L. C., Bhada-Tata, P., & Van Woerden, F. (2018). *What a waste 2.0: A global snapshot of solid waste management to 2050*. World Bank. https://doi.org/10.1596/978-1-4648-1329-0
- Kementerian Lingkungan Hidup dan Kehutanan. (2023). *Sistem Informasi Pengelolaan Sampah Nasional (SIPSN): Capaian kinerja pengelolaan sampah*. https://sipsn.menlhk.go.id/
DAPUS now = 28 entries. Script: scratchpad `add_waste_src.py`. User must Ctrl+A/F9 in Word if he touches DAPUS ordering (entries are static text, already alphabetical).

**WASTENET ARCHITECTURE + NAME-DISAMBIGUATION refs added (2026-07-16):** advisor/examiner concern — "WasteNet ini jenis apa, referensinya mana?" The doc had ZERO description of the student's architecture basis (no depthwise/MobileNet/EfficientNet-as-student). Added a body paragraph in BAB 3 right AFTER the WasteNet-description para (paraId 33288ED7, before the "Knowledge Distillation" subheading): WasteNet = self-designed depthwise-separable / inverted-residual CNN (ConvBNAct stem → DepthwiseSeparableBlock stages, expand 1×1 → dw 3×3 → project 1×1, BN+SiLU, residual when in==out, GAP→FC head; ~256,002 params), citing MobileNet/EfficientNet as design *principles* (not a taken model), and explicitly stating the name coincides with WasteNet (White dkk., 2020, DenseNet-based) but is NOT derived from it. Key point for defense: architecture is ORIGINAL (built from scratch in notebook9), name collision only. Added 3 APA-7 DAPUS entries (spliced via DOI anchors, alphabetical, hanging indent + italic like siblings): Howard dkk. (2017) MobileNet [arXiv:1704.04861]; Sandler dkk. (2018) MobileNetV2 [CVPR]; White dkk. (2020) WasteNet [arXiv:2006.05873 — added so the in-text disambiguation cite isn't orphan]. EfficientNet (Tan & Le, 2019) already present, reused as student-design anchor. **DAPUS now = 31 entries.** Script: scratchpad `add_arch.py`. Verified well-formed + orders: Hinton→Howard→Kaza, Russakovsky→Sandler→Schmidhuber, Thung→White→Zheng. NOTE (web-verified): White's WasteNet reports NO explicit param count but is DenseNet-based (≈tens of millions); best acc 0.970 on their 6-class set, Jetson Nano. See [[thesis-title-final]] for the naming decision.

**BAB 2 additional refs used (verify/add to DAPUS — many carried over from Eriko's doc, all REAL):**
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97*, 6105–6114. [teacher backbone — NEW]
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NeurIPS), 25*.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553), 436–444.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.
- Schmidhuber, J. (2015). Deep learning in neural networks: An overview. *Neural Networks, 61*, 85–117.
- O'Shea, K., & Nash, R. (2015). An introduction to convolutional neural networks. *arXiv preprint arXiv:1511.08458*.
- Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *Proceedings of the 27th International Conference on Machine Learning (ICML)*, 807–814.
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering, 22*(10), 1345–1359.
- Russakovsky, O., dkk. (2015). ImageNet large scale visual recognition challenge. *International Journal of Computer Vision, 115*(3), 211–252.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*(1), 60.
- Perez, L., & Wang, J. (2017). The effectiveness of data augmentation in image classification using deep learning. *arXiv preprint arXiv:1712.04621*.
- Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. *Information, 11*(2), 125.
- Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4th ed.). Pearson.
- Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management, 45*(4), 427–437.
- Powers, D. M. W. (2011). Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation. *Journal of Machine Learning Technologies, 2*(1), 37–63.
- Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics, 21*(1), 6.
- Thung & Yang (2016), Dietterich (1998), Demšar (2006) — already listed above.
