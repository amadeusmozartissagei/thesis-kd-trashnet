# -*- coding: utf-8 -*-
"""Buat gambar untuk BAB 4. Output: work/figs/*.png

Sumber angka: RESULTS_SUMMARY_K5_K6.md (sudah diverifikasi) + split manifest R0
+ artefak run yang sudah ada (confusion matrix, kurva pelatihan).
"""
import os, csv, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K6 = os.path.join(ROOT, "final_research_kd")
OUT = os.path.join(ROOT, "work", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "savefig.dpi": 200,
    "figure.autolayout": True,
})

NAVY, TEAL, GREY, RED = "#1f3864", "#2e75b6", "#a6a6a6", "#c00000"


# --------------------------------------------------------------- Gambar 4.1
def fig_distribusi():
    man = os.path.join(K6, "r0_data_protocol", "split_manifest_seed_42.csv")
    rows = list(csv.DictReader(open(man, newline="")))
    c = collections.Counter((r["split"], r["label"]) for r in rows)
    labs = sorted({r["label"] for r in rows})
    tr = [c[("train", l)] for l in labs]
    va = [c[("val", l)] for l in labs]
    te = [c[("test", l)] for l in labs]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = range(len(labs))
    ax.bar(x, tr, label="Latih", color=NAVY)
    ax.bar(x, va, bottom=tr, label="Validasi", color=TEAL)
    ax.bar(x, te, bottom=[a + b for a, b in zip(tr, va)], label="Uji", color=GREY)
    for i, l in enumerate(labs):
        tot = tr[i] + va[i] + te[i]
        ax.text(i, tot + 12, str(tot), ha="center", fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels(labs)
    ax.set_ylabel("Jumlah citra")
    ax.set_ylim(0, max(tr[i] + va[i] + te[i] for i in range(len(labs))) * 1.10)
    # legenda diletakkan DI LUAR bidang gambar agar tidak menutupi angka total
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(OUT, "gambar_4_1_distribusi_kelas.png"))
    plt.close(fig)
    print("4.1 distribusi kelas")


# --------------------------------------------------------------- Gambar 4.3
SCEN = [
    ("128K\nCE",           0.7652, 0.0204, 0.8067, 0.0200, False),
    ("128K\nKD lgsg",      0.7646, 0.0127, 0.8061, 0.0181, False),
    ("128K\nTA-KD",        0.7578, 0.0322, 0.8117, 0.0200, False),
    ("256K\nCE",           0.7784, 0.0128, 0.7994, 0.0259, False),
    ("256K\nKD lgsg",      0.7736, 0.0178, 0.8056, 0.0156, False),
    ("256K\nTA-KD",        0.7873, 0.0089, 0.8235, 0.0107, True),
    ("256K\nself-KD",      0.7831, 0.0105, 0.8134, 0.0109, False),
]

def fig_perbandingan():
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    x = range(len(SCEN))
    w = 0.38
    k6 = [s[1] for s in SCEN]; e6 = [s[2] for s in SCEN]
    k5 = [s[3] for s in SCEN]; e5 = [s[4] for s in SCEN]
    b1 = ax.bar([i - w / 2 for i in x], k6, w, yerr=e6, capsize=3, label="K6 (enam kelas)",
                color=[RED if s[5] else NAVY for s in SCEN])
    ax.bar([i + w / 2 for i in x], k5, w, yerr=e5, capsize=3, label="K5 (lima kelas)",
           color=[RED if s[5] else TEAL for s in SCEN], alpha=0.75)
    ax.set_xticks(list(x))
    ax.set_xticklabels([s[0] for s in SCEN], fontsize=10)
    ax.set_ylabel("Akurasi data uji")
    ax.set_ylim(0.70, 0.86)
    # legenda di luar bidang gambar agar tidak bertabrakan dengan anotasi "skema usulan"
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.annotate("skema usulan", xy=(5.19, 0.836), xytext=(3.75, 0.851), fontsize=9, color=RED,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    fig.savefig(os.path.join(OUT, "gambar_4_3_perbandingan_skenario.png"))
    plt.close(fig)
    print("4.3 perbandingan skenario")


# --------------------------------------------------------------- Gambar 4.4
def fig_capacity_gap():
    # (nama guru, parameter guru, dAcc K6, dAcc K5, signifikan)
    pts = [
        ("Guru diri sendiri\n(WasteNet-256K)", 256_002,    0.47, 1.40, False),
        ("Asisten menengah\n(Focus-RCNet)",    520_600,    1.00, 1.40, True),
        ("Guru besar\n(EfficientNet-B4)",     17_560_000, -0.47, 0.61, False),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    xs = list(range(len(pts)))
    ax.plot(xs, [p[2] for p in pts], "o-", color=NAVY, lw=1.8, ms=8, label="K6 (enam kelas)")
    ax.plot(xs, [p[3] for p in pts], "s--", color=TEAL, lw=1.8, ms=7, label="K5 (lima kelas)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%s\n%s parameter" % (p[0], format(p[1], ",d").replace(",", "."))
                        for p in pts], fontsize=9)
    ax.set_xlim(-0.35, len(pts) - 0.45)
    ax.set_xlabel("Kapasitas guru (semakin ke kanan semakin besar)")
    ax.set_ylabel("Perubahan akurasi student (poin persentase)")
    ax.set_ylim(-1.2, 2.3)
    # tandai titik signifikan
    ax.scatter([1], [1.00], s=280, facecolors="none", edgecolors=RED, lw=1.8, zorder=5)
    ax.annotate("satu-satunya yang signifikan", xy=(1.08, 1.05), xytext=(1.25, 1.95),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, "gambar_4_4_kurva_capacity_gap.png"))
    plt.close(fig)
    print("4.4 kurva capacity gap")


# --------------------------------------------------------------- Gambar 4.6
def fig_delta_f1():
    kelas = ["glass", "metal", "paper", "cardboard", "plastic", "trash"]
    d6 = [2.23, 1.68, 0.82, 0.58, 0.33, -1.03]
    d5 = [1.98, 0.68, 1.23, 0.69, 2.34, None]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    y = range(len(kelas))
    h = 0.38
    ax.barh([i + h / 2 for i in y], d6, h, color=[NAVY if v >= 0 else RED for v in d6],
            label="K6 (enam kelas)")
    d5p = [0 if v is None else v for v in d5]
    ax.barh([i - h / 2 for i in y], d5p, h,
            color=[TEAL if v >= 0 else RED for v in d5p], alpha=0.8, label="K5 (lima kelas)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(kelas)
    ax.invert_yaxis()
    ax.set_xlabel("Selisih F1 terhadap kontrol (poin persentase)")
    ax.text(0.05, 5 - h / 2, "kelas trash tidak ada pada K5", fontsize=8.5,
            va="center", style="italic", color="#555555")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, "gambar_4_6_delta_f1_per_kelas.png"))
    plt.close(fig)
    print("4.6 delta F1 per kelas")


# --------------------------------------------------------------- Gambar 4.7
def fig_ablasi():
    a128 = [("Baseline", 0.7689, 0.0172, False), ("ReLU", 0.7156, 0.0193, True),
            ("Expand 1,0", 0.7689, 0.0198, False), ("Tanpa\nhidden", 0.7646, 0.0192, False)]
    a256 = [("Baseline", 0.7694, 0.0164, False), ("ReLU", 0.7873, 0.0223, False),
            ("Tanpa\nresidual", 0.7662, 0.0130, False), ("Expand 1,0", 0.7689, 0.0091, False)]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.2), sharey=True)
    for ax, data, title in zip(axes, (a128, a256), ("WasteNet-128K", "WasteNet-256K")):
        x = range(len(data))
        ax.bar(x, [d[1] for d in data], 0.6, yerr=[d[2] for d in data], capsize=3,
               color=[RED if d[3] else NAVY for d in data])
        ax.set_xticks(list(x)); ax.set_xticklabels([d[0] for d in data], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0.66, 0.845)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    axes[0].set_ylabel("Akurasi data uji")
    # anotasi ditempatkan di ruang kosong DI ATAS batang, bukan menindih batang lain
    axes[0].annotate("signifikan\n(p = 0,004)", xy=(1, 0.7156 + 0.021), xytext=(1, 0.806),
                     fontsize=9, color=RED, ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    fig.savefig(os.path.join(OUT, "gambar_4_7_ablasi.png"))
    plt.close(fig)
    print("4.7 ablasi")


# --------------------------------------------------------------- Gambar 4.5
def fig_confusion():
    base = os.path.join(K6, "runs", "final", "R8", "seed_42")
    left = os.path.join(base, "ce_finetune_lr1em4", "confusion_matrix_test.png")
    right = os.path.join(base, "ta_t4_a0p5_lr1em4_av380", "confusion_matrix_test.png")
    for p in (left, right):
        assert os.path.exists(p), p
    a, b = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
    h = min(a.height, b.height)
    a = a.resize((int(a.width * h / a.height), h), Image.LANCZOS)
    b = b.resize((int(b.width * h / b.height), h), Image.LANCZOS)
    pad = 18
    canvas = Image.new("RGB", (a.width + b.width + pad, h), "white")
    canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + pad, 0))
    canvas.save(os.path.join(OUT, "gambar_4_5_confusion_matrix.png"))
    print("4.5 confusion matrix gabungan (kiri: kontrol CE, kanan: TA-KD)")


# --------------------------------------------------------------- Gambar 4.2
def fig_kurva():
    src = os.path.join(K6, "runs", "final", "R8", "seed_42",
                       "ta_t4_a0p5_lr1em4_av380", "training_curves_r8_ta_kd_final_seed_42.png")
    assert os.path.exists(src), src
    im = Image.open(src).convert("RGB")
    # buang panel learning-rate (sepertiga kanan) agar fokus ke loss & akurasi
    im = im.crop((0, 0, int(im.width * 0.665), im.height))
    im.save(os.path.join(OUT, "gambar_4_2_kurva_pelatihan.png"))
    print("4.2 kurva pelatihan (loss & akurasi)")


if __name__ == "__main__":
    fig_distribusi()
    fig_kurva()
    fig_perbandingan()
    fig_capacity_gap()
    fig_confusion()
    fig_delta_f1()
    fig_ablasi()
    print("\nselesai ->", OUT)
