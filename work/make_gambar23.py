# -*- coding: utf-8 -*-
"""Gambar 2.3 - Strategi Distilasi pada Kurva Capacity Gap (versi diperbaiki).

Perbaikan terhadap versi lama:
1. Panel (b) dulu menggambar EfficientNet-B4 -> asisten dengan panah SOLID, seolah
   asisten didistilasi dari guru besar. Yang benar-benar dijalankan dan dilaporkan:
   asisten Focus-RCNet dilatih cross-entropy lalu menjadi guru langsung student.
   Rantai dua hop ala Mirzadeh hanya diuji sebagai ablasi dan tidak signifikan
   (Tabel 4.2: p = 0,62 dan p = 0,77), sehingga panahnya kini PUTUS-PUTUS + berlabel.
2. Kanvas dikecilkan dari 7,14 in menjadi 6,0 in dengan rasio aspek dipertahankan
   persis (0,6699) supaya huruf tidak menyusut berlebihan setelah dipasang 5,0 in
   di dokumen, sekaligus membuat wp:extent lama tetap berlaku tanpa disetel ulang.

Output: work/figs/gambar_2_3_strategi_distilasi.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

W, H = 1200, 804                     # piksel; 1200/804 = 0,6700 ~ rasio lama 0,6699
DPI = 200
FIGSIZE = (W / DPI, H / DPI)

BLUE_F, BLUE_E = "#dae3f3", "#2f5496"
ORNG_F, ORNG_E = "#ffe6a8", "#bf8f00"
GRN_F, GRN_E = "#e2efda", "#375623"
ARROW = "#404040"
RED, GREEN_T, GREY = "#c00000", "#548235", "#7f7f7f"

plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0)   # y ke bawah
ax.axis("off")


def box(x0, y0, x1, y1, lines, fill, edge):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=6",
                                linewidth=1.6, facecolor=fill, edgecolor=edge,
                                mutation_aspect=1))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    n = len(lines)
    step = 26
    top = cy - (n - 1) * step / 2
    for k, (txt, size, style, color) in enumerate(lines):
        ax.text(cx, top + k * step, txt, ha="center", va="center",
                fontsize=size, style=style, color=color)


def arrow(x0, x1, y, dashed=False):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=GREY if dashed else ARROW,
                                lw=1.6, linestyle=(0, (5, 3)) if dashed else "solid",
                                shrinkA=0, shrinkB=0,
                                mutation_scale=16))


def heading(y, txt):
    ax.text(40, y, txt, ha="left", va="center", fontsize=11.5, fontweight="bold",
            color="#262626")


def caption(y, txt, color):
    ax.text(600, y, txt, ha="center", va="center", fontsize=9.5, style="italic",
            color=color)


B = 10.0     # ukuran teks utama dalam kotak
S = 8.2      # ukuran teks keterangan kecil dalam kotak

# ------------------------------------------------------------------ (a)
heading(44, "(a) Direct KD")
box(120, 74, 470, 176,
    [("Guru besar", B, "normal", "black"),
     ("EfficientNet-B4", B, "normal", "black"),
     ("(≈17,6 jt)", S, "normal", "#404040")], BLUE_F, BLUE_E)
arrow(470, 640, 125)
box(640, 88, 940, 162,
    [("Student", B, "normal", "black"),
     ("WasteNet", B, "normal", "black")], GRN_F, GRN_E)
caption(206, "capacity gap besar — transfer sulit", RED)

# ------------------------------------------------------------------ (b)
heading(280, "(b) Teacher-Assistant KD")
box(40, 362, 300, 464,
    [("Guru besar", B, "normal", "black"),
     ("EfficientNet-B4", B, "normal", "black")], BLUE_F, BLUE_E)
arrow(300, 520, 413, dashed=True)
ax.text(410, 322, "hanya diuji sebagai ablasi", ha="center", va="center",
        fontsize=8.2, style="italic", color=GREY)
ax.text(410, 344, "(tidak signifikan)", ha="center", va="center",
        fontsize=8.2, style="italic", color=GREY)
box(520, 356, 880, 470,
    [("Asisten", B, "normal", "black"),
     ("Focus-RCNet (≈520 rb)", B, "normal", "black"),
     ("dilatih dengan cross-entropy", S, "italic", "#404040")], ORNG_F, ORNG_E)
arrow(880, 1000, 413)
box(1000, 376, 1180, 450,
    [("Student", B, "normal", "black"),
     ("WasteNet", B, "normal", "black")], GRN_F, GRN_E)
caption(500, "gap dijembatani asisten menengah yang menjadi guru langsung student",
        GREEN_T)

# ------------------------------------------------------------------ (c)
heading(576, "(c) Self-Distillation (BAN)")
box(120, 606, 470, 708,
    [("Guru = WasteNet", B, "normal", "black"),
     ("generasi i", B, "normal", "black")], GRN_F, GRN_E)
arrow(470, 640, 657)
box(640, 606, 1000, 708,
    [("Student = WasteNet", B, "normal", "black"),
     ("generasi i+1", B, "normal", "black")], GRN_F, GRN_E)
caption(738, "kapasitas guru = student", GREY)

path = os.path.join(OUT, "gambar_2_3_strategi_distilasi.png")
fig.savefig(path, dpi=DPI)
plt.close(fig)
print("tersimpan ->", path)
