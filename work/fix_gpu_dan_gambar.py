# -*- coding: utf-8 -*-
"""Empat perbaikan pada sempro-hamza-1.docx:

1. BAB 3 (Lokasi dan Waktu Penelitian): sebut spesifikasi GPU Kaggle secara eksplisit.
2. Gambar 4.1: angka total kelas paper tidak lagi tertutup legenda "Uji".
3. Gambar 4.2: anotasi "skema usulan" tidak lagi menindih legenda "K5 (lima kelas)".
4. Gambar 4.6: anotasi "signifikan (p = 0,004)" tidak lagi menindih batang.

Gambar 2-4 diperbaiki di work/make_figures.py lalu byte PNG-nya ditukar di dalam docx.
"""
import zipfile, re, os, shutil, datetime, json
from PIL import Image

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")

# hasil pemetaan hash (work/_figmap.json)
SWAP = json.load(open(os.path.join(HERE, "_figmap.json")))

lock = os.path.join(os.path.dirname(SRC), "~$mpro-hamza-1.docx")
assert not os.path.exists(lock), "Word masih membuka dokumen - tutup dulu."

RPR = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/>')
RPR_I = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
         'w:cs="Times New Roman"/><w:i/><w:iCs/><w:sz w:val="24"/><w:szCs w:val="24"/>')

OLD = (" dan Albumentations, serta dijalankan pada lingkungan Kaggle Notebook "
       "yang menyediakan akselerator GPU.")
NEW_T1 = (" dan Albumentations, serta dijalankan pada lingkungan Kaggle Notebook "
          "dengan sistem operasi Linux. Akselerator yang digunakan adalah satu unit "
          "GPU NVIDIA Tesla T4 bermemori 16 GB, dan seluruh pelatihan memanfaatkan ")
NEW_T2 = (" (AMP). Kaggle sebenarnya menyediakan dua unit GPU Tesla T4 pada satu sesi, "
          "tetapi seluruh eksperimen sengaja dijalankan pada satu GPU saja agar "
          "pengukuran waktu latih dan penggunaan memori tetap sebanding antarskenario.")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")
data = {n: zin.read(n) for n in zin.namelist()}
infos = zin.infolist()
zin.close()

# ---------------------------------------------------------------- 1. teks GPU
assert doc.count(OLD) == 1, "kalimat GPU tidak unik: %d" % doc.count(OLD)
repl = (f'{NEW_T1}</w:t></w:r>'
        f'<w:r><w:rPr>{RPR_I}</w:rPr><w:t>mixed precision</w:t></w:r>'
        f'<w:r><w:rPr>{RPR}</w:rPr><w:t xml:space="preserve">{NEW_T2}')
doc = doc.replace(OLD, repl)
print("1. kalimat spesifikasi GPU ditambahkan")

# ------------------------------------------------------------- 2-4. tukar PNG
for fname, media in SWAP.items():
    path = os.path.join(FIGS, fname)
    old_w, old_h = Image.open(__import__("io").BytesIO(data[media])).size
    with Image.open(path) as im:
        new_w, new_h = im.size
    assert (old_w, old_h) == (new_w, new_h), \
        f"{fname}: ukuran berubah {old_w}x{old_h} -> {new_w}x{new_h} (extent perlu disetel)"
    data[media] = open(path, "rb").read()
    print(f"   {media} <- {fname}  ({new_w}x{new_h})")

data["word/document.xml"] = doc.encode("utf-8")

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in infos:
    zout.writestr(item, data[item.filename])
zout.close()
os.replace(tmp, SRC)
print("\nselesai ->", SRC)
