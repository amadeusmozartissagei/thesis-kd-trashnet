# -*- coding: utf-8 -*-
"""Tiga perbaikan pada sempro-hamza-1.docx (disetujui user 2026-08-10):

1. Gambar 2.3 diganti versi baru — panah EfficientNet-B4 -> asisten kini putus-putus
   dan berlabel "hanya diuji sebagai ablasi (tidak signifikan)", sesuai prosa yang
   menyatakan asisten dilatih cross-entropy dan menjadi guru langsung student.
2. BAB 3: klaim ablasi asisten diperbaiki dari "perbedaannya terhadap performa
   student" menjadi "terhadap akurasi asisten" — sebab yang dibuktikan BAB 4
   (Tabel 4.2) memang akurasi asisten, bukan akurasi student.
3. BAB 4 Pembahasan: ditambahkan keterbatasan bahwa hyperparameter lengan 128K
   diimpor dari uji pendahuluan skema K5 sehingga tidak sepadan dengan lengan 256K.
"""
import zipfile, re, os, shutil, datetime, io
from PIL import Image

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs", "gambar_2_3_strategi_distilasi.png")
MEDIA = "word/media/image6.png"          # = Gambar 2.3 (terverifikasi lewat r:embed)

lock = os.path.join(os.path.dirname(SRC), "~$mpro-hamza-1.docx")
assert not os.path.exists(lock), "Word masih membuka dokumen - tutup dulu."

# ---------------------------------------------------------------- teks lama/baru
OLD_BAB3 = ("perbedaannya terhadap performa student tidak signifikan secara "
            "statistik sehingga asisten hasil pelatihan ")
NEW_BAB3 = ("perbedaannya terhadap akurasi asisten tidak signifikan secara "
            "statistik sehingga asisten hasil pelatihan ")

OLD_LIM = ("Selain itu besarnya perbaikan yang diperoleh berkisar pada satu poin "
           "persentase, sehingga penerapannya perlu mempertimbangkan kebutuhan "
           "ketelitian pada kasus penggunaan yang dituju.")
NEW_LIM = OLD_LIM + (
    " Keterbatasan lain terletak pada penetapan hyperparameter distilasi. Laju "
    "pembelajaran dan resolusi masukan asisten pada setiap lengan ditentukan "
    "melalui uji pendahuluan, dan pada varian WasteNet-128K nilai yang digunakan "
    "mengikuti hasil uji pendahuluan skema lima kelas sehingga tidak sepadan "
    "dengan nilai yang digunakan pada varian WasteNet-256K. Dengan demikian "
    "simpulan bahwa distilasi tidak memperbaiki varian WasteNet-128K belum dapat "
    "dipisahkan sepenuhnya dari pengaruh pemilihan hyperparameter tersebut.")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")
data = {n: zin.read(n) for n in zin.namelist()}
infos = zin.infolist()
zin.close()

# ------------------------------------------------------------------- 1. gambar
ps = re.findall(r"<w:p [^>]*>.*?</w:p>", doc, re.S)
target = [p for p in ps if 'r:embed="rId28"' in p]
assert len(target) == 1, "paragraf Gambar 2.3 tidak unik"
m = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', target[0])
cx, cy = int(m.group(1)), int(m.group(2))

old_w, old_h = Image.open(io.BytesIO(data[MEDIA])).size
with Image.open(FIG) as im:
    new_w, new_h = im.size
r_old, r_new, r_ext = old_h / old_w, new_h / new_w, cy / cx
assert abs(r_new - r_ext) < 0.005, f"rasio baru {r_new:.4f} != extent {r_ext:.4f}"
data[MEDIA] = open(FIG, "rb").read()
print(f"1. {MEDIA}: {old_w}x{old_h} (r={r_old:.4f}) -> {new_w}x{new_h} (r={r_new:.4f}); "
      f"extent tetap {cx/914400:.2f}x{cy/914400:.2f} in")

# --------------------------------------------------------------------- 2 & 3
for label, old, new in [("2. BAB 3 klaim ablasi asisten", OLD_BAB3, NEW_BAB3),
                        ("3. BAB 4 keterbatasan hyperparameter", OLD_LIM, NEW_LIM)]:
    assert doc.count(old) == 1, f"{label}: pola cocok {doc.count(old)} kali"
    doc = doc.replace(old, new)
    print(label, "- OK")

data["word/document.xml"] = doc.encode("utf-8")

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in infos:
    zout.writestr(item, data[item.filename])
zout.close()
os.replace(tmp, SRC)
print("\nselesai ->", SRC)
