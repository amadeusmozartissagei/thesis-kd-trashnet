# -*- coding: utf-8 -*-
"""Perbaikan hasil audit BAB 4.

Temuan:
  A. Tujuh gambar tidak pernah dirujuk di dalam teks -> tambahkan kalimat rujukan.
  B. Caption Gambar 4.4 identik dengan caption Tabel 4.5 -> bedakan.
"""
import zipfile, os, shutil, datetime

SRC = r"C:/Users/Pratama/Downloads/smpro/sempro-hamza-1.docx"
HERE = os.path.dirname(os.path.abspath(__file__))

EDITS = [
    # (lama, baru)
    # --- A. rujukan gambar dalam teks -------------------------------------
    ("dengan protokol yang identik pada aspek lainnya.",
     "dengan protokol yang identik pada aspek lainnya. Adapun distribusi jumlah citra "
     "pada setiap kelas dan subset data disajikan pada Gambar 4.1."),

    ("Rangkuman akurasi setiap skenario disajikan pada Tabel 4.3, sedangkan pengujian "
     "signifikansinya dibahas pada subbab 4.4.",
     "Rangkuman akurasi setiap skenario disajikan pada Tabel 4.3 dan divisualkan pada "
     "Gambar 4.2, sedangkan pengujian signifikansinya dibahas pada subbab 4.4."),

    ("sehingga perbaikan yang terukur tidak dapat dikaitkan dengan perbedaan jadwal pelatihan.",
     "sehingga perbaikan yang terukur tidak dapat dikaitkan dengan perbedaan jadwal pelatihan. "
     "Kurva loss dan akurasi selama pelatihan skema tersebut disajikan pada Gambar 4.3."),

    ("kemiripan antarkelas yang terkandung pada distribusi keluaran guru, bukan sekadar dari "
     "efek penghalusan target.",
     "kemiripan antarkelas yang terkandung pada distribusi keluaran guru, bukan sekadar dari "
     "efek penghalusan target. Sebaran selisih F1 pada kedua skema label tersebut divisualkan "
     "pada Gambar 4.4."),

    ("Hal tersebut sekaligus menjelaskan mengapa perbaikan pada skema K5 yang tidak memuat "
     "kelas trash terlihat lebih besar dan lebih bersih.",
     "Hal tersebut sekaligus menjelaskan mengapa perbaikan pada skema K5 yang tidak memuat "
     "kelas trash terlihat lebih besar dan lebih bersih. Pola kesalahan klasifikasi kedua skema "
     "dapat diamati pada Gambar 4.5, yang memperlihatkan bahwa kekeliruan terbanyak memang "
     "terjadi di antara kelas glass, metal, dan plastic."),

    ("meskipun perlu dibaca dengan hati-hati karena variasi tersebut termasuk variasi yang "
     "mengubah jumlah parameter.",
     "meskipun perlu dibaca dengan hati-hati karena variasi tersebut termasuk variasi yang "
     "mengubah jumlah parameter. Perbandingan akurasi seluruh konfigurasi ablasi pada kedua "
     "varian disajikan pada Gambar 4.6."),

    ("melainkan guru yang jarak kapasitasnya terhadap student tidak terlalu jauh dan tidak "
     "terlalu dekat.",
     "melainkan guru yang jarak kapasitasnya terhadap student tidak terlalu jauh dan tidak "
     "terlalu dekat. Pola tersebut divisualkan pada Gambar 4.7."),

    # --- B. bedakan caption gambar dari caption tabel ----------------------
    (" Selisih F1 per Kelas pada Skema TA-KD terhadap Kontrolnya",
     " Perbandingan Selisih F1 per Kelas pada Skema TA-KD antara Skema K6 dan K5"),
]

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bak = os.path.join(HERE, f"sempro-hamza-1_BACKUP_{ts}.docx")
shutil.copy2(SRC, bak)
print("backup ->", bak)

zin = zipfile.ZipFile(SRC, "r")
doc = zin.read("word/document.xml").decode("utf-8")

for old, new in EDITS:
    n = doc.count(old)
    assert n == 1, "cocok %d kali (harus 1): %r" % (n, old[:70])
    doc = doc.replace(old, new, 1)
    print("OK ->", new[-60:].strip())

tmp = SRC + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf-8")
    zout.writestr(item, data)
zin.close(); zout.close()
os.replace(tmp, SRC)
print("\n%d perbaikan diterapkan" % len(EDITS))
