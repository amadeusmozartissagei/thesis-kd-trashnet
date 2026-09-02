# -*- coding: utf-8 -*-
"""Buang contoh "misalnya WasteNet-256K" dari BAB 5 Saran butir 3.

Frasa itu tambahan sendiri waktu menulis add_saran_asisten.py, BUKAN bagian dari
masukan pembimbing. Masalahnya: WasteNet-256K CE (R6) hanya 0,7784 di K6 dan
justru 0,7994 di K5 -- di bawah student 128K-nya sendiri (0,8067), jadi calon
asisten yang lemah sekaligus memancing pertanyaan yang tidak perlu. Butir saran
dikembalikan ke bentuk sederhana: asisten berparameter lebih kecil dari
Focus-RCNet, tanpa menyebut model tertentu.
"""
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

DOCX = Path(r"C:\Users\Pratama\Downloads\smpro\sempro-hamza-1.docx")
WORK = Path(__file__).resolve().parent
PARA_ID = "7F7B944C"

LAMA = ("menguji asisten dengan jumlah parameter di bawah 520 ribu bagi ")
BARU = ("menguji asisten dengan jumlah parameter yang lebih kecil daripada "
        "Focus-RCNet bagi ")
BUANG = (" WasteNet-128K, misalnya WasteNet-256K yang telah dilatih pada "
         "penelitian ini, agar jarak kapasitasnya setara dengan lengan yang "
         "terbukti berhasil sekaligus")
GANTI = (" WasteNet-128K, sekaligus")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = WORK / f"sempro-hamza-1_BACKUP_{ts}.docx"
    shutil.copy2(DOCX, backup)
    print("backup ->", backup.name)

    with zipfile.ZipFile(DOCX) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    doc = blobs["word/document.xml"].decode("utf-8")

    i = doc.find(PARA_ID)
    assert i > 0, "paraId tidak ketemu"
    s = doc.rfind("<w:p ", 0, i)
    e = doc.find("</w:p>", i) + len("</w:p>")
    para = doc[s:e]

    assert "misalnya WasteNet-256K" in para, "frasa sudah tidak ada"
    assert para.count(LAMA) == 1 and para.count(BUANG) == 1
    baru = para.replace(LAMA, BARU).replace(BUANG, GANTI)
    assert "WasteNet-256K" not in re.sub(r"terhadap WasteNet-256K", "", baru), \
        "masih ada sebutan WasteNet-256K selain kalimat rasio"

    doc = doc[:s] + baru + doc[e:]
    blobs["word/document.xml"] = doc.encode("utf-8")
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    print("tersimpan ->", DOCX)


if __name__ == "__main__":
    main()
