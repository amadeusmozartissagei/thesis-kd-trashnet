# -*- coding: utf-8 -*-
"""Masukan pembimbing (Kholiq Budiman, 2026-08-11): untuk student WasteNet-128K,
asisten seharusnya berkapasitas DI BAWAH 520 ribu parameter milik Focus-RCNet.

Dua sisipan:
  A. BAB 4 Pembahasan, paragraf keterbatasan (paraId 587B6D51) -> tambah kalimat
     rasio kapasitas asisten (2x pada 256K yang berhasil vs 4x pada 128K).
  B. BAB 5 Saran butir 3 (paraId 7F7B944C) -> ditulis ulang jadi spesifik.

Angka yang dipakai (terverifikasi dari RESULTS_SUMMARY_K5_K6.md + memori proyek):
  Focus-RCNet 520,6K ; WasteNet-256K 256.002 ; WasteNet-128K 128.086
  520,6/256,0 = 2,03x   520,6/128,1 = 4,06x
"""
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

DOCX = Path(r"C:\Users\Pratama\Downloads\smpro\sempro-hamza-1.docx")
WORK = Path(__file__).resolve().parent

RPR = ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
       'w:cs="Times New Roman"/>{i}<w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>')
ITAL = "<w:i/><w:iCs/>"


def runs(markup):
    """`*istilah*` -> run miring; sisanya run tegak. Konvensi naskah: student,
    capacity gap, seed miring; hyperparameter & nama model tegak."""
    out = []
    for j, chunk in enumerate(markup.split("*")):
        if not chunk:
            continue
        rpr = RPR.format(i=ITAL if j % 2 else "")
        txt = (chunk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{txt}</w:t></w:r>')
    return "".join(out)


TAMBAHAN_BAB4 = (
    " Keterbatasan berikutnya berkaitan dengan pilihan asisten. Penelitian ini "
    "hanya menguji satu kandidat asisten, yaitu Focus-RCNet dengan sekitar 520 "
    "ribu parameter, dan asisten yang sama digunakan untuk kedua varian "
    "*student*. Akibatnya rasio kapasitas asisten terhadap *student* berbeda "
    "pada kedua lengan, yaitu sekitar dua kali pada WasteNet-256K yang berhasil "
    "dan sekitar empat kali pada WasteNet-128K yang tidak berhasil. Dengan "
    "demikian hasil pada varian WasteNet-128K lebih tepat dibaca sebagai belum "
    "ditemukannya asisten yang jarak kapasitasnya sesuai bagi varian tersebut, "
    "bukan sebagai bukti bahwa varian tersebut tidak dapat menerima distilasi."
)

SARAN_BARU = (
    "Rentang kapasitas asisten perlu ditelusuri lebih lanjut, terutama bagi "
    "*student* yang paling kecil. Penelitian ini baru menguji satu titik "
    "kapasitas asisten, yaitu Focus-RCNet dengan sekitar 520 ribu parameter, "
    "dan jarak kapasitasnya sekitar dua kali terhadap WasteNet-256K tetapi "
    "sekitar empat kali terhadap WasteNet-128K. Oleh karena itu penelitian "
    "selanjutnya disarankan menguji asisten dengan jumlah parameter di bawah "
    "520 ribu bagi *student* WasteNet-128K, misalnya WasteNet-256K yang telah "
    "dilatih pada penelitian ini, agar jarak kapasitasnya setara dengan lengan "
    "yang terbukti berhasil sekaligus menyeragamkan penetapan hyperparameter "
    "antarlengan. Pengujian beberapa titik kapasitas asisten sekaligus akan "
    "memperjelas bentuk kurva *capacity gap* sekaligus menunjukkan titik "
    "kapasitas asisten yang benar-benar optimal."
)


def span(doc, para_id):
    i = doc.find(para_id)
    assert i > 0, f"paraId {para_id} tidak ketemu"
    s = doc.rfind("<w:p ", 0, i)
    e = doc.find("</w:p>", i) + len("</w:p>")
    return s, e


def main():
    assert DOCX.exists(), DOCX
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = WORK / f"sempro-hamza-1_BACKUP_{ts}.docx"
    shutil.copy2(DOCX, backup)
    print("backup ->", backup.name)

    with zipfile.ZipFile(DOCX) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}
    doc = blobs["word/document.xml"].decode("utf-8")

    # --- A. BAB 4 keterbatasan -------------------------------------------
    s, e = span(doc, "587B6D51")
    para = doc[s:e]
    assert "belum dapat dipisahkan sepenuhnya" in para, "paragraf keterbatasan berubah"
    assert "rasio kapasitas asisten" not in para, "sudah pernah dijalankan"
    baru = para[: -len("</w:p>")] + runs(TAMBAHAN_BAB4) + "</w:p>"
    doc = doc[:s] + baru + doc[e:]
    print("A: BAB 4 keterbatasan +", len(TAMBAHAN_BAB4), "karakter")

    # --- B. BAB 5 saran butir 3 ------------------------------------------
    s, e = span(doc, "7F7B944C")
    para = doc[s:e]
    assert "Rentang kapasitas asisten perlu ditelusuri" in para, "butir saran berubah"
    m = re.search(r"</w:pPr>", para)
    assert m, "pPr tidak ketemu"
    baru = para[: m.end()] + runs(SARAN_BARU) + "</w:p>"
    doc = doc[:s] + baru + doc[e:]
    print("B: BAB 5 saran butir 3 ditulis ulang")

    blobs["word/document.xml"] = doc.encode("utf-8")
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    print("tersimpan ->", DOCX)


if __name__ == "__main__":
    main()
